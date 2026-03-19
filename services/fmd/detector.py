# services/fmd/detector.py
# Failure Memory Deficiency — weight 0.20
# LLMs optimize for correctness and best practices.
# They don't naturally produce: specific library version that had a bug +
# workaround implemented + thing they'd do differently.
# Real operational experience has war stories. Fraud doesn't.

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kive.shared.schemas import FlagDetail, ProbeSuggestion, SignalRequest

WEIGHT = 0.20

# Regex patterns detecting failure narrative components
# Pattern: negation/failure_word + context + version/specificity
_FAILURE_PATTERNS = [
    # Direct failure words with past context
    re.compile(
        r"\b(never|wouldn't|stopped|dropped|abandoned|regret|bad|bug|broke|crashed|"
        r"failed|hung|leaked|corrupted|lost|deleted|wiped|killed|deadlock|timeout)\b"
        r".{0,80}"
        r"\b(version|v\d+|\d+\.\d+|library|package|framework|api|config|migration)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    # "Had to rewrite/replace/migrate X because Y"
    re.compile(
        r"\b(had to|needed to|ended up|eventually|finally|forced to)\b"
        r".{0,100}"
        r"\b(rewrite|replace|migrate|abandon|stop using|switch|refactor|rollback)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    # Lesson learned / retrospective framing
    re.compile(
        r"\b(lesson|learned|wouldn't do|do differently|mistake|regret|in retrospect|"
        r"looking back|next time|that's when I realized|cost us|took .{0,20} (hours|days|weeks))\b",
        re.IGNORECASE,
    ),
    # Specific version references in past tense
    re.compile(
        r"\b(v\d+\.\d+|\d+\.\d+\.\d+|version \d+|pandas \d|pytorch \d|k8s 1\.\d+|"
        r"react \d|node \d+|python 3\.\d+)\b"
        r".{0,150}"
        r"\b(bug|issue|problem|broke|crash|fail|memory|leak|deadlock|silent|dropped|wrong)\b",
        re.IGNORECASE | re.DOTALL,
    ),
]

# Curated failure narrative embedding anchors (text similarity targets)
FAILURE_REFERENCE_TEXTS = [
    "I caused a production outage by running a migration without testing it on a clone first",
    "Pandas merge silently dropped rows with NaN keys and we didn't catch it for days",
    "We had a memory leak in our PyTorch training loop that only showed up after 50 epochs",
    "Redis evicted session tokens under high load because I chose the wrong eviction policy",
    "I deployed to prod without a rollback plan and had to manually restore from backup",
    "The liveness probe was misconfigured and took down all replicas during a rolling update",
    "I used the wrong index type in Postgres and a query that ran in 10ms took 45 seconds under load",
    "We lost 3 months of soft-deleted records to a migration that ran against prod instead of staging",
]


@dataclass
class FMDResult:
    score: float           # [0,1] — high = no failure memory = fraud signal
    confidence: float
    pattern_matches: int   # How many failure patterns were found
    flags: list[FlagDetail] = field(default_factory=list)
    probe_suggestion: Optional[ProbeSuggestion] = None


class FMDDetector:
    """
    Failure Memory Deficiency detector.

    Scans screening answers for failure narrative patterns:
    negation + past-tense + specific context + version reference.

    Absence of ALL patterns across ALL answers = strong fraud signal.
    """

    def __init__(self):
        self._embedder = None
        self._reference_embeddings = None
        self._initialized = False

    async def initialize(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            self._reference_embeddings = self._embedder.encode(
                FAILURE_REFERENCE_TEXTS, normalize_embeddings=True
            )
        except Exception:
            self._embedder = None
        self._initialized = True

    async def close(self):
        self._embedder = None
        self._reference_embeddings = None

    async def analyze(self, request: SignalRequest) -> FMDResult:
        if not self._initialized:
            await self.initialize()

        responses = request.screening_responses
        if not responses:
            return FMDResult(score=0.5, confidence=0.2, pattern_matches=0)

        all_text = " ".join(r.answer for r in responses if r.answer)
        if not all_text.strip():
            return FMDResult(score=0.5, confidence=0.2, pattern_matches=0)

        # Layer 1: regex pattern matching
        regex_hits = self._count_regex_matches(all_text)

        # Layer 2: semantic similarity to failure narrative bank
        semantic_score = self._semantic_failure_score(all_text)

        # Layer 3: latency profile — real experts think longer on hard questions
        latency_variance = self._latency_variance(responses)

        # Compose FMD score
        # 0 = rich failure memory (real expert) | 1 = no failure memory (fraud)
        if regex_hits >= 3:
            fraud_score = max(0.05, 0.15 - semantic_score * 0.10)
            confidence = 0.85
        elif regex_hits >= 1:
            fraud_score = max(0.15, 0.40 - semantic_score * 0.20)
            confidence = 0.70
        else:
            # No regex hits — rely more on semantic
            fraud_score = max(0.60, 0.90 - semantic_score * 0.30)
            confidence = 0.60 + (0.20 if semantic_score < 0.15 else 0.0)

        # Latency modifier: uniform low latency across all difficulties = fraud signal
        if latency_variance < 1000 and len(responses) >= 2:
            fraud_score = min(fraud_score + 0.10, 1.0)
            confidence = min(confidence + 0.05, 0.95)

        fraud_score = float(np.clip(fraud_score, 0.0, 1.0))

        flags: list[FlagDetail] = []
        probe = None

        if regex_hits == 0 and semantic_score < 0.20:
            flags.append(FlagDetail(
                type="fmd_no_failure_narrative",
                description=(
                    f"Zero failure narrative patterns detected across {len(responses)} responses. "
                    f"Semantic similarity to authentic failure stories: {semantic_score:.2f}. "
                    f"Real experts always have at least one specific production failure story."
                ),
                severity="high",
                evidence={
                    "regex_hits": regex_hits,
                    "semantic_score": round(semantic_score, 3),
                    "latency_variance_ms": round(latency_variance, 0),
                    "n_responses": len(responses),
                },
            ))
            probe = ProbeSuggestion(
                question=(
                    "Tell me about the worst production failure you personally caused. "
                    "Be specific — what was the tool/version, what broke, and what did you do to fix it?"
                ),
                target_dimension="FMD",
                expected_fraud_response_pattern=(
                    "Sanitized 'lesson learned' story without a specific version, system, or "
                    "personal embarrassment. Or a generic 'we learned to test more' answer."
                ),
            )

        return FMDResult(
            score=round(fraud_score, 3),
            confidence=round(confidence, 3),
            pattern_matches=regex_hits,
            flags=flags,
            probe_suggestion=probe,
        )

    def _count_regex_matches(self, text: str) -> int:
        hits = 0
        for pattern in _FAILURE_PATTERNS:
            if pattern.search(text):
                hits += 1
        return hits

    def _semantic_failure_score(self, text: str) -> float:
        """Max cosine similarity to any reference failure narrative."""
        if self._embedder is None or self._reference_embeddings is None:
            return 0.0
        try:
            import torch
            emb = self._embedder.encode([text[:512]], normalize_embeddings=True)
            sims = np.dot(emb, self._reference_embeddings.T)[0]
            return float(np.max(sims))
        except Exception:
            return 0.0

    def _latency_variance(self, responses: list) -> float:
        """Variance of response latency in ms. Low variance across difficulties = fraud."""
        latencies = [r.latency_ms for r in responses if r.latency_ms and r.latency_ms > 0]
        if len(latencies) < 2:
            return 99999.0  # Can't determine — give benefit of doubt
        return float(np.var(latencies))
