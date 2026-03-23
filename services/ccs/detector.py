# services/ccs/detector.py
# Cross-Candidate Similarity — weight: contextual (population-level signal)
# Detects template-based fraud by measuring cosine similarity of screening
# answers against a rolling index of all previous candidate submissions.
# When multiple candidates submit near-identical answers to the same question,
# it indicates a shared prompt template is circulating.
# Secondary value: flags compromised questions for rotation.

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kive.shared.schemas import FlagDetail, ProbeSuggestion, SignalRequest

WEIGHT = 0.10  # Lower weight — population signal, not individual

# Similarity thresholds
SIMILARITY_THRESHOLD = 0.85        # Cosine similarity for cluster detection
MIN_CLUSTER_SIZE = 3               # Need 3+ near-identical answers to flag
CLUSTER_WINDOW_DAYS = 30           # Only compare within rolling window
NGRAM_SIZE = 3                     # Character n-gram size for shingling


@dataclass
class CCSResult:
    score: float           # [0,1] — high = template cluster detected
    confidence: float
    max_similarity: float = 0.0
    cluster_size: int = 0
    flagged_questions: list[str] = field(default_factory=list)
    flags: list[FlagDetail] = field(default_factory=list)
    probe_suggestion: Optional[ProbeSuggestion] = None


class CCSDetector:
    """
    Cross-Candidate Similarity detector.

    Maintains an in-memory rolling index of answer shingle-hashes per question.
    On each new submission, computes Jaccard similarity against stored answers.
    If similarity > threshold for 3+ candidates, flags as template cluster.

    In production, this would be backed by a Redis or Postgres store.
    For training/evaluation, operates on the in-memory store.
    """

    def __init__(self):
        # question_id -> list of (candidate_id, shingle_set)
        self._answer_index: dict[str, list[tuple[str, set[int]]]] = defaultdict(list)
        self._initialized = False

    async def initialize(self):
        self._initialized = True

    async def close(self):
        self._answer_index.clear()

    async def analyze(self, request: SignalRequest) -> CCSResult:
        if not self._initialized:
            await self.initialize()

        responses = request.screening_responses
        candidate_id = request.candidate_id

        if not responses:
            return CCSResult(score=0.10, confidence=0.10)

        flags: list[FlagDetail] = []
        max_similarity = 0.0
        max_cluster_size = 0
        flagged_questions: list[str] = []
        per_question_scores: list[float] = []

        for resp in responses:
            if not resp.answer or len(resp.answer) < 30:
                continue

            q_id = resp.question_id or "unknown"
            shingles = self._shingle(resp.answer)

            # Compare against stored answers for this question
            stored = self._answer_index.get(q_id, [])
            similarities = []

            for stored_cid, stored_shingles in stored:
                if stored_cid == candidate_id:
                    continue  # Skip self-comparison
                sim = self._jaccard_similarity(shingles, stored_shingles)
                similarities.append(sim)

            # Count how many are above threshold
            cluster_hits = [s for s in similarities if s >= SIMILARITY_THRESHOLD]
            cluster_size = len(cluster_hits)

            if cluster_size >= MIN_CLUSTER_SIZE:
                best_sim = max(cluster_hits) if cluster_hits else 0.0
                max_similarity = max(max_similarity, best_sim)
                max_cluster_size = max(max_cluster_size, cluster_size)
                flagged_questions.append(q_id)
                per_question_scores.append(min(0.50 + cluster_size * 0.10, 0.95))
                flags.append(FlagDetail(
                    type="template_cluster_detected",
                    description=(
                        f"Answer to question '{q_id}' matches {cluster_size} previous "
                        f"candidates at >{SIMILARITY_THRESHOLD:.0%} similarity. "
                        f"Indicates a shared prompt template in circulation."
                    ),
                    severity="high" if cluster_size >= 5 else "medium",
                    evidence={
                        "question_id": q_id,
                        "cluster_size": cluster_size,
                        "max_similarity": round(best_sim, 3),
                    },
                ))
            elif similarities:
                best_sim = max(similarities) if similarities else 0.0
                max_similarity = max(max_similarity, best_sim)
                per_question_scores.append(max(0.05, best_sim * 0.30))
            else:
                per_question_scores.append(0.05)

            # Store this answer in the index
            self._answer_index[q_id].append((candidate_id, shingles))

        if not per_question_scores:
            return CCSResult(score=0.10, confidence=0.10)

        score = float(np.clip(np.mean(per_question_scores), 0.0, 1.0))
        confidence = min(0.30 + len(per_question_scores) * 0.08, 0.80)

        probe = None
        if flagged_questions:
            probe = ProbeSuggestion(
                question=(
                    f"Your answer to question '{flagged_questions[0]}' is very similar "
                    f"to answers we've seen from other candidates. Can you explain your "
                    f"specific approach in your own words?"
                ),
                target_dimension="CCS",
                expected_fraud_response_pattern=(
                    "Unable to elaborate beyond the template. Restates the same "
                    "structure with minor word substitutions."
                ),
            )

        return CCSResult(
            score=round(score, 3),
            confidence=round(confidence, 3),
            max_similarity=round(max_similarity, 3),
            cluster_size=max_cluster_size,
            flagged_questions=flagged_questions,
            flags=flags,
            probe_suggestion=probe,
        )

    def _shingle(self, text: str, n: int = NGRAM_SIZE) -> set[int]:
        """Convert text into a set of hashed character n-grams (shingles)."""
        text = text.lower().strip()
        # Normalize whitespace
        text = " ".join(text.split())
        if len(text) < n:
            return {hash(text)}

        shingles = set()
        for i in range(len(text) - n + 1):
            gram = text[i:i + n]
            shingles.add(hash(gram))
        return shingles

    def _jaccard_similarity(self, a: set[int], b: set[int]) -> float:
        """Jaccard similarity between two shingle sets."""
        if not a or not b:
            return 0.0
        intersection = len(a & b)
        union = len(a | b)
        return intersection / union if union > 0 else 0.0
