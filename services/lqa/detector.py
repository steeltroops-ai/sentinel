# services/lqa/detector.py
# Linguistic Quality Assurance — weight 0.14
# Detects GPT-generated text via statistical fingerprints:
# artifact phrase density, syntactic complexity variance, hedging asymmetry,
# sentence length distribution, and transition phrase overuse.
# Key insight: GPT has a characteristic "voice" that is measurable even when
# the content is factually correct.

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kive.shared.schemas import FlagDetail, ProbeSuggestion, SignalRequest

WEIGHT = 0.14

# GPT artifact phrases — high-probability output tokens and constructions
_ARTIFACT_PHRASES = [
    "it's worth noting", "it is worth noting",
    "certainly", "absolutely",
    "as an ai", "as a language model",
    "in most cases", "it depends on the context",
    "when it comes to", "one of the key",
    "it is important to note", "it should be noted",
    "let me explain", "great question",
    "this is a great", "that's a great",
    "here's the thing", "at the end of the day",
    "there are several", "there are various",
    "plays a crucial role", "a key aspect",
    "in today's", "in the realm of",
    "leverage", "utilize",
    "delve into", "explore",
    "robust", "comprehensive",
    "cutting-edge", "state-of-the-art",
    "seamless", "streamline",
]

# Transition phrases that GPT overuses at paragraph boundaries
_TRANSITION_PHRASES = re.compile(
    r"^(Furthermore|Additionally|Moreover|In conclusion|To summarize|"
    r"That being said|Having said that|On the other hand|In addition|"
    r"It's also worth|Another important|First and foremost|Last but not least|"
    r"With that in mind|Building on this|To elaborate)",
    re.MULTILINE | re.IGNORECASE,
)

# Hedging phrases
_HEDGING_PATTERN = re.compile(
    r"\b(it depends|generally speaking|in most cases|typically|arguably|"
    r"broadly speaking|to some extent|it varies|in general|one could say|"
    r"it's subjective|there's no one-size|it really depends)\b",
    re.IGNORECASE,
)

# Sentence splitting regex
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


@dataclass
class LQAResult:
    score: float           # [0,1] — high = GPT fingerprint detected
    confidence: float
    artifact_density: float = 0.0
    syntactic_variance: float = 0.0
    transition_ratio: float = 0.0
    hedging_density: float = 0.0
    sentence_length_cv: float = 0.0   # coefficient of variation
    flags: list[FlagDetail] = field(default_factory=list)
    probe_suggestion: Optional[ProbeSuggestion] = None


class LQADetector:
    """
    Linguistic Quality Assurance detector.

    Identifies GPT-generated responses through 5 statistical text fingerprints.
    Does NOT evaluate correctness — a perfectly correct answer can still be
    flagged as AI-generated based on stylistic markers.
    """

    async def initialize(self):
        pass

    async def close(self):
        pass

    async def analyze(self, request: SignalRequest) -> LQAResult:
        responses = request.screening_responses
        if not responses:
            return LQAResult(score=0.5, confidence=0.15)

        all_text = " ".join(r.answer for r in responses if r.answer)
        if not all_text.strip() or len(all_text) < 50:
            return LQAResult(score=0.5, confidence=0.15)

        words = all_text.split()
        n_words = max(len(words), 1)
        flags: list[FlagDetail] = []
        component_scores: list[float] = []

        # --- Signal 1: Artifact Phrase Density ---
        text_lower = all_text.lower()
        artifact_count = sum(1 for phrase in _ARTIFACT_PHRASES if phrase in text_lower)
        artifact_density = artifact_count / (n_words / 100)  # per 100 words
        artifact_score = float(np.clip(artifact_density * 0.30, 0.0, 1.0))
        component_scores.append(artifact_score)

        if artifact_count >= 3:
            flags.append(FlagDetail(
                type="gpt_artifact_phrases",
                description=(
                    f"{artifact_count} GPT-characteristic phrases detected "
                    f"({artifact_density:.1f} per 100 words). "
                    f"Human technical writing rarely uses these constructions."
                ),
                severity="high" if artifact_count >= 5 else "medium",
                evidence={
                    "artifact_count": artifact_count,
                    "density_per_100w": round(artifact_density, 2),
                },
            ))

        # --- Signal 2: Syntactic Complexity Variance ---
        per_response_complexity = []
        for resp in responses:
            if resp.answer and len(resp.answer) > 20:
                complexity = self._flesch_kincaid_proxy(resp.answer)
                per_response_complexity.append(complexity)

        syntactic_variance = 0.0
        syntactic_score = 0.0
        if len(per_response_complexity) >= 2:
            syntactic_variance = float(np.var(per_response_complexity))
            # GPT has low syntactic variance (uniform complexity)
            # Human has high variance (different effort per question)
            if syntactic_variance < 2.0:
                syntactic_score = min(0.80, 0.40 + (2.0 - syntactic_variance) * 0.20)
            else:
                syntactic_score = max(0.05, 0.30 - syntactic_variance * 0.03)
            component_scores.append(syntactic_score)

        # --- Signal 3: Transition Phrase Overuse ---
        transition_matches = _TRANSITION_PHRASES.findall(all_text)
        sentences = _SENTENCE_SPLIT.split(all_text)
        n_sentences = max(len(sentences), 1)
        transition_ratio = len(transition_matches) / n_sentences
        transition_score = float(np.clip(transition_ratio * 2.0, 0.0, 1.0))
        component_scores.append(transition_score)

        if len(transition_matches) >= 3:
            flags.append(FlagDetail(
                type="transition_overuse",
                description=(
                    f"{len(transition_matches)} formal transition phrases across "
                    f"{n_sentences} sentences ({transition_ratio:.0%}). "
                    f"GPT overuses 'Furthermore,' 'Additionally,' 'Moreover' at "
                    f"paragraph boundaries."
                ),
                severity="medium",
                evidence={
                    "transition_count": len(transition_matches),
                    "ratio": round(transition_ratio, 3),
                    "examples": transition_matches[:5],
                },
            ))

        # --- Signal 4: Hedging Density Asymmetry ---
        hedging_matches = _HEDGING_PATTERN.findall(all_text)
        hedging_density = len(hedging_matches) / (n_words / 100)
        # GPT hedges on factual questions where experts would be direct
        hedging_score = float(np.clip(hedging_density * 0.25, 0.0, 1.0))
        component_scores.append(hedging_score)

        # --- Signal 5: Sentence Length Distribution ---
        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
        sentence_length_cv = 0.0
        sentence_score = 0.0
        if len(sentence_lengths) >= 3:
            mean_len = float(np.mean(sentence_lengths))
            std_len = float(np.std(sentence_lengths))
            sentence_length_cv = std_len / max(mean_len, 1) if mean_len > 0 else 0
            # GPT has lower CV (more uniform sentence lengths)
            # Human has higher CV (variable sentence lengths)
            if sentence_length_cv < 0.40:
                sentence_score = min(0.70, 0.35 + (0.40 - sentence_length_cv) * 1.0)
            else:
                sentence_score = max(0.05, 0.25 - sentence_length_cv * 0.20)
            component_scores.append(sentence_score)

        # --- Aggregate ---
        if not component_scores:
            return LQAResult(score=0.5, confidence=0.15)

        score = float(np.clip(np.mean(component_scores), 0.0, 1.0))
        confidence = min(0.35 + len(component_scores) * 0.10, 0.85)

        probe = None
        if score > 0.55:
            probe = ProbeSuggestion(
                question=(
                    "Can you restate your answer to the previous question using "
                    "only your own words, without any notes or references?"
                ),
                target_dimension="LQA",
                expected_fraud_response_pattern=(
                    "The restated answer has identical artifact phrases and "
                    "structure, or the candidate cannot produce an equivalent "
                    "explanation in a different style."
                ),
            )

        return LQAResult(
            score=round(score, 3),
            confidence=round(confidence, 3),
            artifact_density=round(artifact_density, 3),
            syntactic_variance=round(syntactic_variance, 3),
            transition_ratio=round(transition_ratio, 3),
            hedging_density=round(hedging_density, 3),
            sentence_length_cv=round(sentence_length_cv, 3),
            flags=flags,
            probe_suggestion=probe,
        )

    def _flesch_kincaid_proxy(self, text: str) -> float:
        """
        Approximate Flesch-Kincaid grade level without an NLP library.
        Uses average sentence length and average syllable count per word.
        """
        sentences = _SENTENCE_SPLIT.split(text)
        n_sentences = max(len(sentences), 1)
        words = text.split()
        n_words = max(len(words), 1)
        syllables = sum(self._count_syllables(w) for w in words)

        asl = n_words / n_sentences
        asw = syllables / n_words
        # FK grade level formula
        return 0.39 * asl + 11.8 * asw - 15.59

    def _count_syllables(self, word: str) -> int:
        """Approximate syllable count for English words."""
        word = word.lower().strip(".,!?;:'\"")
        if not word:
            return 1
        count = 0
        vowels = "aeiouy"
        prev_vowel = False
        for char in word:
            if char in vowels:
                if not prev_vowel:
                    count += 1
                prev_vowel = True
            else:
                prev_vowel = False
        if word.endswith("e") and count > 1:
            count -= 1
        return max(count, 1)
