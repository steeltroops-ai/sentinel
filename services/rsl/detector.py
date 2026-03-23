# services/rsl/detector.py
# Response Latency Slope — weight 0.16 (live mode)
# The single hardest signal to fake without knowing it is being measured.
# Maps response latency against question difficulty tiers.
# Real experts: positive slope within domain (harder = longer think time),
#               steep negative slope outside domain (easy foreign questions = slower).
# AI-assisted: flat slope everywhere (latency = AI query time, not cognitive load).

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kive.shared.schemas import FlagDetail, ProbeSuggestion, SignalRequest

WEIGHT = 0.16

# Difficulty tier ordering for regression
DIFFICULTY_MAP: dict[str, int] = {
    "basic": 1,
    "intermediate": 2,
    "expert": 3,
}

# Topic tier ordering
TOPIC_MAP: dict[str, int] = {
    "core": 3,      # Claimed domain -- should be fastest
    "adjacent": 2,   # Related domain -- moderate speed
    "edge": 1,       # Foreign domain -- slowest for real experts
}

# Minimum data points required for slope computation
MIN_DATA_POINTS = 3

# Slope thresholds
FLAT_SLOPE_THRESHOLD = 50.0          # ms per difficulty tier -- below = flat
EXPECTED_MIN_SLOPE = 200.0           # Human minimum expected slope


@dataclass
class RSLResult:
    score: float           # [0,1] — high = flat or inverted latency pattern
    confidence: float
    difficulty_slope: float = 0.0     # ms per difficulty tier
    topic_slope: float = 0.0          # ms per topic tier
    latency_cv: float = 0.0           # coefficient of variation
    flags: list[FlagDetail] = field(default_factory=list)
    probe_suggestion: Optional[ProbeSuggestion] = None


class RSLDetector:
    """
    Response Latency Slope detector.

    Computes linear regression of response latency vs. question difficulty
    and question topic tier. Real experts show a positive slope (harder = slower),
    while AI-assisted candidates show a flat slope (latency determined by
    AI query time, not cognitive complexity).
    """

    async def initialize(self):
        pass

    async def close(self):
        pass

    async def analyze(self, request: SignalRequest) -> RSLResult:
        responses = request.screening_responses
        if not responses or len(responses) < MIN_DATA_POINTS:
            return RSLResult(score=0.5, confidence=0.10)

        flags: list[FlagDetail] = []
        component_scores: list[float] = []

        # --- Analysis 1: Latency vs Difficulty Slope ---
        difficulty_data = []
        for resp in responses:
            if resp.question_difficulty and resp.latency_ms and resp.latency_ms > 0:
                tier = DIFFICULTY_MAP.get(resp.question_difficulty, 0)
                if tier > 0:
                    difficulty_data.append((tier, resp.latency_ms))

        difficulty_slope = 0.0
        if len(difficulty_data) >= MIN_DATA_POINTS:
            x = np.array([d[0] for d in difficulty_data], dtype=np.float64)
            y = np.array([d[1] for d in difficulty_data], dtype=np.float64)
            difficulty_slope = self._compute_slope(x, y)

            if abs(difficulty_slope) < FLAT_SLOPE_THRESHOLD:
                # Flat slope = latency independent of difficulty = AI pattern
                component_scores.append(0.80)
                flags.append(FlagDetail(
                    type="flat_difficulty_slope",
                    description=(
                        f"Response latency slope vs difficulty: {difficulty_slope:.0f} ms/tier. "
                        f"Expected >{EXPECTED_MIN_SLOPE:.0f} ms/tier for human experts. "
                        f"Flat slope indicates latency is determined by AI query time, "
                        f"not cognitive complexity."
                    ),
                    severity="high",
                    evidence={
                        "slope_ms_per_tier": round(difficulty_slope, 1),
                        "threshold": FLAT_SLOPE_THRESHOLD,
                        "n_data_points": len(difficulty_data),
                    },
                ))
            elif difficulty_slope < 0:
                # Inverted slope = faster on harder questions = suspicious
                component_scores.append(0.70)
                flags.append(FlagDetail(
                    type="inverted_difficulty_slope",
                    description=(
                        f"Negative latency slope: {difficulty_slope:.0f} ms/tier. "
                        f"Expert questions answered faster than basic questions. "
                        f"Cognitive load should increase with difficulty."
                    ),
                    severity="medium",
                    evidence={
                        "slope_ms_per_tier": round(difficulty_slope, 1),
                        "n_data_points": len(difficulty_data),
                    },
                ))
            else:
                # Positive slope = authentic cognitive load pattern
                normalized = min(difficulty_slope / EXPECTED_MIN_SLOPE, 1.0)
                component_scores.append(max(0.05, 0.40 - normalized * 0.35))

        # --- Analysis 2: Latency vs Topic Tier ---
        topic_data = []
        for resp in responses:
            if resp.topic and resp.latency_ms and resp.latency_ms > 0:
                tier = TOPIC_MAP.get(resp.topic, 0)
                if tier > 0:
                    topic_data.append((tier, resp.latency_ms))

        topic_slope = 0.0
        if len(topic_data) >= MIN_DATA_POINTS:
            x = np.array([d[0] for d in topic_data], dtype=np.float64)
            y = np.array([d[1] for d in topic_data], dtype=np.float64)
            topic_slope = self._compute_slope(x, y)

            # Real experts: negative slope (core domain = fast, edge = slow)
            # AI-assisted: flat slope
            if abs(topic_slope) < FLAT_SLOPE_THRESHOLD:
                component_scores.append(0.75)
            elif topic_slope > 0:
                # Slower in core domain than edge — counterintuitive, authentic if expert
                # Actually: topic_tier 3 = core, and real expert should be FAST in core
                # So positive slope (higher tier = slower) is suspicious
                component_scores.append(0.60)
            else:
                # Negative slope = faster in core domain = authentic
                normalized_t = min(abs(topic_slope) / EXPECTED_MIN_SLOPE, 1.0)
                component_scores.append(max(0.05, 0.35 - normalized_t * 0.30))

        # --- Analysis 3: Latency Coefficient of Variation ---
        all_latencies = [r.latency_ms for r in responses if r.latency_ms and r.latency_ms > 0]
        latency_cv = 0.0
        if len(all_latencies) >= MIN_DATA_POINTS:
            mean_lat = float(np.mean(all_latencies))
            std_lat = float(np.std(all_latencies))
            latency_cv = std_lat / max(mean_lat, 1)

            # Low CV = uniform latency = suspicious
            if latency_cv < 0.20:
                component_scores.append(0.70)
                flags.append(FlagDetail(
                    type="uniform_latency",
                    description=(
                        f"Latency coefficient of variation {latency_cv:.2f}. "
                        f"Suspiciously uniform response times across questions of "
                        f"varying difficulty."
                    ),
                    severity="medium",
                    evidence={
                        "cv": round(latency_cv, 3),
                        "mean_ms": round(mean_lat, 0),
                        "std_ms": round(std_lat, 0),
                    },
                ))
            else:
                component_scores.append(max(0.05, 0.30 - latency_cv * 0.40))

        # --- Aggregate ---
        if not component_scores:
            return RSLResult(score=0.5, confidence=0.10)

        score = float(np.clip(np.mean(component_scores), 0.0, 1.0))
        confidence = min(0.30 + len(component_scores) * 0.12, 0.80)

        probe = None
        if score > 0.55:
            probe = ProbeSuggestion(
                question=(
                    "Here's a more advanced question in your claimed area of expertise. "
                    "Take your time."
                ),
                target_dimension="RSL",
                expected_fraud_response_pattern=(
                    "Response time remains flat regardless of difficulty increase, "
                    "or decreases (AI query time is constant regardless of question "
                    "complexity)."
                ),
            )

        return RSLResult(
            score=round(score, 3),
            confidence=round(confidence, 3),
            difficulty_slope=round(difficulty_slope, 1),
            topic_slope=round(topic_slope, 1),
            latency_cv=round(latency_cv, 3),
            flags=flags,
            probe_suggestion=probe,
        )

    def _compute_slope(self, x: np.ndarray, y: np.ndarray) -> float:
        """Simple OLS slope: dy/dx."""
        n = len(x)
        if n < 2:
            return 0.0
        x_mean = float(np.mean(x))
        y_mean = float(np.mean(y))
        numerator = float(np.sum((x - x_mean) * (y - y_mean)))
        denominator = float(np.sum((x - x_mean) ** 2))
        if abs(denominator) < 1e-10:
            return 0.0
        return numerator / denominator
