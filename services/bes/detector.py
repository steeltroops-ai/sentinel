# services/bes/detector.py
# Behavioral Entropy Service — weight 0.18 (live mode: primary behavioral signal)
# Detects AI-assisted responses via interaction telemetry patterns:
# keystroke timing entropy, paste events, window focus loss, correction patterns.
# These signals are invisible to the candidate and cannot be fabricated
# without knowing they are being measured.

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kive.shared.schemas import FlagDetail, ProbeSuggestion, SignalRequest

WEIGHT = 0.18

# Thresholds derived from empirical observation of human vs AI-assisted typing
PASTE_RATIO_THRESHOLD = 0.70        # >70% of answer arrived via paste
ZERO_BACKSPACE_WORD_THRESHOLD = 80  # 0 corrections on 80+ word answer = suspicious
FIRST_CHAR_LATENCY_THRESHOLD = 8000 # ms before first character
BLUR_DURATION_QUERY_PATTERN = (8000, 15000)  # 8-15 second blur = AI query window
MIN_KEYSTROKE_ENTROPY = 1.5         # bits; below = artificial/robotic typing


@dataclass
class BESResult:
    score: float           # [0,1] — high = robotic/AI-assisted behavior
    confidence: float
    keystroke_entropy: float = 0.0
    paste_ratio: float = 0.0
    correction_rate: float = 0.0
    blur_count: int = 0
    flags: list[FlagDetail] = field(default_factory=list)
    probe_suggestion: Optional[ProbeSuggestion] = None


class BESDetector:
    """
    Behavioral Entropy Service detector.

    Analyzes interaction telemetry to detect AI-assisted response patterns.
    Operates on behavioral signals that are orthogonal to text content.

    Expected telemetry fields in profile["behavioral_telemetry"]:
        keystroke_timings: list[float]  — inter-key intervals in ms
        paste_events: list[dict]        — {byte_length: int, timestamp_ms: int}
        backspace_count: int
        total_chars_typed: int
        first_char_latency_ms: float
        time_to_submit_ms: float
        blur_events: list[dict]         — {duration_ms: float, timestamp_ms: int}
        scroll_back_count: int
        mouse_movement_points: int
    """

    async def initialize(self):
        pass

    async def close(self):
        pass

    async def analyze(self, request: SignalRequest) -> BESResult:
        telemetry = request.profile.get("behavioral_telemetry", {})

        if not telemetry:
            # No behavioral data available — return neutral with low confidence
            return BESResult(score=0.5, confidence=0.10)

        flags: list[FlagDetail] = []
        fraud_signals: list[float] = []

        # --- Signal 1: Keystroke Entropy ---
        keystroke_timings = telemetry.get("keystroke_timings", [])
        keystroke_entropy = self._compute_keystroke_entropy(keystroke_timings)
        if len(keystroke_timings) >= 10:
            if keystroke_entropy < MIN_KEYSTROKE_ENTROPY:
                fraud_signals.append(0.85)
                flags.append(FlagDetail(
                    type="low_keystroke_entropy",
                    description=(
                        f"Keystroke timing entropy {keystroke_entropy:.2f} bits. "
                        f"Human typing typically exceeds {MIN_KEYSTROKE_ENTROPY} bits. "
                        f"Low entropy indicates robotic or artificially uniform key timing."
                    ),
                    severity="high",
                    evidence={
                        "entropy_bits": round(keystroke_entropy, 3),
                        "threshold": MIN_KEYSTROKE_ENTROPY,
                    },
                ))
            else:
                fraud_signals.append(max(0.10, 0.50 - keystroke_entropy * 0.15))

        # --- Signal 2: Paste Ratio ---
        paste_ratio = self._compute_paste_ratio(telemetry)
        if paste_ratio > PASTE_RATIO_THRESHOLD:
            fraud_signals.append(0.90)
            flags.append(FlagDetail(
                type="high_paste_ratio",
                description=(
                    f"Paste ratio {paste_ratio:.0%} — most of the answer arrived "
                    f"via clipboard paste rather than direct typing."
                ),
                severity="critical" if paste_ratio > 0.90 else "high",
                evidence={"paste_ratio": round(paste_ratio, 3)},
            ))
        elif paste_ratio >= 0:
            fraud_signals.append(paste_ratio * 0.60)

        # --- Signal 3: Zero Corrections ---
        backspace_count = telemetry.get("backspace_count", -1)
        total_chars = telemetry.get("total_chars_typed", 0)
        if backspace_count == 0 and total_chars > ZERO_BACKSPACE_WORD_THRESHOLD * 5:
            fraud_signals.append(0.80)
            flags.append(FlagDetail(
                type="zero_corrections",
                description=(
                    f"Zero backspace events on {total_chars}-character answer. "
                    f"No human types a long response without any corrections."
                ),
                severity="high",
                evidence={
                    "backspace_count": backspace_count,
                    "total_chars": total_chars,
                },
            ))
        elif backspace_count > 0 and total_chars > 0:
            correction_rate = backspace_count / total_chars
            # Normal correction rate: 5-15%
            if correction_rate < 0.02:
                fraud_signals.append(0.55)
            else:
                fraud_signals.append(0.10)

        # --- Signal 4: First Character Latency ---
        first_char_latency = telemetry.get("first_char_latency_ms", 0)
        if first_char_latency > FIRST_CHAR_LATENCY_THRESHOLD:
            # Long delay before first character + check if followed by paste
            paste_events = telemetry.get("paste_events", [])
            if paste_events:
                fraud_signals.append(0.85)
                flags.append(FlagDetail(
                    type="latency_then_paste",
                    description=(
                        f"First character latency {first_char_latency}ms followed by paste event. "
                        f"Pattern consistent with AI query -> copy -> paste workflow."
                    ),
                    severity="high",
                    evidence={
                        "first_char_latency_ms": first_char_latency,
                        "paste_count": len(paste_events),
                    },
                ))
            else:
                fraud_signals.append(0.35)  # Long pause but typed directly = thinking

        # --- Signal 5: Window Blur Events ---
        blur_events = telemetry.get("blur_events", [])
        suspicious_blurs = [
            b for b in blur_events
            if BLUR_DURATION_QUERY_PATTERN[0] <= b.get("duration_ms", 0) <= BLUR_DURATION_QUERY_PATTERN[1]
        ]
        if suspicious_blurs:
            fraud_signals.append(min(0.50 + len(suspicious_blurs) * 0.15, 0.90))
            flags.append(FlagDetail(
                type="suspicious_blur_pattern",
                description=(
                    f"{len(suspicious_blurs)} window blur events in the 8-15 second range. "
                    f"Duration consistent with switching to an AI tool, querying, and returning."
                ),
                severity="high" if len(suspicious_blurs) >= 2 else "medium",
                evidence={
                    "suspicious_blur_count": len(suspicious_blurs),
                    "total_blur_count": len(blur_events),
                    "blur_durations_ms": [b.get("duration_ms", 0) for b in suspicious_blurs],
                },
            ))

        # --- Signal 6: Scroll-Back Absence ---
        scroll_back_count = telemetry.get("scroll_back_count", -1)
        if scroll_back_count == 0 and total_chars > 300:
            fraud_signals.append(0.40)
            flags.append(FlagDetail(
                type="no_scroll_back",
                description=(
                    "Zero scroll-back events on a multi-part answer. "
                    "Real humans re-read their work and scroll back to earlier sections."
                ),
                severity="medium",
                evidence={"scroll_back_count": 0, "total_chars": total_chars},
            ))

        # --- Aggregate ---
        if not fraud_signals:
            return BESResult(score=0.5, confidence=0.10)

        score = float(np.clip(np.mean(fraud_signals), 0.0, 1.0))
        confidence = min(0.40 + len(fraud_signals) * 0.10, 0.90)

        # Build probe suggestion
        probe = None
        if score > 0.60:
            probe = ProbeSuggestion(
                question=(
                    "For this next question, please think through your answer "
                    "before typing and try to answer in one continuous pass."
                ),
                target_dimension="BES",
                expected_fraud_response_pattern=(
                    "Same paste-heavy pattern persists. Or latency profile changes "
                    "dramatically, suggesting they cannot maintain the facade under "
                    "behavioral monitoring."
                ),
            )

        return BESResult(
            score=round(score, 3),
            confidence=round(confidence, 3),
            keystroke_entropy=round(keystroke_entropy, 3),
            paste_ratio=round(paste_ratio, 3),
            correction_rate=round(backspace_count / max(total_chars, 1), 4) if backspace_count >= 0 else 0.0,
            blur_count=len(blur_events),
            flags=flags,
            probe_suggestion=probe,
        )

    def _compute_keystroke_entropy(self, timings: list[float]) -> float:
        """Shannon entropy of inter-key interval distribution (in bits)."""
        if len(timings) < 10:
            return 0.0

        # Quantize timings into 20ms buckets
        bucket_size = 20
        buckets: dict[int, int] = {}
        for t in timings:
            bucket = int(t // bucket_size)
            buckets[bucket] = buckets.get(bucket, 0) + 1

        total = sum(buckets.values())
        entropy = 0.0
        for count in buckets.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        return entropy

    def _compute_paste_ratio(self, telemetry: dict) -> float:
        """Fraction of total content that arrived via paste events."""
        paste_events = telemetry.get("paste_events", [])
        total_chars = telemetry.get("total_chars_typed", 0)

        if total_chars <= 0:
            return 0.0

        paste_bytes = sum(p.get("byte_length", 0) for p in paste_events)
        return min(paste_bytes / total_chars, 1.0)
