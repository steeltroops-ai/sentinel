# services/tsi/detector.py
# Trajectory Smoothness Index — weight 0.12 (ESTABLISHED signal)
# Real careers have lateral moves, gaps, pivots, and down-steps.
# Fabricated CVs trend linearly upward with progressive role labels.
# Penalize perfect monotone upward trajectory. Reward authentic variance.

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kive.shared.schemas import FlagDetail, ProbeSuggestion, SignalRequest

WEIGHT = 0.12

SENIORITY_MAP: dict[str, int] = {
    "intern": 0, "trainee": 0, "apprentice": 0,
    "junior": 1, "associate": 1, "entry": 1,
    "engineer": 2, "developer": 2, "analyst": 2, "specialist": 2,
    "senior": 3,
    "staff": 4, "lead": 4, "tech lead": 4,
    "principal": 5, "architect": 5, "expert": 5,
    "manager": 5, "engineering manager": 5,
    "director": 6, "head": 6, "group": 6,
    "vp": 7, "vice president": 7,
    "cto": 8, "ceo": 8, "founder": 7, "co-founder": 7, "president": 8,
}


@dataclass
class TSIResult:
    score: float
    confidence: float
    seniority_deltas: list[int] = field(default_factory=list)
    flags: list[FlagDetail] = field(default_factory=list)
    probe_suggestion: Optional[ProbeSuggestion] = None


class TSIDetector:

    async def initialize(self):
        pass

    async def close(self):
        pass

    async def analyze(self, request: SignalRequest) -> TSIResult:
        history = request.profile.get("employment_history", [])

        if len(history) < 2:
            return TSIResult(score=0.10, confidence=0.15)

        # Sort by start_year ascending
        sorted_history = sorted(history, key=lambda r: r.get("start_year", 0))

        seniority_levels = [self._classify_seniority(r.get("title", "")) for r in sorted_history]
        deltas = [
            seniority_levels[i] - seniority_levels[i - 1]
            for i in range(1, len(seniority_levels))
        ]

        # Check for gaps between roles
        gaps = self._detect_gaps(sorted_history)

        # Fraud signals
        all_positive = all(d >= 0 for d in deltas)
        zero_variance = float(np.var(deltas)) < 0.10
        no_gaps = len(gaps) == 0
        n_roles = len(sorted_history)
        span_years = self._career_span(sorted_history)

        fraud_score = 0.0
        reasons = []

        if all_positive and n_roles >= 3:
            fraud_score += 0.35
            reasons.append("all transitions are upward (no lateral moves or down-steps)")

        if zero_variance and n_roles >= 3:
            fraud_score += 0.25
            reasons.append(f"seniority delta variance {np.var(deltas):.3f} — perfectly uniform progression")

        if no_gaps and span_years > 7:
            fraud_score += 0.15
            reasons.append(f"no employment gaps over {span_years:.0f}-year career")

        fraud_score = float(np.clip(fraud_score, 0.0, 1.0))
        confidence = 0.40 + min(n_roles * 0.05, 0.30)

        flags: list[FlagDetail] = []
        probe = None

        if fraud_score >= 0.50:
            flags.append(FlagDetail(
                type="tsi_smooth_trajectory",
                description=(
                    f"Career trajectory is suspiciously smooth: {', '.join(reasons)}. "
                    f"Real careers have at least one lateral move, gap, or pivot over {span_years:.0f} years."
                ),
                severity="medium",
                evidence={
                    "n_roles": n_roles,
                    "span_years": round(span_years, 1),
                    "seniority_sequence": seniority_levels,
                    "deltas": deltas,
                    "gaps_found": gaps,
                },
            ))
            probe = ProbeSuggestion(
                question=(
                    "Your career shows a very linear progression. Has there been a role "
                    "you took for reasons other than seniority — a lateral move, a startup "
                    "bet, or a step back for learning? Walk me through the decision."
                ),
                target_dimension="TSI",
                expected_fraud_response_pattern=(
                    "Generic answer about 'always challenging myself' or 'seeking growth' "
                    "without a specific non-linear career decision with a concrete reason."
                ),
            )

        return TSIResult(
            score=round(fraud_score, 3),
            confidence=round(confidence, 3),
            seniority_deltas=deltas,
            flags=flags,
            probe_suggestion=probe,
        )

    def _classify_seniority(self, title: str) -> int:
        title_lower = title.lower()
        best_match = 2  # Default to mid-level if unclassifiable
        best_len = 0
        for keyword, level in SENIORITY_MAP.items():
            if keyword in title_lower and len(keyword) > best_len:
                best_match = level
                best_len = len(keyword)
        return best_match

    def _detect_gaps(self, history: list[dict]) -> list[dict]:
        gaps = []
        for i in range(1, len(history)):
            prev_end = history[i - 1].get("end_year")
            curr_start = history[i].get("start_year", 0)
            if prev_end and curr_start and curr_start - prev_end > 1:
                gaps.append({"gap_years": curr_start - prev_end, "after_company": history[i-1].get("company", "?")})
        return gaps

    def _career_span(self, history: list[dict]) -> float:
        from datetime import datetime
        start = min(r.get("start_year", datetime.now().year) for r in history)
        end_years = [r.get("end_year") or datetime.now().year for r in history]
        return max(end_years) - start
