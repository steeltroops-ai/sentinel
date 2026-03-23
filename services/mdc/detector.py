# services/mdc/detector.py
# Market Demand Correlation — weight 0.16 (ESTABLISHED signal)
# Detects retroactive skill inflation: bulk skill additions 2-4 weeks after demand spikes.
# Most effective as a corroborating signal. Requires skill_timestamps + market data.

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kive.shared.schemas import FlagDetail, ProbeSuggestion, SignalRequest

WEIGHT = 0.16

# Approximate demand inflection dates per technology (YYYY-MM format)
# When Google Trends / job posting volume peaked for that skill
DEMAND_SPIKES: dict[str, list[str]] = {
    "langchain":     ["2023-03", "2023-06"],
    "llm":           ["2023-01", "2023-11"],
    "chatgpt":       ["2023-01"],
    "kubernetes":    ["2018-06", "2019-09"],
    "docker":        ["2015-06"],
    "pytorch":       ["2019-01", "2020-06"],
    "rust":          ["2021-01", "2022-06"],
    "fastapi":       ["2021-01"],
    "react":         ["2016-01"],
    "typescript":    ["2018-06"],
    "terraform":     ["2019-06"],
    "airflow":       ["2019-01"],
    "spark":         ["2017-06"],
}

# Max lag window: if skills were added within this many months of a demand spike, flag it
RETROACTIVE_LAG_MONTHS = 3


@dataclass
class MDCResult:
    score: float
    confidence: float
    retroactive_hits: list[dict] = field(default_factory=list)
    flags: list[FlagDetail] = field(default_factory=list)
    probe_suggestion: Optional[ProbeSuggestion] = None


class MDCDetector:

    async def initialize(self):
        pass

    async def close(self):
        pass

    async def analyze(self, request: SignalRequest) -> MDCResult:
        skill_timestamps: dict[str, str] = request.profile.get("skill_timestamps", {})

        if not skill_timestamps:
            # No timestamp data — can't compute MDC, return low-confidence neutral
            return MDCResult(score=0.2, confidence=0.15)

        retroactive_hits = []
        for skill, ts_str in skill_timestamps.items():
            skill_lower = skill.lower()
            if skill_lower not in DEMAND_SPIKES:
                continue

            claim_date = self._parse_ym(ts_str)
            if claim_date is None:
                continue

            for spike_str in DEMAND_SPIKES[skill_lower]:
                spike_date = self._parse_ym(spike_str)
                if spike_date is None:
                    continue

                lag_months = self._months_diff(claim_date, spike_date)
                # Claim appears 0-3 months AFTER demand spike = retroactive inflation
                if 0 <= lag_months <= RETROACTIVE_LAG_MONTHS:
                    retroactive_hits.append({
                        "skill": skill,
                        "claim_date": ts_str,
                        "demand_spike": spike_str,
                        "lag_months": lag_months,
                    })

        score = 0.10
        confidence = 0.50
        flags = []
        probe = None

        if retroactive_hits:
            n = len(retroactive_hits)
            score = min(0.40 + n * 0.15, 0.85)
            confidence = min(0.45 + n * 0.10, 0.75)

            flags.append(FlagDetail(
                type="retroactive_skill_inflation",
                description=(
                    f"{n} skill(s) added within {RETROACTIVE_LAG_MONTHS} months of demand spike. "
                    f"Pattern suggests resume padding to match job market trends."
                ),
                severity="medium" if n == 1 else "high",
                evidence={"hits": retroactive_hits},
            ))

            probe = ProbeSuggestion(
                question=(
                    f"You listed {retroactive_hits[0]['skill']} as a skill around "
                    f"{retroactive_hits[0]['claim_date']}. What were you specifically "
                    f"building with it at that time?"
                ),
                target_dimension="MDC",
                expected_fraud_response_pattern=(
                    "Vague answer about 'exploring it' or 'learning it for a project' "
                    "without a concrete deliverable or codebase."
                ),
            )



        # --- Skill burst detection (NEW) ---
        # Multiple skills added on the exact same date = fabrication burst
        date_counts: dict[str, list[str]] = {}
        for skill, ts_str in skill_timestamps.items():
            ts_norm = str(ts_str).strip()
            if ts_norm not in date_counts:
                date_counts[ts_norm] = []
            date_counts[ts_norm].append(skill)

        burst_skills: list[str] = []
        for date_str, skills_on_date in date_counts.items():
            if len(skills_on_date) >= 3:  # 3+ skills on same date = suspicious
                burst_skills.extend(skills_on_date)

        if burst_skills:
            burst_penalty = min(len(burst_skills) * 0.08, 0.25)
            score = min(score + burst_penalty, 0.90)
            confidence = min(confidence + 0.10, 0.80)
            flags.append(FlagDetail(
                type="skill_addition_burst",
                description=(
                    f"{len(burst_skills)} skills added on the same date. "
                    f"Real skill acquisition is gradual. Bulk additions suggest "
                    f"profile padding in a single session."
                ),
                severity="medium",
                evidence={"burst_skills": burst_skills},
            ))

        return MDCResult(
            score=round(score, 3),
            confidence=round(confidence, 3),
            retroactive_hits=retroactive_hits,
            flags=flags,
            probe_suggestion=probe,
        )

    def _parse_ym(self, s: str) -> Optional[tuple[int, int]]:
        """Parse YYYY-MM into (year, month). Returns None on failure."""
        try:
            parts = str(s).strip().split("-")
            return int(parts[0]), int(parts[1])
        except Exception:
            return None

    def _months_diff(self, d1: tuple[int, int], d2: tuple[int, int]) -> int:
        """Return d1 - d2 in months. Positive = d1 is after d2."""
        return (d1[0] - d2[0]) * 12 + (d1[1] - d2[1])
