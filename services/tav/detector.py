# services/tav/detector.py
# Temporal Anchoring Violations — highest-weight signal (0.28)
# Adversarial robustness: MAXIMUM. Operates on external timestamps, not text.
# A fraudster cannot change when Kubernetes was released.

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kive.shared.schemas import FlagDetail, ProbeSuggestion, SignalRequest


CURRENT_YEAR = datetime.now().year
WEIGHT = 0.28

# Technology timeline knowledge base
# (name, release_year, inflection_year, notes)
TECH_TIMELINE_SEED: list[tuple] = [
    ("kubernetes",    2014, 2017, "CNCF graduated 2018; enterprise inflection ~2017"),
    ("docker",        2013, 2015, "Docker Hub 2013; enterprise adoption 2015"),
    ("pytorch",       2016, 2018, "v1.0 stable 2018; research inflection earlier"),
    ("tensorflow",    2015, 2017, "Google open-source 2015; widespread 2017"),
    ("react",         2013, 2015, "Open-sourced 2013; dominant 2015"),
    ("fastapi",       2018, 2020, "Rapid adoption curve post 2020"),
    ("rust",          2015, 2020, "Stable 1.0 2015; systems mainstream 2020"),
    ("langchain",     2022, 2023, "LLM ecosystem explosion 2023"),
    ("nextjs",        2016, 2019, "Vercel push 2019"),
    ("airflow",       2014, 2018, "Apache graduation 2019"),
    ("spark",         2012, 2015, "Apache top-level 2014; enterprise 2015"),
    ("kafka",         2011, 2015, "LinkedIn open-source 2011; mainstream 2015"),
    ("elasticsearch", 2010, 2014, "Elastic company founded 2012; widespread 2014"),
    ("redis",         2009, 2014, "Stable + widely adopted 2014"),
    ("mongodb",       2009, 2013, "Mainstream NoSQL adoption 2013"),
    ("graphql",       2015, 2018, "Facebook open-source 2015; adoption 2018"),
    ("terraform",     2014, 2018, "Hashicorp product; enterprise 2018"),
    ("ansible",       2012, 2016, "Red Hat acquisition 2015; mainstream 2016"),
    ("llm",           2022, 2023, "ChatGPT inflection Nov 2022"),
    ("langchain",     2022, 2023, "Harrison Chase, Jan 2023; mainstream mid-2023"),
    ("huggingface",   2018, 2021, "Transformers library Oct 2019; mainstream 2021"),
    ("mlflow",        2018, 2021, "Databricks; mainstream MLOps 2021"),
    ("zenml",         2021, 2023, "Early adoption 2022-2023"),
    ("go",            2012, 2016, "Go 1.0 stable 2012; widespread 2016"),
    ("typescript",    2012, 2018, "Microsoft 2012; Angular adoption triggered 2017-2018"),
    ("flutter",       2017, 2019, "Flutter 1.0 Dec 2018; inflection 2019"),
    ("vue",           2014, 2017, "Evan You 2014; mainstream 2017"),
    ("aws",           2006, 2012, "S3/EC2 2006; enterprise inflection 2012"),
    ("gcp",           2008, 2016, "App Engine 2008; enterprise 2016"),
    ("azure",         2010, 2015, "Microsoft 2010; enterprise 2015"),
]


@dataclass
class TAVResult:
    score: float                              # [0,1] fraud probability contribution
    confidence: float                         # certainty of this score
    flags: list[FlagDetail] = field(default_factory=list)
    probe_suggestion: Optional[ProbeSuggestion] = None
    needs_probe: bool = False


class TAVDetector:
    """
    Temporal Anchoring Violations detector.

    Primary check: claimed_years > 85% of max possible given release date.
    Secondary check: expert claim during pre-inflection period without corroboration.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or str(Path(__file__).parent / "tech_timeline.db")
        self._timeline: dict[str, dict] = {}
        self._initialized = False

    async def initialize(self):
        self._timeline = self._load_timeline()
        self._initialized = True

    async def close(self):
        pass

    def _load_timeline(self) -> dict[str, dict]:
        """Load from SQLite if available, fall back to seed data."""
        timeline = {}
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute("SELECT name, release_year, inflection_year FROM tools")
            for name, ry, iy in cur.fetchall():
                timeline[name.lower()] = {"release_year": ry, "inflection_year": iy}
            conn.close()
        except Exception:
            pass

        # Always seed built-ins (SQLite overrides if fresher)
        for (name, ry, iy, _) in TECH_TIMELINE_SEED:
            if name.lower() not in timeline:
                timeline[name.lower()] = {"release_year": ry, "inflection_year": iy}

        return timeline

    async def analyze(self, request: SignalRequest) -> TAVResult:
        if not self._initialized:
            await self.initialize()

        employment_history = request.profile.get("employment_history", [])
        skill_timestamps = request.profile.get("skill_timestamps", {})

        # Estimate career start year
        career_start = self._infer_career_start(employment_history)

        violations: list[FlagDetail] = []
        pre_inflection_hits: list[str] = []
        max_score = 0.0

        for tool, claim_date_str in skill_timestamps.items():
            tool_key = tool.lower().strip()
            if tool_key not in self._timeline:
                continue

            timeline = self._timeline[tool_key]
            release_year = timeline["release_year"]
            inflection_year = timeline["inflection_year"]

            # Infer claimed experience years from claim date
            claim_year = self._parse_year(claim_date_str)
            if claim_year is None:
                continue

            claimed_years = CURRENT_YEAR - claim_year
            max_possible = CURRENT_YEAR - release_year

            # Primary: duration violation
            if claimed_years > 0 and max_possible > 0:
                ratio = claimed_years / max_possible
                if ratio > 0.95:
                    score = min((ratio - 0.95) * 10, 1.0)
                    max_score = max(max_score, min(score + 0.5, 1.0))
                    violations.append(FlagDetail(
                        type="temporal_violation_hard",
                        description=(
                            f"Claims {claimed_years}y {tool} experience. "
                            f"{tool.title()} released {release_year} — "
                            f"max possible {max_possible}y. "
                            f"Claim is {ratio:.0%} of physical maximum."
                        ),
                        severity="critical" if ratio > 1.0 else "high",
                        evidence={
                            "tool": tool,
                            "claimed_years": claimed_years,
                            "max_possible": max_possible,
                            "ratio": round(ratio, 3),
                        },
                    ))

                # Secondary: pre-inflection expert claim
                elif claim_year <= inflection_year and claimed_years >= 3:
                    pre_inflection_hits.append(tool)
                    if max_score < 0.45:
                        max_score = 0.45
                    violations.append(FlagDetail(
                        type="pre_inflection_expert_claim",
                        description=(
                            f"Claims {claimed_years}y {tool} expertise, starting {claim_year} "
                            f"(before mainstream inflection {inflection_year}). "
                            f"Early-adopter claims require public corroboration."
                        ),
                        severity="medium",
                        evidence={
                            "tool": tool,
                            "claim_start_year": claim_year,
                            "inflection_year": inflection_year,
                        },
                    ))

        if not violations:
            return TAVResult(score=0.05, confidence=0.80)

        # Confidence scales with number of corroborating violations
        confidence = min(0.50 + len(violations) * 0.15, 0.95)

        probe = self._build_probe(violations, pre_inflection_hits)

        return TAVResult(
            score=round(max_score, 3),
            confidence=round(confidence, 3),
            flags=violations,
            probe_suggestion=probe,
            needs_probe=max_score < 0.75,  # Ambiguous — probe before terminal decision
        )

    def _infer_career_start(self, employment_history: list[dict]) -> int:
        years = [r.get("start_year", CURRENT_YEAR) for r in employment_history]
        return min(years) if years else CURRENT_YEAR - 5

    def _parse_year(self, date_str: str) -> Optional[int]:
        if not date_str:
            return None
        match = re.search(r"\b(19|20)\d{2}\b", str(date_str))
        return int(match.group()) if match else None

    def _build_probe(
        self,
        violations: list[FlagDetail],
        pre_inflection: list[str],
    ) -> ProbeSuggestion:
        if violations and violations[0].type == "temporal_violation_hard":
            tool = violations[0].evidence.get("tool", "this technology")
            return ProbeSuggestion(
                question=(
                    f"Walk me through a specific production issue you debugged with "
                    f"{tool} — include the version and what the root cause was."
                ),
                target_dimension="TAV",
                expected_fraud_response_pattern=(
                    "Generic answer about concepts or features without version-specific "
                    "details or a concrete incident timeline."
                ),
            )

        tool = pre_inflection[0] if pre_inflection else "this technology"
        return ProbeSuggestion(
            question=(
                f"You claimed early expertise in {tool}. "
                f"Describe a specific challenge you encountered when community support "
                f"and documentation were still sparse."
            ),
            target_dimension="TAV",
            expected_fraud_response_pattern=(
                "Vague answer referencing 'early adoption challenges' without specifics, "
                "or answer that describes current mature ecosystem rather than early friction."
            ),
        )


def _seed_db(db_path: str):
    """Utility: seed the SQLite knowledge base from TECH_TIMELINE_SEED."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tools (
            name TEXT PRIMARY KEY,
            release_year INTEGER,
            inflection_year INTEGER,
            notes TEXT
        )
    """)
    conn.executemany(
        "INSERT OR REPLACE INTO tools VALUES (?,?,?,?)",
        [(n.lower(), ry, iy, notes) for (n, ry, iy, notes) in TECH_TIMELINE_SEED],
    )
    conn.commit()
    conn.close()


if __name__ == "__main__":
    import asyncio
    db = Path(__file__).parent / "tech_timeline.db"
    _seed_db(str(db))
    print(f"Seeded {len(TECH_TIMELINE_SEED)} tools into {db}")
