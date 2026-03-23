# data/export_signal_distributions.py
# Loads synthetic profiles, runs all 5 detectors, exports scores as CSV
# for use in notebook signal analysis plots.

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.synthetic_generator import ExpertProfile


async def _score_profile(profile: ExpertProfile, detectors: dict) -> dict:
    row = {
        "id": profile.id,
        "label": profile.label,
    }
    for name, detector in detectors.items():
        try:
            from kive.shared.schemas import SignalRequest, ScreeningResponse
            req = SignalRequest(
                candidate_id=profile.id,
                profile={
                    "employment_history": [r.to_dict() for r in profile.employment_history],
                    "skill_timestamps": profile.skill_timestamps,
                    "education": [],
                },
                screening_responses=[
                    ScreeningResponse(
                        question_id=r.question_id,
                        answer=r.answer,
                        latency_ms=r.latency_ms,
                        topic=r.topic,
                        question_difficulty=r.question_difficulty,
                    )
                    for r in profile.screening_responses
                ],
            )
            result = await detector.analyze(req)
            row[f"{name}_score"] = round(result.score, 4)
            row[f"{name}_confidence"] = round(result.confidence, 4)
            row[f"{name}_n_flags"] = len(result.flags)
        except Exception as e:
            row[f"{name}_score"] = -1.0
            row[f"{name}_confidence"] = -1.0
            row[f"{name}_n_flags"] = 0
    return row


async def export(profiles_path: str, output_path: str, n: int = 200):
    with open(profiles_path) as f:
        raw = json.load(f)

    profiles = [ExpertProfile.from_dict(d) for d in raw[:n]]
    print(f"Scoring {len(profiles)} profiles...")

    from services.tav.detector import TAVDetector
    from services.svp.detector import SVPDetector
    from services.fmd.detector import FMDDetector
    from services.mdc.detector import MDCDetector
    from services.tsi.detector import TSIDetector
    from services.bes.detector import BESDetector
    from services.lqa.detector import LQADetector
    from services.ccs.detector import CCSDetector
    from services.rsl.detector import RSLDetector

    detectors = {
        "tav": TAVDetector(),
        "svp": SVPDetector(),
        "fmd": FMDDetector(),
        "mdc": MDCDetector(),
        "tsi": TSIDetector(),
        "bes": BESDetector(),
        "lqa": LQADetector(),
        "ccs": CCSDetector(),
        "rsl": RSLDetector(),
    }
    for d in detectors.values():
        await d.initialize()

    rows = []
    for i, profile in enumerate(profiles):
        row = await _score_profile(profile, detectors)
        rows.append(row)
        if (i + 1) % 50 == 0:
            print(f"  Scored {i + 1}/{len(profiles)}")

    # Write CSV
    import csv
    if rows:
        fieldnames = list(rows[0].keys())
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Exported {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/synthetic_profiles.json")
    parser.add_argument("--output", default="data/signal_distributions.csv")
    parser.add_argument("--n", type=int, default=200)
    args = parser.parse_args()
    asyncio.run(export(args.input, args.output, args.n))
