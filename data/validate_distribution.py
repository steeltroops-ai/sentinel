# data/validate_distribution.py
# Validates that synthetic profiles produce separable signal distributions.
# Must pass before training — if signals aren't separable, RL agent cannot learn.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.synthetic_generator import ExpertProfile


def validate(profiles_path: str) -> dict:
    print(f"Loading profiles from {profiles_path}...")
    with open(profiles_path) as f:
        raw = json.load(f)

    profiles = [ExpertProfile.from_dict(d) for d in raw]
    real = [p for p in profiles if p.label == "REAL"]
    fraud = [p for p in profiles if p.label == "FRAUD"]

    print(f"Total: {len(profiles)} | Real: {len(real)} | Fraud: {len(fraud)}")

    results = {
        "n_total": len(profiles),
        "n_real": len(real),
        "n_fraud": len(fraud),
        "fraud_ratio": round(len(fraud) / len(profiles), 3),
        "checks": {},
    }

    # ── TAV: fraud profiles should have borderline/inflated timestamps ──
    tav_real_violations  = _tav_violation_rate(real)
    tav_fraud_violations = _tav_violation_rate(fraud)
    results["checks"]["tav_separability"] = {
        "real_violation_rate":  round(tav_real_violations, 3),
        "fraud_violation_rate": round(tav_fraud_violations, 3),
        "gap": round(tav_fraud_violations - tav_real_violations, 3),
        "passed": (tav_fraud_violations - tav_real_violations) >= 0.25,
    }

    # ── FMD: real profiles have failure stories, fraud profiles don't ──
    import re
    failure_pattern = re.compile(
        r"\b(never|stopped|bug|broke|failed|crashed|regret|lesson|outage|issue|misconfigured|"
        r"memory.leak|dropped|wiped|restored|rollback|deadlock|timeout|corrupted|lost|leaked|"
        r"worst|caused|trace|fixed|hit.a|ran.without|silently|evicted|OOM|cost.us|took.\d+)\b",
        re.IGNORECASE,
    )
    fmd_real_presence  = _pattern_presence_rate(real, failure_pattern)
    fmd_fraud_presence = _pattern_presence_rate(fraud, failure_pattern)
    results["checks"]["fmd_separability"] = {
        "real_failure_presence":  round(fmd_real_presence, 3),
        "fraud_failure_presence": round(fmd_fraud_presence, 3),
        "gap": round(fmd_real_presence - fmd_fraud_presence, 3),
        "passed": (fmd_real_presence - fmd_fraud_presence) >= 0.40,
    }

    # ── TSI: fraud profiles should have monotone careers ──
    tsi_real_smooth  = _smooth_career_rate(real)
    tsi_fraud_smooth = _smooth_career_rate(fraud)
    results["checks"]["tsi_separability"] = {
        "real_smooth_rate":  round(tsi_real_smooth, 3),
        "fraud_smooth_rate": round(tsi_fraud_smooth, 3),
        "gap": round(tsi_fraud_smooth - tsi_real_smooth, 3),
        "passed": (tsi_fraud_smooth - tsi_real_smooth) >= 0.20,
    }

    # ── LQA: fraud profiles have AI phrasing artifacts ──
    artifact_pattern = re.compile(r"(It is important to note that|Certainly!|In most cases|depends on the context|As an AI)", re.IGNORECASE)
    lqa_real_art = _pattern_presence_rate(real, artifact_pattern)
    lqa_fraud_art = _pattern_presence_rate(fraud, artifact_pattern)
    results["checks"]["lqa_separability"] = {
        "real_artifact_presence": round(lqa_real_art, 3),
        "fraud_artifact_presence": round(lqa_fraud_art, 3),
        "gap": round(lqa_fraud_art - lqa_real_art, 3),
        "passed": (lqa_fraud_art - lqa_real_art) >= 0.40,
    }

    # ── BES: fraud profiles show pasted telemetry ──
    bes_real_paste = _telemetry_paste_rate(real)
    bes_fraud_paste = _telemetry_paste_rate(fraud)
    results["checks"]["bes_separability"] = {
        "real_paste_rate": round(bes_real_paste, 3),
        "fraud_paste_rate": round(bes_fraud_paste, 3),
        "gap": round(bes_fraud_paste - bes_real_paste, 3),
        "passed": (bes_fraud_paste - bes_real_paste) >= 0.40,
    }

    # ── RSL: latency slope variance (real experts are slower, fraud is uniform) ──
    rsl_real_mean = _mean_latency(real)
    rsl_fraud_mean = _mean_latency(fraud)
    results["checks"]["rsl_separability"] = {
        "real_mean_latency_ms": round(rsl_real_mean, 3),
        "fraud_mean_latency_ms": round(rsl_fraud_mean, 3),
        "passed": rsl_real_mean > rsl_fraud_mean + 2000, 
    }

    # ── Response count sanity ──
    real_with_responses  = sum(1 for p in real  if p.screening_responses)
    fraud_with_responses = sum(1 for p in fraud if p.screening_responses)
    results["checks"]["data_completeness"] = {
        "real_with_responses": real_with_responses,
        "fraud_with_responses": fraud_with_responses,
        "passed": real_with_responses > len(real) * 0.9 and fraud_with_responses > len(fraud) * 0.9,
    }

    # ── Summary ──
    all_passed = all(v["passed"] for v in results["checks"].values())
    results["all_passed"] = all_passed

    _print_report(results)

    if not all_passed:
        print("\nFAIL: Some separability checks did not pass.")
        print("Fix the synthetic generator before training.")
        sys.exit(1)

    print("\nAll separability checks PASSED. Safe to train.")
    return results


def _tav_violation_rate(profiles: list[ExpertProfile]) -> float:
    """Fraction of profiles with at least one timestamp at or before tool release year."""
    from data.synthetic_generator import TECH_TIMELINE
    hits = 0
    for p in profiles:
        for tool, ts in p.skill_timestamps.items():
            tl = TECH_TIMELINE.get(tool.lower())
            if tl is None:
                continue
            try:
                claim_year = int(ts.split("-")[0])
                if claim_year <= tl["release_year"]:
                    hits += 1
                    break
            except Exception:
                pass
    return hits / max(len(profiles), 1)


def _pattern_presence_rate(profiles: list[ExpertProfile], pattern) -> float:
    """Fraction of profiles where any screening answer matches the pattern."""
    hits = 0
    for p in profiles:
        text = " ".join(r.answer for r in p.screening_responses)
        if pattern.search(text):
            hits += 1
    return hits / max(len(profiles), 1)


def _smooth_career_rate(profiles: list[ExpertProfile]) -> float:
    """Fraction of profiles with perfectly monotone upward career trajectories."""
    from data.synthetic_generator import SENIORITY_MAP
    smooth = 0
    for p in profiles:
        history = sorted(p.employment_history, key=lambda r: r.start_year)
        if len(history) < 2:
            continue
        levels = []
        for role in history:
            title_lower = role.title.lower()
            level = 2  # default mid
            # Use max() across ALL matches — longer/more specific titles win
            # e.g.  "senior engineer" → max(engineer=2, senior=3) = 3
            # e.g.  "junior engineer" → max(junior=1, engineer=2) = 2
            matched_level = None
            for kw, lv in SENIORITY_MAP.items():
                if kw in title_lower:
                    matched_level = max(matched_level, lv) if matched_level is not None else lv
            if matched_level is not None:
                level = matched_level
            levels.append(level)
        deltas = [levels[i] - levels[i-1] for i in range(1, len(levels))]
        if all(d >= 0 for d in deltas):
            smooth += 1
    return smooth / max(len(profiles), 1)


def _telemetry_paste_rate(profiles: list[ExpertProfile]) -> float:
    """Fraction of profiles exhibiting heavy paste behaviors."""
    suspicious = 0
    for p in profiles:
        bt = p.behavioral_telemetry
        if not bt:
            continue
        if len(bt.get("paste_events", [])) > 0:
            suspicious += 1
    return suspicious / max(len(profiles), 1)

def _mean_latency(profiles: list[ExpertProfile]) -> float:
    latencies = []
    for p in profiles:
        for r in p.screening_responses:
            latencies.append(r.latency_ms)
    return sum(latencies) / len(latencies) if latencies else 0.0


def _print_report(results: dict):
    print("\n" + "=" * 55)
    print("  KIVE Synthetic Data Distribution Validation")
    print("=" * 55)
    print(f"  Profiles: {results['n_total']} "
          f"(real={results['n_real']}, fraud={results['n_fraud']}, "
          f"ratio={results['fraud_ratio']})")
    print("-" * 55)
    for name, check in results["checks"].items():
        status = "PASS" if check["passed"] else "FAIL"
        print(f"  [{status}] {name}")
        for k, v in check.items():
            if k != "passed":
                print(f"           {k}: {v}")
    print("=" * 55)
    print(f"  Overall: {'PASS' if results['all_passed'] else 'FAIL'}")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/synthetic_profiles.json")
    args = parser.parse_args()
    validate(args.input)
