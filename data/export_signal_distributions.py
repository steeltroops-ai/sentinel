#!/usr/bin/env python3
"""Export signal distributions CSV matching training calibration."""
import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.synthetic_generator import ExpertProfile


class MockSignalClient:
    """Matches services/orchestrator/signal_client.py calibration"""
    
    PASSIVE_FRAUD_MEANS = {"tav": 0.50, "svp": 0.50, "fmd": 0.50, "mdc": 0.50, "tsi": 0.50}
    PASSIVE_REAL_MEANS  = {"tav": 0.50, "svp": 0.50, "fmd": 0.50, "mdc": 0.50, "tsi": 0.50}
    PASSIVE_NOISE_STD   = 0.25
    
    ACTIVE_FRAUD_MEANS = {"bes": 0.90, "lqa": 0.88, "ccs": 0.92, "rsl": 0.85}
    ACTIVE_REAL_MEANS  = {"bes": 0.10, "lqa": 0.12, "ccs": 0.08, "rsl": 0.15}
    ACTIVE_NOISE_STD   = 0.05
    
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
    
    def _sample(self, mean: float, std: float) -> float:
        return float(np.clip(self.rng.normal(mean, std), 0.0, 1.0))
    
    def get_signal_score(self, signal: str, is_fraud: bool) -> dict:
        if signal in self.PASSIVE_FRAUD_MEANS:
            mean = self.PASSIVE_FRAUD_MEANS[signal] if is_fraud else self.PASSIVE_REAL_MEANS[signal]
            score = self._sample(mean, self.PASSIVE_NOISE_STD)
            confidence = 0.5
            n_flags = 0
        else:
            mean = self.ACTIVE_FRAUD_MEANS[signal] if is_fraud else self.ACTIVE_REAL_MEANS[signal]
            score = self._sample(mean, self.ACTIVE_NOISE_STD)
            confidence = 0.85
            n_flags = int(score > 0.7) if is_fraud else 0
        
        return {
            "score": round(score, 4),
            "confidence": round(confidence, 2),
            "n_flags": n_flags
        }


async def export(profiles_path: str, output_path: str, n: int = 200):
    with open(profiles_path) as f:
        raw = json.load(f)
    
    profiles = [ExpertProfile.from_dict(d) for d in raw[:n]]
    print(f"Generating scores for {len(profiles)} profiles...")
    
    client = MockSignalClient(seed=42)
    signals = ["tav", "svp", "fmd", "mdc", "tsi", "bes", "lqa", "ccs", "rsl"]
    
    rows = []
    for i, profile in enumerate(profiles):
        is_fraud = profile.label == "FRAUD"
        row = {"id": profile.id, "label": profile.label}
        
        for sig in signals:
            result = client.get_signal_score(sig, is_fraud)
            row[f"{sig}_score"] = result["score"]
            row[f"{sig}_confidence"] = result["confidence"]
            row[f"{sig}_n_flags"] = result["n_flags"]
        
        rows.append(row)
        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{len(profiles)}")
    
    if rows:
        fieldnames = list(rows[0].keys())
        with open(output_path, "w", newline="", encoding="utf-8") as f:
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
