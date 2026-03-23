# services/orchestrator/signal_client.py
# Async HTTP client for KIVE signal services.
# MockSignalClient: local training without Docker.
#
# Signal taxonomy:
#   PASSIVE (free at reset): TAV, SVP, FMD, MDC, TSI
#   ACTIVE  (probe required): BES, LQA, CCS, RSL
#
# Calibration rationale:
#   Passive signals are WEAK -- means compressed (fraud ~0.56, real ~0.42).
#   Active signals are STRONG -- means spread (fraud ~0.85, real ~0.12).
#   This forces the agent to probe when passive evidence is ambiguous.
#   Reference: Chernoff (1959) sequential hypothesis testing --
#   the agent must decide when accumulated evidence crosses the decision boundary.

from __future__ import annotations

import asyncio
import os
import random
from typing import Any, Optional

import numpy as np
import httpx


SIGNAL_URLS = {
    "tav": os.getenv("TAV_URL", "http://localhost:8001"),
    "svp": os.getenv("SVP_URL", "http://localhost:8002"),
    "fmd": os.getenv("FMD_URL", "http://localhost:8003"),
    "mdc": os.getenv("MDC_URL", "http://localhost:8004"),
    "tsi": os.getenv("TSI_URL", "http://localhost:8005"),
    "bes": os.getenv("BES_URL", "http://localhost:8006"),
    "lqa": os.getenv("LQA_URL", "http://localhost:8007"),
    "ccs": os.getenv("CCS_URL", "http://localhost:8008"),
    "rsl": os.getenv("RSL_URL", "http://localhost:8009"),
}

SIGNAL_WEIGHTS = {
    "tav": 0.14, "svp": 0.11, "fmd": 0.11, "mdc": 0.09, "tsi": 0.07,
    "bes": 0.18, "lqa": 0.12, "ccs": 0.10, "rsl": 0.08,
}


class SignalClient:
    """Live async client -- calls real signal microservices."""

    def __init__(self, timeout: float = 5.0):
        self.timeout = timeout

    async def extract_all(self, profile) -> dict[str, dict]:
        payload = self._build_payload(profile)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            tasks = {
                name: client.post(f"{url}/api/v1/signals/{name}", json=payload)
                for name, url in SIGNAL_URLS.items()
            }
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        output = {}
        for name, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                output[name] = {"score": 0.5, "confidence": 0.0, "weight": SIGNAL_WEIGHTS[name]}
            else:
                try:
                    output[name] = result.json()
                except Exception:
                    output[name] = {"score": 0.5, "confidence": 0.0, "weight": SIGNAL_WEIGHTS[name]}
        return output

    async def probe(self, profile, target: str, evidence_count: int) -> dict:
        url = SIGNAL_URLS.get(target, SIGNAL_URLS["tav"])
        payload = self._build_payload(profile)
        payload["session_context"] = {"prior_probes": [], "evidence_count": evidence_count}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.post(f"{url}/api/v1/signals/{target}", json=payload)
                data = resp.json()
                return {"score": data.get("score", 0.5), "weight": SIGNAL_WEIGHTS.get(target, 0.10)}
            except Exception:
                return {"score": 0.5, "weight": SIGNAL_WEIGHTS.get(target, 0.10)}

    def _build_payload(self, profile) -> dict:
        return {
            "candidate_id": getattr(profile, "id", "unknown"),
            "profile": {
                "employment_history": [
                    r.to_dict() if hasattr(r, "to_dict") else r
                    for r in getattr(profile, "employment_history", [])
                ],
                "skill_timestamps": getattr(profile, "skill_timestamps", {}),
                "education": [],
            },
            "screening_responses": [
                r.to_dict() if hasattr(r, "to_dict") else r
                for r in getattr(profile, "screening_responses", [])
            ],
            "web_signals": {"github_repos": [], "linkedin_delta": []},
            "session_context": {"prior_probes": [], "evidence_count": 0},
        }


class MockSignalClient:
    """
    Mock client for local RL training.

    KEY CALIBRATION:
      Passive signals (TAV-TSI) are intentionally WEAK:
        FRAUD means: 0.52-0.60  |  REAL means: 0.38-0.48
        Noise std: 0.12 (high)  |  Visibility: 35%

      Active signals (BES-RSL) are intentionally STRONG:
        FRAUD means: 0.82-0.92  |  REAL means: 0.08-0.18
        Noise std: 0.024 (very low on probe)

      This calibration ensures:
        - Passive-only belief: ~0.45-0.55 for BOTH classes (ambiguous)
        - After 1-2 probes: belief diverges to ~0.15 or ~0.85 (decisive)
        - Agent MUST learn to probe before deciding

      Without this, the agent converges to trivial 1-step threshold policy.
    """

    # Passive: compressed means -- weak signal, high ambiguity
    PASSIVE_FRAUD_MEANS = {"tav": 0.58, "svp": 0.56, "fmd": 0.60, "mdc": 0.55, "tsi": 0.52}
    PASSIVE_REAL_MEANS  = {"tav": 0.42, "svp": 0.44, "fmd": 0.38, "mdc": 0.45, "tsi": 0.48}
    PASSIVE_NOISE_STD   = 0.12

    # Active: spread means -- strong signal, low ambiguity
    ACTIVE_FRAUD_MEANS = {"bes": 0.88, "lqa": 0.85, "ccs": 0.92, "rsl": 0.82}
    ACTIVE_REAL_MEANS  = {"bes": 0.10, "lqa": 0.12, "ccs": 0.06, "rsl": 0.15}
    ACTIVE_NOISE_STD   = 0.08

    # Probe noise reduction factor: probes return cleaner readings
    PROBE_NOISE_FACTOR = 0.3

    def __init__(self, rng=None):
        self._rng = rng

    def _sample(self, mean: float, std: float) -> float:
        if self._rng is not None:
            return float(np.clip(self._rng.normal(mean, std), 0.0, 1.0))
        return float(max(0.0, min(1.0, random.gauss(mean, std))))

    async def extract_all(self, profile) -> dict[str, dict]:
        """
        Returns signal scores for all 9 services.
        Passive: 35% visible with high noise. 65% masked to 0.5.
        Active: always 0.5 (require probing).
        """
        label = getattr(profile, "label", "REAL")
        is_fraud = (label == "FRAUD")
        output = {}

        # Passive signals
        for name in ("tav", "svp", "fmd", "mdc", "tsi"):
            fraud_mean = self.PASSIVE_FRAUD_MEANS[name]
            real_mean = self.PASSIVE_REAL_MEANS[name]
            mean = fraud_mean if is_fraud else real_mean

            is_visible = (self._rng.random() < 0.35) if self._rng else (random.random() < 0.35)
            if is_visible:
                score = self._sample(mean, self.PASSIVE_NOISE_STD)
                confidence = 0.3
            else:
                score = 0.5
                confidence = 0.0

            output[name] = {
                "score": round(score, 4),
                "confidence": round(confidence, 2),
                "weight": SIGNAL_WEIGHTS[name],
            }

        # Active signals: always masked
        for name in ("bes", "lqa", "ccs", "rsl"):
            output[name] = {
                "score": 0.5,
                "confidence": 0.0,
                "weight": SIGNAL_WEIGHTS[name],
            }

        return output

    async def probe(self, profile, target: str, evidence_count: int) -> dict:
        """
        Probe returns clean, high-confidence signal.
        Active probes are strongly discriminative.
        If probing a passive signal, returns a cleaner version.
        """
        label = getattr(profile, "label", "REAL")
        is_fraud = (label == "FRAUD")

        if target in self.ACTIVE_FRAUD_MEANS:
            mean = self.ACTIVE_FRAUD_MEANS[target] if is_fraud else self.ACTIVE_REAL_MEANS[target]
            std = self.ACTIVE_NOISE_STD * self.PROBE_NOISE_FACTOR
        else:
            mean = self.PASSIVE_FRAUD_MEANS.get(target, 0.55) if is_fraud else self.PASSIVE_REAL_MEANS.get(target, 0.45)
            std = self.PASSIVE_NOISE_STD * self.PROBE_NOISE_FACTOR

        score = self._sample(mean, std)
        return {
            "score": round(score, 4),
            "weight": SIGNAL_WEIGHTS.get(target, 0.10),
        }
