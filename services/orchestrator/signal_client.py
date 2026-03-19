# services/orchestrator/signal_client.py
# Async HTTP client that calls all 5 signal services.
# MockSignalClient used for local training (no Docker required).

from __future__ import annotations

import asyncio
import os
import random
from typing import Any, Optional

import httpx


SIGNAL_URLS = {
    "tav": os.getenv("TAV_URL", "http://localhost:8001"),
    "svp": os.getenv("SVP_URL", "http://localhost:8002"),
    "fmd": os.getenv("FMD_URL", "http://localhost:8003"),
    "mdc": os.getenv("MDC_URL", "http://localhost:8004"),
    "tsi": os.getenv("TSI_URL", "http://localhost:8005"),
}

SIGNAL_WEIGHTS = {"tav": 0.28, "svp": 0.24, "fmd": 0.20, "mdc": 0.16, "tsi": 0.12}


class SignalClient:
    """
    Live async client — calls real signal microservices.
    Use when all 5 services are running (Docker Compose or local uvicorn).
    """

    def __init__(self, timeout: float = 5.0):
        self.timeout = timeout

    async def extract_all(self, profile) -> dict[str, dict]:
        """Call all 5 services in parallel, return {signal_name: response_dict}."""
        payload = self._build_payload(profile)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            tasks = {
                name: client.post(
                    f"{url}/api/v1/signals/{name}",
                    json=payload,
                )
                for name, url in SIGNAL_URLS.items()
            }
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        output = {}
        for name, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                # Graceful degradation: service unavailable = neutral score
                output[name] = {"score": 0.5, "confidence": 0.0, "weight": SIGNAL_WEIGHTS[name]}
            else:
                try:
                    output[name] = result.json()
                except Exception:
                    output[name] = {"score": 0.5, "confidence": 0.0, "weight": SIGNAL_WEIGHTS[name]}

        return output

    async def probe(self, profile, target: str, evidence_count: int) -> dict:
        """Submit a targeted probe to the relevant signal service."""
        url = SIGNAL_URLS.get(target, SIGNAL_URLS["tav"])
        payload = self._build_payload(profile)
        payload["session_context"] = {"prior_probes": [], "evidence_count": evidence_count}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.post(f"{url}/api/v1/signals/{target}", json=payload)
                data = resp.json()
                return {
                    "score": data.get("score", 0.5),
                    "weight": SIGNAL_WEIGHTS.get(target, 0.20),
                    "probe_question": data.get("probe_suggestion", {}).get("question", ""),
                }
            except Exception:
                return {"score": 0.5, "weight": SIGNAL_WEIGHTS.get(target, 0.20)}

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
    Mock client for local RL training without running services.
    Scores are derived from profile labels — separability is calibrated to
    match expected real-service behavior.
    """

    FRAUD_SIGNAL_MEANS = {"tav": 0.55, "svp": 0.72, "fmd": 0.82, "mdc": 0.45, "tsi": 0.68}
    REAL_SIGNAL_MEANS  = {"tav": 0.07, "svp": 0.22, "fmd": 0.12, "mdc": 0.15, "tsi": 0.15}
    NOISE_STD = 0.08

    def __init__(self, rng=None):
        """
        rng: optional numpy.random.Generator. Pass env.np_random for gym compliance.
        Falls back to Python random if None.
        """
        self._rng = rng

    def _sample(self, mean: float) -> float:
        if self._rng is not None:
            return float(self._rng.normal(mean, self.NOISE_STD))
        return float(random.gauss(mean, self.NOISE_STD))

    async def extract_all(self, profile) -> dict[str, dict]:
        label = getattr(profile, "label", "REAL")
        means = self.FRAUD_SIGNAL_MEANS if label == "FRAUD" else self.REAL_SIGNAL_MEANS

        output = {}
        for name, mean in means.items():
            score = max(0.0, min(1.0, self._sample(mean)))
            output[name] = {
                "score": round(score, 3),
                "confidence": round(0.80, 2),  # fixed for determinism
                "weight": SIGNAL_WEIGHTS[name],
            }
        return output

    async def probe(self, profile, target: str, evidence_count: int) -> dict:
        label = getattr(profile, "label", "REAL")
        mean = (
            self.FRAUD_SIGNAL_MEANS.get(target, 0.5)
            if label == "FRAUD"
            else self.REAL_SIGNAL_MEANS.get(target, 0.2)
        )
        score = max(0.0, min(1.0, self._sample(mean * 0.9)))  # probes slightly more discriminative
        return {
            "score": round(score, 3),
            "weight": SIGNAL_WEIGHTS.get(target, 0.20),
            "probe_question": f"Tell me more about your {target} experience.",
        }
