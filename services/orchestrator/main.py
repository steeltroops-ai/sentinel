# services/orchestrator/main.py
# FastAPI session API for the KIVE RL Orchestrator.
# Manages POMDP vetting sessions: start -> probe loop -> terminal decision -> ground_truth.

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kive.shared.schemas import (
    DecisionResponse,
    GroundTruthRequest,
    HealthResponse,
    ProbeResponseRequest,
    SessionStartRequest,
    SessionStartResponse,
    SignalResponse,
)
from services.orchestrator.signal_client import SignalClient, MockSignalClient
from services.orchestrator.env import ExpertFraudEnv

# ── In-memory session store (replace with Redis for production) ──────────────

_sessions: dict[str, dict[str, Any]] = {}
_signal_client: SignalClient | MockSignalClient | None = None
_start_time = time.time()

SERVICE_VERSION = "1.0.0"
SIGNAL_WEIGHTS = {
    "tav": 0.14, "svp": 0.11, "fmd": 0.11, "mdc": 0.09, "tsi": 0.07,
    "bes": 0.18, "lqa": 0.12, "ccs": 0.10, "rsl": 0.08,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _signal_client
    import os
    # Use mock client if TAV_URL not set (local dev / testing)
    if os.getenv("TAV_URL"):
        _signal_client = SignalClient()
    else:
        _signal_client = MockSignalClient()
    yield
    _sessions.clear()


app = FastAPI(
    title="KIVE Orchestrator",
    description="RL orchestrator for expert fraud vetting — POMDP session management",
    version=SERVICE_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="orchestrator",
        version=SERVICE_VERSION,
        uptime_seconds=round(time.time() - _start_time, 1),
    )


# ── Session Start ─────────────────────────────────────────────────────────────

@app.post(
    "/api/v1/orchestrator/session/start",
    response_model=SessionStartResponse,
    tags=["session"],
)
async def start_session(request: SessionStartRequest) -> SessionStartResponse:
    """
    Initialize a vetting session. Calls all 5 signal services in parallel.
    Returns initial fraud_belief, confidence, and all signal scores.
    """
    # Build a lightweight profile object for the signal client
    profile = _ProfileProxy(request)

    signals_raw = await _signal_client.extract_all(profile)

    # Compute initial belief from weighted signals
    belief = sum(
        float(signals_raw.get(name, {}).get("score", 0.5)) * weight
        for name, weight in SIGNAL_WEIGHTS.items()
    )
    belief = max(0.0, min(1.0, belief))

    import numpy as np
    scores = [float(signals_raw.get(n, {}).get("score", 0.5)) for n in SIGNAL_WEIGHTS]
    confidence = max(0.0, min(1.0, 1.0 - float(np.std(scores)) * 2.0))

    # Build SignalResponse objects for the response payload
    initial_signals: dict[str, SignalResponse] = {}
    for name, data in signals_raw.items():
        initial_signals[name] = SignalResponse(
            signal=name.upper(),
            score=float(data.get("score", 0.5)),
            confidence=float(data.get("confidence", 0.5)),
            weight=SIGNAL_WEIGHTS.get(name, 0.2),
            flags=[],
            probe_suggestion=None,
            latency_ms=int(data.get("latency_ms", 0)),
            model_version=data.get("model_version", f"{name}-v1.0.0"),
        )

    # Determine initial recommendation
    if belief > 0.75 and confidence > 0.6:
        recommendation = "REJECT"
    elif belief < 0.25 and confidence > 0.6:
        recommendation = "PASS"
    else:
        recommendation = "PROBE"

    session_id = str(uuid.uuid4())
    _sessions[session_id] = {
        "candidate_id": request.candidate_id,
        "profile": request,
        "fraud_belief": belief,
        "confidence": confidence,
        "evidence_count": 0,
        "signal_scores": {k: float(v.get("score", 0.5)) for k, v in signals_raw.items()},
        "terminated": False,
        "action": recommendation,
        "probe_question": None,
        "probe_target": None,
        "created_at": time.time(),
    }

    return SessionStartResponse(
        session_id=session_id,
        initial_belief=round(belief, 3),
        initial_confidence=round(confidence, 3),
        initial_signals=initial_signals,
        next_action_recommendation=recommendation,
    )


# ── Probe Response ────────────────────────────────────────────────────────────

@app.post(
    "/api/v1/orchestrator/session/{session_id}/probe_response",
    tags=["session"],
)
async def submit_probe_response(
    session_id: str,
    request: ProbeResponseRequest,
) -> dict[str, Any]:
    """
    Submit the candidate's answer to the probe question.
    Updates fraud_belief via Bayesian update, increments evidence_count.
    """
    session = _get_session(session_id)
    if session["terminated"]:
        raise HTTPException(400, detail="Session already terminated")
    if session["evidence_count"] >= 5:
        raise HTTPException(400, detail="Max probes reached — call /decision to terminate")

    # Identify weakest active signal dimension to probe
    scores = session["signal_scores"]
    ambiguity = {k: 1.0 - abs(v - 0.5) * 2 for k, v in scores.items() if k in ("bes", "lqa", "ccs", "rsl")}
    target = max(ambiguity, key=ambiguity.get) if ambiguity else "bes"

    # Build probe profile with the answer injected
    probe_profile = _ProbeProxy(session["profile"], request.probe_answer, request.latency_ms)
    probe_result = await _signal_client.probe(probe_profile, target, session["evidence_count"])

    # Bayesian update
    prior = session["fraud_belief"]
    score = float(probe_result.get("score", 0.5))
    weight = float(probe_result.get("weight", 0.20))
    import numpy as np
    numerator = score * prior
    denominator = score * prior + (1 - score) * (1 - prior) + 1e-8
    posterior = numerator / denominator
    new_belief = float(np.clip(prior + weight * (posterior - prior), 0.0, 1.0))

    session["fraud_belief"] = new_belief
    session["evidence_count"] += 1
    session["confidence"] = min(session["confidence"] + 0.15, 1.0)
    session["signal_scores"][target] = score

    return {
        "session_id": session_id,
        "updated_belief": round(new_belief, 3),
        "confidence": round(session["confidence"], 3),
        "evidence_count": session["evidence_count"],
        "probed_dimension": target,
    }


# ── Decision ──────────────────────────────────────────────────────────────────

@app.get(
    "/api/v1/orchestrator/session/{session_id}/decision",
    response_model=DecisionResponse,
    tags=["session"],
)
async def get_decision(session_id: str) -> DecisionResponse:
    """
    Get current agent decision. May return PROBE (with question) or terminal action.
    """
    session = _get_session(session_id)
    belief = session["fraud_belief"]
    confidence = session["confidence"]
    evidence_count = session["evidence_count"]

    # Determine action using POMDP decision logic
    if evidence_count >= 5:
        action = "FLAG"
        terminated = True
    elif belief > 0.75 and confidence > 0.6:
        action = "REJECT"
        terminated = True
    elif belief < 0.25 and confidence > 0.6:
        action = "PASS"
        terminated = True
    elif 0.4 <= belief <= 0.6 or confidence < 0.5:
        action = "PROBE"
        terminated = False
    elif belief > 0.6:
        action = "FLAG" if confidence < 0.5 else "REJECT"
        terminated = True
    else:
        action = "PASS"
        terminated = True

    probe_question = None
    probe_target = None

    if action == "PROBE":
        scores = session["signal_scores"]
        ambiguity = {k: 1.0 - abs(v - 0.5) * 2 for k, v in scores.items() if k in ("bes", "lqa", "ccs", "rsl")}
        probe_target = max(ambiguity, key=ambiguity.get) if ambiguity else "bes"
        probe_question = _generate_probe_question(probe_target, scores)

    session["action"] = action
    session["terminated"] = terminated
    session["probe_question"] = probe_question
    session["probe_target"] = probe_target

    return DecisionResponse(
        session_id=session_id,
        action=action,
        fraud_belief=round(belief, 3),
        confidence=round(confidence, 3),
        probe_question=probe_question,
        probe_target_dimension=probe_target,
        evidence_count=evidence_count,
        terminated=terminated,
    )


# ── Ground Truth ──────────────────────────────────────────────────────────────

@app.post("/api/v1/orchestrator/ground_truth", tags=["training"])
async def submit_ground_truth(request: GroundTruthRequest) -> dict[str, str]:
    """
    Submit post-call ground truth for retrospective model training.
    Logged to MLflow. Used in next retraining cycle.
    """
    try:
        import mlflow
        with mlflow.start_run(run_name="ground_truth_log", nested=True):
            mlflow.log_params({
                "candidate_id": request.candidate_id,
                "actual_label": request.actual_label,
            })
            if request.client_rating is not None:
                mlflow.log_metric("client_rating", request.client_rating)
    except Exception:
        pass  # MLflow optional in live serving context

    return {
        "status": "logged",
        "candidate_id": request.candidate_id,
        "actual_label": request.actual_label,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_session(session_id: str) -> dict[str, Any]:
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(404, detail=f"Session {session_id} not found")
    return session


def _generate_probe_question(target: str, scores: dict[str, float]) -> str:
    PROBE_QUESTIONS = {
        "tav": (
            "Walk me through a specific production issue you debugged with one of your "
            "claimed technologies — include the version and what the root cause was."
        ),
        "svp": (
            "Shift topics for a moment: tell me something you genuinely don't know well "
            "in your field, and why you haven't prioritized learning it."
        ),
        "fmd": (
            "What's the worst production failure you personally caused? Be specific — "
            "what was the tool or version, what broke, and how did you recover?"
        ),
        "mdc": (
            "Pick a skill you listed and tell me specifically what you were building with "
            "it when you first added it to your profile."
        ),
        "tsi": (
            "Has there been a role you took for reasons other than seniority — a lateral "
            "move, startup bet, or deliberate step back? Walk me through the decision."
        ),
    }
    return PROBE_QUESTIONS.get(target, PROBE_QUESTIONS["fmd"])


class _ProfileProxy:
    """Lightweight wrapper to make SessionStartRequest look like a profile for signal_client."""
    def __init__(self, req: SessionStartRequest):
        self.id = req.candidate_id
        self.employment_history = req.profile.get("employment_history", [])
        self.skill_timestamps = req.profile.get("skill_timestamps", {})
        self.screening_responses = req.screening_responses

    def to_dict(self):
        return {
            "id": self.id,
            "employment_history": self.employment_history,
            "skill_timestamps": self.skill_timestamps,
        }


class _ProbeProxy:
    """Wraps probe answer into the profile interface for signal_client.probe()."""
    def __init__(self, base_req: SessionStartRequest, answer: str, latency_ms: int):
        self.id = base_req.candidate_id
        self.employment_history = base_req.profile.get("employment_history", [])
        self.skill_timestamps = base_req.profile.get("skill_timestamps", {})
        from kive.shared.schemas import ScreeningResponse
        self.screening_responses = [
            ScreeningResponse(
                question_id="probe",
                answer=answer,
                latency_ms=latency_ms,
            )
        ]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
