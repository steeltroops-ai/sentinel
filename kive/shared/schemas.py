# kive/shared/schemas.py
# Unified Pydantic v2 schemas shared across ALL signal services and orchestrator.
# Import from here. Never define request/response models per-service.

from __future__ import annotations

import uuid
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ─── Request Models ──────────────────────────────────────────────────────────

class Role(BaseModel):
    title: str
    company: str
    start_year: int
    end_year: Optional[int] = None
    skills: list[str] = Field(default_factory=list)
    claimed_skills_added_date: dict[str, str] = Field(default_factory=dict)
    # YYYY-MM format. Mapped to skill name for MDC cross-reference.


class ScreeningResponse(BaseModel):
    question_id: str
    answer: str
    latency_ms: int
    topic: Optional[str] = None          # "core" | "adjacent" | "edge"
    question_difficulty: Optional[str] = None  # "basic" | "intermediate" | "expert"


class WebSignals(BaseModel):
    github_repos: list[dict[str, Any]] = Field(default_factory=list)
    linkedin_delta: list[dict[str, Any]] = Field(default_factory=list)


class SessionContext(BaseModel):
    prior_probes: list[dict[str, Any]] = Field(default_factory=list)
    evidence_count: int = 0


class SignalRequest(BaseModel):
    candidate_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    profile: dict[str, Any]
    # Expected keys: employment_history (list[dict]), skill_timestamps (dict), education (list)
    screening_responses: list[ScreeningResponse] = Field(default_factory=list)
    web_signals: WebSignals = Field(default_factory=WebSignals)
    session_context: SessionContext = Field(default_factory=SessionContext)


# ─── Response Models ─────────────────────────────────────────────────────────

class FlagDetail(BaseModel):
    type: str
    description: str
    severity: Literal["low", "medium", "high", "critical"]
    evidence: dict[str, Any] = Field(default_factory=dict)


class ProbeSuggestion(BaseModel):
    question: str
    target_dimension: str   # "TAV" | "SVP" | "FMD" | "MDC" | "TSI"
    expected_fraud_response_pattern: str


class SignalResponse(BaseModel):
    signal: str                                   # "TAV" | "SVP" | "FMD" | "MDC" | "TSI"
    score: float = Field(ge=0.0, le=1.0)          # Fraud probability contribution
    confidence: float = Field(ge=0.0, le=1.0)    # Certainty of this score
    weight: float                                  # Fixed signal weight in belief computation
    flags: list[FlagDetail] = Field(default_factory=list)
    probe_suggestion: Optional[ProbeSuggestion] = None
    latency_ms: int
    model_version: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "error"]
    service: str
    version: str
    uptime_seconds: Optional[float] = None


# ─── Orchestrator Session Models ─────────────────────────────────────────────

class SessionStartRequest(BaseModel):
    candidate_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    profile: dict[str, Any]
    screening_responses: list[ScreeningResponse] = Field(default_factory=list)
    web_signals: WebSignals = Field(default_factory=WebSignals)


class SessionStartResponse(BaseModel):
    session_id: str
    initial_belief: float
    initial_confidence: float
    initial_signals: dict[str, SignalResponse]
    next_action_recommendation: Literal["PASS", "REJECT", "FLAG", "PROBE"]


class ProbeResponseRequest(BaseModel):
    session_id: str
    probe_answer: str
    latency_ms: int


class DecisionResponse(BaseModel):
    session_id: str
    action: Literal["PASS", "REJECT", "FLAG", "PROBE"]
    fraud_belief: float
    confidence: float
    probe_question: Optional[str] = None
    probe_target_dimension: Optional[str] = None
    evidence_count: int
    terminated: bool


class GroundTruthRequest(BaseModel):
    candidate_id: str
    actual_label: Literal["REAL", "FRAUD"]
    client_rating: Optional[float] = Field(default=None, ge=0.0, le=5.0)
