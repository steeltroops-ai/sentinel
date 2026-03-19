# tests/conftest.py
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def real_profile():
    """A synthetic profile expected to score LOW on all fraud signals."""
    from data.synthetic_generator import RealExpertGenerator
    gen = RealExpertGenerator()
    return gen.generate()


@pytest.fixture
def fraud_profile():
    """A synthetic profile expected to score HIGH on fraud signals."""
    from data.synthetic_generator import FraudExpertGenerator
    gen = FraudExpertGenerator()
    return gen.generate()


@pytest.fixture
def real_request(real_profile):
    from kive.shared.schemas import SignalRequest, ScreeningResponse
    return SignalRequest(
        candidate_id=real_profile.id,
        profile={
            "employment_history": [r.to_dict() for r in real_profile.employment_history],
            "skill_timestamps": real_profile.skill_timestamps,
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
            for r in real_profile.screening_responses
        ],
    )


@pytest.fixture
def fraud_request(fraud_profile):
    from kive.shared.schemas import SignalRequest, ScreeningResponse
    return SignalRequest(
        candidate_id=fraud_profile.id,
        profile={
            "employment_history": [r.to_dict() for r in fraud_profile.employment_history],
            "skill_timestamps": fraud_profile.skill_timestamps,
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
            for r in fraud_profile.screening_responses
        ],
    )
