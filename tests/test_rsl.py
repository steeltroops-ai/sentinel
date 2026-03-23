# tests/test_rsl.py
# Full test coverage for RSL (Response Latency Slope)

import pytest
from kive.shared.schemas import SignalRequest, ScreeningResponse


def _make_latency_request(responses: list[dict]) -> SignalRequest:
    """responses = [{answer, latency_ms, question_difficulty, topic}]."""
    return SignalRequest(
        candidate_id="test-rsl",
        profile={},
        screening_responses=[
            ScreeningResponse(
                question_id=f"q{i}",
                answer=r.get("answer", "test answer"),
                latency_ms=r["latency_ms"],
                question_difficulty=r.get("question_difficulty"),
                topic=r.get("topic"),
            )
            for i, r in enumerate(responses)
        ],
    )


@pytest.mark.asyncio
async def test_rsl_no_data():
    """No responses -> neutral."""
    from services.rsl.detector import RSLDetector
    det = RSLDetector()
    await det.initialize()
    result = await det.analyze(SignalRequest(candidate_id="x", profile={}))
    assert result.score == 0.5
    assert result.confidence == 0.10


@pytest.mark.asyncio
async def test_rsl_flat_difficulty_slope():
    """Same response time for all difficulties -> flat slope -> fraud signal."""
    from services.rsl.detector import RSLDetector
    det = RSLDetector()
    await det.initialize()
    result = await det.analyze(_make_latency_request([
        {"latency_ms": 5000, "question_difficulty": "basic", "topic": "core"},
        {"latency_ms": 5000, "question_difficulty": "intermediate", "topic": "core"},
        {"latency_ms": 5000, "question_difficulty": "expert", "topic": "core"},
        {"latency_ms": 5000, "question_difficulty": "basic", "topic": "core"},
    ]))
    assert abs(result.difficulty_slope) < 100
    flag_types = [f.type for f in result.flags]
    assert "flat_difficulty_slope" in flag_types
    assert result.score > 0.50


@pytest.mark.asyncio
async def test_rsl_positive_slope_authentic():
    """Harder questions take longer -> positive slope -> authentic."""
    from services.rsl.detector import RSLDetector
    det = RSLDetector()
    await det.initialize()
    result = await det.analyze(_make_latency_request([
        {"latency_ms": 3000, "question_difficulty": "basic", "topic": "core"},
        {"latency_ms": 8000, "question_difficulty": "intermediate", "topic": "core"},
        {"latency_ms": 15000, "question_difficulty": "expert", "topic": "core"},
        {"latency_ms": 4000, "question_difficulty": "basic", "topic": "core"},
    ]))
    assert result.difficulty_slope > 200
    assert result.score < 0.50


@pytest.mark.asyncio
async def test_rsl_inverted_slope():
    """Expert questions faster than basic -> inverted -> suspicious."""
    from services.rsl.detector import RSLDetector
    det = RSLDetector()
    await det.initialize()
    result = await det.analyze(_make_latency_request([
        {"latency_ms": 10000, "question_difficulty": "basic", "topic": "core"},
        {"latency_ms": 7000, "question_difficulty": "intermediate", "topic": "core"},
        {"latency_ms": 3000, "question_difficulty": "expert", "topic": "core"},
    ]))
    assert result.difficulty_slope < 0
    flag_types = [f.type for f in result.flags]
    assert "inverted_difficulty_slope" in flag_types


@pytest.mark.asyncio
async def test_rsl_uniform_latency_cv():
    """Very uniform latency across questions -> low CV -> suspicious."""
    from services.rsl.detector import RSLDetector
    det = RSLDetector()
    await det.initialize()
    result = await det.analyze(_make_latency_request([
        {"latency_ms": 5000, "question_difficulty": "basic", "topic": "core"},
        {"latency_ms": 5050, "question_difficulty": "intermediate", "topic": "adjacent"},
        {"latency_ms": 4980, "question_difficulty": "expert", "topic": "edge"},
        {"latency_ms": 5020, "question_difficulty": "basic", "topic": "core"},
    ]))
    assert result.latency_cv < 0.20
    flag_types = [f.type for f in result.flags]
    assert "uniform_latency" in flag_types


@pytest.mark.asyncio
async def test_rsl_variable_latency_authentic():
    """High latency variation = human-like = authentic."""
    from services.rsl.detector import RSLDetector
    det = RSLDetector()
    await det.initialize()
    result = await det.analyze(_make_latency_request([
        {"latency_ms": 2000, "question_difficulty": "basic", "topic": "core"},
        {"latency_ms": 12000, "question_difficulty": "expert", "topic": "core"},
        {"latency_ms": 25000, "question_difficulty": "expert", "topic": "edge"},
        {"latency_ms": 3500, "question_difficulty": "basic", "topic": "adjacent"},
    ]))
    assert result.latency_cv > 0.40
    assert result.score < 0.50


@pytest.mark.asyncio
async def test_rsl_compute_slope():
    """OLS slope computation should be correct."""
    import numpy as np
    from services.rsl.detector import RSLDetector
    det = RSLDetector()
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([100.0, 200.0, 300.0])
    slope = det._compute_slope(x, y)
    assert abs(slope - 100.0) < 0.01


@pytest.mark.asyncio
async def test_rsl_compute_slope_flat():
    """Flat data -> zero slope."""
    import numpy as np
    from services.rsl.detector import RSLDetector
    det = RSLDetector()
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([100.0, 100.0, 100.0])
    slope = det._compute_slope(x, y)
    assert abs(slope) < 0.01


@pytest.mark.asyncio
async def test_rsl_result_schema():
    """RSLResult must have valid fields."""
    from services.rsl.detector import RSLDetector
    det = RSLDetector()
    await det.initialize()
    result = await det.analyze(_make_latency_request([
        {"latency_ms": 5000, "question_difficulty": "basic", "topic": "core"},
        {"latency_ms": 7000, "question_difficulty": "intermediate", "topic": "core"},
        {"latency_ms": 10000, "question_difficulty": "expert", "topic": "core"},
    ]))
    assert 0.0 <= result.score <= 1.0
    assert 0.0 <= result.confidence <= 1.0
    assert isinstance(result.difficulty_slope, float)
    assert isinstance(result.topic_slope, float)
    assert isinstance(result.latency_cv, float)
    assert isinstance(result.flags, list)
