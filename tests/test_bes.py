# tests/test_bes.py
# Full test coverage for BES (Behavioral Entropy Service)

import pytest
import asyncio
from kive.shared.schemas import SignalRequest, ScreeningResponse


def _make_request(telemetry: dict) -> SignalRequest:
    """Build a SignalRequest with behavioral_telemetry data."""
    return SignalRequest(
        candidate_id="test-bes-candidate",
        profile={"behavioral_telemetry": telemetry},
        screening_responses=[
            ScreeningResponse(
                question_id="q1", answer="Some test answer", latency_ms=5000, topic="core"
            )
        ],
    )


@pytest.mark.asyncio
async def test_bes_no_telemetry():
    """No behavioral data -> neutral score with very low confidence."""
    from services.bes.detector import BESDetector
    det = BESDetector()
    await det.initialize()
    result = await det.analyze(SignalRequest(candidate_id="x", profile={}))
    assert result.score == 0.5
    assert result.confidence == 0.10
    assert len(result.flags) == 0


@pytest.mark.asyncio
async def test_bes_high_paste_ratio():
    """Answer composed mostly via paste -> high fraud score."""
    from services.bes.detector import BESDetector
    det = BESDetector()
    await det.initialize()
    result = await det.analyze(_make_request({
        "keystroke_timings": [],
        "paste_events": [{"byte_length": 900, "timestamp_ms": 1000}],
        "backspace_count": 0,
        "total_chars_typed": 1000,
        "first_char_latency_ms": 2000,
        "blur_events": [],
        "scroll_back_count": 1,
    }))
    assert result.paste_ratio > 0.70
    assert result.score > 0.50
    flag_types = [f.type for f in result.flags]
    assert "high_paste_ratio" in flag_types


@pytest.mark.asyncio
async def test_bes_zero_corrections():
    """Zero backspace on a long answer -> fraud signal."""
    from services.bes.detector import BESDetector
    det = BESDetector()
    await det.initialize()
    result = await det.analyze(_make_request({
        "keystroke_timings": [100.0] * 20,
        "paste_events": [],
        "backspace_count": 0,
        "total_chars_typed": 800,
        "first_char_latency_ms": 1500,
        "blur_events": [],
        "scroll_back_count": 2,
    }))
    flag_types = [f.type for f in result.flags]
    assert "zero_corrections" in flag_types


@pytest.mark.asyncio
async def test_bes_low_keystroke_entropy():
    """Robotic uniform timing -> low entropy -> fraud signal."""
    from services.bes.detector import BESDetector
    det = BESDetector()
    await det.initialize()
    # All timings exactly 100ms = near-zero entropy
    result = await det.analyze(_make_request({
        "keystroke_timings": [100.0] * 50,
        "paste_events": [],
        "backspace_count": 5,
        "total_chars_typed": 200,
        "first_char_latency_ms": 500,
        "blur_events": [],
        "scroll_back_count": 1,
    }))
    assert result.keystroke_entropy < 1.5
    flag_types = [f.type for f in result.flags]
    assert "low_keystroke_entropy" in flag_types


@pytest.mark.asyncio
async def test_bes_suspicious_blur():
    """Window blur in 8-15sec range -> AI query pattern."""
    from services.bes.detector import BESDetector
    det = BESDetector()
    await det.initialize()
    result = await det.analyze(_make_request({
        "keystroke_timings": list(range(50, 300, 5)) * 2,
        "paste_events": [],
        "backspace_count": 10,
        "total_chars_typed": 400,
        "first_char_latency_ms": 1000,
        "blur_events": [
            {"duration_ms": 10000, "timestamp_ms": 5000},
            {"duration_ms": 12000, "timestamp_ms": 20000},
        ],
        "scroll_back_count": 2,
    }))
    assert result.blur_count == 2
    flag_types = [f.type for f in result.flags]
    assert "suspicious_blur_pattern" in flag_types


@pytest.mark.asyncio
async def test_bes_latency_then_paste():
    """Long first-char delay followed by paste -> AI query -> paste pattern."""
    from services.bes.detector import BESDetector
    det = BESDetector()
    await det.initialize()
    result = await det.analyze(_make_request({
        "keystroke_timings": [],
        "paste_events": [{"byte_length": 500, "timestamp_ms": 9000}],
        "backspace_count": 0,
        "total_chars_typed": 600,
        "first_char_latency_ms": 9000,
        "blur_events": [],
        "scroll_back_count": 0,
    }))
    flag_types = [f.type for f in result.flags]
    assert "latency_then_paste" in flag_types


@pytest.mark.asyncio
async def test_bes_authentic_behavior():
    """Human-like behavior: varied timings, corrections, low paste ratio."""
    from services.bes.detector import BESDetector
    import random
    det = BESDetector()
    await det.initialize()
    # Simulate variable human typing
    timings = [random.gauss(150, 60) for _ in range(100)]
    result = await det.analyze(_make_request({
        "keystroke_timings": timings,
        "paste_events": [],
        "backspace_count": 30,
        "total_chars_typed": 400,
        "first_char_latency_ms": 2000,
        "blur_events": [{"duration_ms": 2000, "timestamp_ms": 1000}],
        "scroll_back_count": 3,
    }))
    assert result.score < 0.60
    assert result.keystroke_entropy > 1.5


@pytest.mark.asyncio
async def test_bes_result_schema():
    """BESResult fields must all be within expected ranges."""
    from services.bes.detector import BESDetector
    det = BESDetector()
    await det.initialize()
    result = await det.analyze(_make_request({
        "keystroke_timings": [100.0, 200.0, 150.0] * 10,
        "paste_events": [{"byte_length": 100, "timestamp_ms": 500}],
        "backspace_count": 5,
        "total_chars_typed": 300,
        "first_char_latency_ms": 1500,
        "blur_events": [],
        "scroll_back_count": 1,
    }))
    assert 0.0 <= result.score <= 1.0
    assert 0.0 <= result.confidence <= 1.0
    assert result.keystroke_entropy >= 0.0
    assert 0.0 <= result.paste_ratio <= 1.0
    assert isinstance(result.flags, list)
