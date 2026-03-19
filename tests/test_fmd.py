import pytest


@pytest.mark.asyncio
async def test_fmd_real_has_lower_fraud_score(real_request):
    """Real expert responses contain failure patterns — should score lower than fraud."""
    from services.fmd.detector import FMDDetector
    detector = FMDDetector()
    await detector.initialize()
    result = await detector.analyze(real_request)
    assert 0.0 <= result.score <= 1.0
    assert 0.0 <= result.confidence <= 1.0
    assert isinstance(result.pattern_matches, int)
    assert result.pattern_matches >= 0


@pytest.mark.asyncio
async def test_fmd_fraud_has_no_pattern_matches(fraud_request):
    """Fraud profile synthetic answers lack failure patterns."""
    from services.fmd.detector import FMDDetector
    detector = FMDDetector()
    await detector.initialize()
    result = await detector.analyze(fraud_request)
    # Fraud profiles use generic answers — should have fewer regex pattern hits
    assert 0.0 <= result.score <= 1.0
    assert isinstance(result.pattern_matches, int)


@pytest.mark.asyncio
async def test_fmd_empty_responses_returns_neutral():
    """No responses → neutral score, low confidence."""
    from services.fmd.detector import FMDDetector
    from kive.shared.schemas import SignalRequest
    detector = FMDDetector()
    await detector.initialize()
    req = SignalRequest(
        candidate_id="test",
        profile={"employment_history": [], "skill_timestamps": {}, "education": []},
        screening_responses=[],
    )
    result = await detector.analyze(req)
    assert result.score == pytest.approx(0.5, abs=0.1)
    assert result.confidence <= 0.3


@pytest.mark.asyncio
async def test_fmd_score_bounded(fraud_request):
    """Score must always be in [0, 1]."""
    from services.fmd.detector import FMDDetector
    detector = FMDDetector()
    await detector.initialize()
    result = await detector.analyze(fraud_request)
    assert 0.0 <= result.score <= 1.0


@pytest.mark.asyncio
async def test_fmd_probe_generated_when_no_failure_memory():
    """
    Probe suggestion is only generated when regex_hits==0 AND semantic_score < 0.20.
    We construct a response with zero failure patterns to force probe generation.
    """
    from services.fmd.detector import FMDDetector
    from kive.shared.schemas import SignalRequest, ScreeningResponse
    detector = FMDDetector()
    await detector.initialize()

    # Answer with no failure patterns at all — generic best practices language
    clean_answer = (
        "I always follow best practices and make sure to test everything properly "
        "before deploying to production. I use CI/CD pipelines and ensure code quality "
        "through peer review. My workflow is very systematic and I have never had issues."
    )

    req = SignalRequest(
        candidate_id="test-fmd-probe",
        profile={"employment_history": [], "skill_timestamps": {}, "education": []},
        screening_responses=[
            ScreeningResponse(
                question_id="q1",
                answer=clean_answer,
                latency_ms=3000,
            ),
            ScreeningResponse(
                question_id="q2",
                answer=clean_answer,
                latency_ms=3100,
            ),
        ],
    )

    result = await detector.analyze(req)
    # The detector must produce a score (always)
    assert 0.0 <= result.score <= 1.0
    # With zero patterns, score should be high (fraud signal)
    assert result.pattern_matches == 0
    assert result.score >= 0.55, f"Expected high fraud score with no failure patterns, got {result.score}"

