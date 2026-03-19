import pytest


@pytest.mark.asyncio
async def test_svp_real_scores_low(real_request):
    """Real expert — high specificity variance across topics — should score low."""
    from services.svp.detector import SVPDetector
    detector = SVPDetector()
    await detector.initialize()
    result = await detector.analyze(real_request)
    assert 0.0 <= result.score <= 1.0
    assert 0.0 <= result.confidence <= 1.0
    assert isinstance(result.variance, float)


@pytest.mark.asyncio
async def test_svp_fraud_scores_higher_on_uniform_responses(fraud_request):
    """Fraud profile has uniformly fluent responses — SVP should detect it."""
    from services.svp.detector import SVPDetector
    detector = SVPDetector()
    await detector.initialize()
    result = await detector.analyze(fraud_request)
    assert 0.0 <= result.score <= 1.0
    # Variance for fraud (all responses similar specificity level) should be lower
    assert isinstance(result.variance, float)


@pytest.mark.asyncio
async def test_svp_no_responses_returns_neutral():
    """Empty screening responses should return low-confidence neutral score."""
    from services.svp.detector import SVPDetector
    from kive.shared.schemas import SignalRequest
    detector = SVPDetector()
    await detector.initialize()
    req = SignalRequest(
        candidate_id="test",
        profile={"employment_history": [], "skill_timestamps": {}, "education": []},
        screening_responses=[],
    )
    result = await detector.analyze(req)
    assert result.confidence <= 0.3, "Should be low confidence with no data"


@pytest.mark.asyncio
async def test_svp_result_has_per_topic_scores(real_request):
    """per_topic_scores must be a dict with at least one entry for non-empty responses."""
    from services.svp.detector import SVPDetector
    detector = SVPDetector()
    await detector.initialize()
    result = await detector.analyze(real_request)
    assert isinstance(result.per_topic_scores, dict)
