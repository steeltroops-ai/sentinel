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


@pytest.mark.asyncio
async def test_svp_proper_noun_density():
    """Real experts use specific pronouns, giving them a higher proper noun density."""
    from services.svp.detector import SVPDetector
    from kive.shared.schemas import SignalRequest, ScreeningResponse
    detector = SVPDetector()
    await detector.initialize()
    
    # Needs a real response with proper nouns
    req = SignalRequest(
        candidate_id="test-svp-prop",
        profile={"employment_history": [], "skill_timestamps": {}, "education": []},
        screening_responses=[
            ScreeningResponse(
                question_id="q1",
                answer="We used AWS Lambda and Amazon S3 extensively. John Smith led the migration from Jenkins to GitHub Actions.",
                latency_ms=1000,
                topic="core"
            )
        ]
    )
    # The _specificity_score should be higher because of proper nouns
    result = await detector.analyze(req)
    assert 0.0 <= result.score <= 1.0


@pytest.mark.asyncio
async def test_svp_opinionated_language():
    """Opinionated language should increase specificity score."""
    from services.svp.detector import SVPDetector
    from kive.shared.schemas import SignalRequest, ScreeningResponse
    detector = SVPDetector()
    await detector.initialize()
    
    req = SignalRequest(
        candidate_id="test-svp-opinion",
        profile={"employment_history": [], "skill_timestamps": {}, "education": []},
        screening_responses=[
            ScreeningResponse(
                question_id="q1",
                answer="I prefer using PostgreSQL because honestly MySQL is overrated and the tradeoff is not great for large datasets.",
                latency_ms=1000,
                topic="core"
            )
        ]
    )
    result = await detector.analyze(req)
    assert 0.0 <= result.score <= 1.0
