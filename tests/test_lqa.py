# tests/test_lqa.py
# Full test coverage for LQA (Linguistic Quality Assurance)

import pytest
from kive.shared.schemas import SignalRequest, ScreeningResponse


def _make_text_request(answers: list[tuple[str, str]]) -> SignalRequest:
    """Build a SignalRequest with screening answers. answers = [(answer_text, topic)]."""
    return SignalRequest(
        candidate_id="test-lqa",
        profile={},
        screening_responses=[
            ScreeningResponse(
                question_id=f"q{i}", answer=a, latency_ms=5000, topic=t
            )
            for i, (a, t) in enumerate(answers)
        ],
    )


@pytest.mark.asyncio
async def test_lqa_no_responses():
    """No screening data -> neutral."""
    from services.lqa.detector import LQADetector
    det = LQADetector()
    await det.initialize()
    result = await det.analyze(SignalRequest(candidate_id="x", profile={}))
    assert result.score == 0.5
    assert result.confidence == 0.15


@pytest.mark.asyncio
async def test_lqa_gpt_artifact_phrases():
    """Text loaded with GPT artifact phrases -> high score."""
    from services.lqa.detector import LQADetector
    det = LQADetector()
    await det.initialize()
    gpt_text = (
        "It's worth noting that Kubernetes plays a crucial role in modern infrastructure. "
        "Furthermore, it is important to note that containerization provides a seamless "
        "experience. Additionally, there are various approaches one could leverage to "
        "streamline deployment. Certainly, the state-of-the-art tooling makes this "
        "comprehensive and robust. Moreover, when it comes to orchestration, one of the "
        "key aspects is the ability to delve into the underlying architecture."
    )
    result = await det.analyze(_make_text_request([(gpt_text, "core")]))
    assert result.artifact_density > 1.0
    flag_types = [f.type for f in result.flags]
    assert "gpt_artifact_phrases" in flag_types
    assert result.score > 0.35


@pytest.mark.asyncio
async def test_lqa_transition_overuse():
    """Excessive transition phrases -> flag."""
    from services.lqa.detector import LQADetector
    det = LQADetector()
    await det.initialize()
    text = (
        "First thing to know is containers.\n"
        "Furthermore, networking is important.\n"
        "Additionally, storage management matters.\n"
        "Moreover, security cannot be overlooked.\n"
        "In conclusion, Kubernetes is critical."
    )
    result = await det.analyze(_make_text_request([(text, "core")]))
    assert result.transition_ratio > 0.20
    flag_types = [f.type for f in result.flags]
    assert "transition_overuse" in flag_types


@pytest.mark.asyncio
async def test_lqa_human_writing():
    """Natural human technical writing -> low score."""
    from services.lqa.detector import LQADetector
    det = LQADetector()
    await det.initialize()
    human_text = (
        "I prefer using Redis over Memcached for session caching because the "
        "persistence options give me more flexibility when things go wrong. "
        "We had a bad week last March when our Elasticache cluster ran out of "
        "memory at 3am because I picked the wrong eviction policy. allkeys-lru "
        "would have been correct but I went with volatile-lru and half our keys "
        "weren't set to expire. My fault entirely. "
        "Honestly the documentation was confusing about this at the time."
    )
    result = await det.analyze(_make_text_request([(human_text, "backend")]))
    assert result.score < 0.60


@pytest.mark.asyncio
async def test_lqa_syntactic_variance():
    """Multiple responses with uniform complexity -> low variance -> fraud signal."""
    from services.lqa.detector import LQADetector
    det = LQADetector()
    await det.initialize()
    # All answers have similar sentence structure
    answers = [
        ("The system uses containers for deployment. Kubernetes orchestrates them effectively. "
         "Monitoring is handled by Prometheus and Grafana.", "core"),
        ("The pipeline processes data in batches. Spark handles the computation efficiently. "
         "Results are stored in a data warehouse.", "data"),
        ("The frontend uses React for rendering. Components are structured modularly. "
         "State management is handled by Redux.", "edge"),
    ]
    result = await det.analyze(_make_text_request(answers))
    assert 0.0 <= result.syntactic_variance


@pytest.mark.asyncio
async def test_lqa_result_schema():
    """LQAResult must have all fields in valid ranges."""
    from services.lqa.detector import LQADetector
    det = LQADetector()
    await det.initialize()
    result = await det.analyze(_make_text_request([
        ("This is a test answer about ML pipelines.", "core")
    ]))
    assert 0.0 <= result.score <= 1.0
    assert 0.0 <= result.confidence <= 1.0
    assert result.artifact_density >= 0.0
    assert result.syntactic_variance >= 0.0
    assert result.transition_ratio >= 0.0
    assert result.hedging_density >= 0.0
    assert isinstance(result.flags, list)


@pytest.mark.asyncio
async def test_lqa_syllable_counter():
    """Internal syllable counter should give reasonable estimates."""
    from services.lqa.detector import LQADetector
    det = LQADetector()
    assert det._count_syllables("kubernetes") >= 3
    assert det._count_syllables("the") == 1
    assert det._count_syllables("infrastructure") >= 4
    assert det._count_syllables("") == 1


@pytest.mark.asyncio
async def test_lqa_flesch_kincaid():
    """FK proxy should return a numeric grade level."""
    from services.lqa.detector import LQADetector
    det = LQADetector()
    fk = det._flesch_kincaid_proxy(
        "This is a simple sentence. This is another one."
    )
    assert isinstance(fk, float)
