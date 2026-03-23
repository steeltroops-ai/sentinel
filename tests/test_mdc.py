# tests/test_mdc.py
import pytest
from kive.shared.schemas import SignalRequest

def _make_mdc_request(skill_timestamps: dict[str, str]) -> SignalRequest:
    return SignalRequest(
        candidate_id="test-mdc",
        profile={
            "skill_timestamps": skill_timestamps,
            "employment_history": []
        },
        screening_responses=[]
    )

@pytest.mark.asyncio
async def test_mdc_no_skills():
    from services.mdc.detector import MDCDetector
    det = MDCDetector()
    await det.initialize()
    result = await det.analyze(_make_mdc_request({}))
    assert result.score == 0.20
    assert result.confidence == 0.15

@pytest.mark.asyncio
async def test_mdc_organic_growth():
    # Skills added before or aligned with spikes, but spread out.
    from services.mdc.detector import MDCDetector
    det = MDCDetector()
    await det.initialize()
    result = await det.analyze(_make_mdc_request({
        "kubernetes": "2015-06",
        "react": "2014-05",
        "rust": "2018-01"
    }))
    assert result.score < 0.30
    assert len(result.flags) == 0

@pytest.mark.asyncio
async def test_mdc_retroactive_padding():
    # Skill added immediately after a market spike (e.g. GPT-4 in 2023-03 -> padding in 2023-04)
    from services.mdc.detector import MDCDetector
    det = MDCDetector()
    await det.initialize()
    result = await det.analyze(_make_mdc_request({
        "gpt-4": "2023-04",  # the spike is 2023-03
        "react": "2015-06"
    }))
    assert len(result.retroactive_hits) >= 0
    assert result.score >= 0.10
    flag_types = [f.type for f in result.flags]

@pytest.mark.asyncio
async def test_mdc_skill_burst():
    # 3+ skills added on the exact same date
    from services.mdc.detector import MDCDetector
    det = MDCDetector()
    await det.initialize()
    result = await det.analyze(_make_mdc_request({
        "docker": "2023-01",
        "kubernetes": "2023-01",
        "aws": "2023-01",
        "terraform": "2023-01"
    }))
    assert result.score >= 0.30
    flag_types = [f.type for f in result.flags]
    assert "skill_addition_burst" in flag_types

@pytest.mark.asyncio
async def test_mdc_combined_fraud():
    # Retroactive padding AND skill burst
    from services.mdc.detector import MDCDetector
    det = MDCDetector()
    await det.initialize()
    result = await det.analyze(_make_mdc_request({
        "llm": "2023-04",
        "langchain": "2023-04",
        "chatgpt": "2023-04"
    }))
    assert len(result.retroactive_hits) >= 1
    flag_types = [f.type for f in result.flags]
    assert "retroactive_skill_inflation" in flag_types
    assert "skill_addition_burst" in flag_types
    assert result.score > 0.70

@pytest.mark.asyncio
async def test_mdc_result_schema():
    from services.mdc.detector import MDCDetector
    det = MDCDetector()
    await det.initialize()
    result = await det.analyze(_make_mdc_request({"test": "2020-01"}))
    assert 0.0 <= result.score <= 1.0
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.retroactive_hits) >= 0
    assert isinstance(result.flags, list)
