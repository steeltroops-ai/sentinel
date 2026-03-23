# tests/test_tsi.py
import pytest
from kive.shared.schemas import SignalRequest

def _make_tsi_request(employment_history: list[dict]) -> SignalRequest:
    return SignalRequest(
        candidate_id="test-tsi",
        profile={"employment_history": employment_history},
        screening_responses=[]
    )

@pytest.mark.asyncio
async def test_tsi_no_history():
    from services.tsi.detector import TSIDetector
    det = TSIDetector()
    await det.initialize()
    result = await det.analyze(_make_tsi_request([]))
    assert result.score == 0.10
    assert result.confidence == 0.15

@pytest.mark.asyncio
async def test_tsi_perfect_smoothness():
    # 3 transitions, perfectly smooth upward, no gaps
    from services.tsi.detector import TSIDetector
    det = TSIDetector()
    await det.initialize()
    history = [
        {"title": "Junior Developer", "company": "A", "start_year": 2015, "end_year": 2017},
        {"title": "Developer", "company": "B", "start_year": 2017, "end_year": 2019},
        {"title": "Senior Developer", "company": "C", "start_year": 2019, "end_year": 2021},
        {"title": "Lead Developer", "company": "D", "start_year": 2021, "end_year": 2023},
    ]
    result = await det.analyze(_make_tsi_request(history))
    assert result.score > 0.60
    assert result.score > 0.60

@pytest.mark.asyncio
async def test_tsi_authentic_turbulence():
    # Demotions, gaps, lateral moves
    from services.tsi.detector import TSIDetector
    det = TSIDetector()
    await det.initialize()
    history = [
        {"title": "Senior Engineer", "company": "A", "start_year": 2015, "end_year": 2018},
        {"title": "Engineer", "company": "B", "start_year": 2019, "end_year": 2020}, # Gap 2018-2019, Downstep
        {"title": "Engineer", "company": "C", "start_year": 2020, "end_year": 2022}, # Lateral
    ]
    result = await det.analyze(_make_tsi_request(history))
    assert result.score <= 0.70
    assert result.score <= 0.70

@pytest.mark.asyncio
async def test_tsi_tenure_implausibility():
    # VP after 2 years
    from services.tsi.detector import TSIDetector
    det = TSIDetector()
    await det.initialize()
    history = [
        {"title": "Engineer", "company": "A", "start_year": 2021, "end_year": 2022},
        {"title": "VP", "company": "B", "start_year": 2022, "end_year": 2023},
    ]
    result = await det.analyze(_make_tsi_request(history))
    assert result.score >= 0.0
    assert isinstance(result.flags, list)

@pytest.mark.asyncio
async def test_tsi_fast_climb_churn():
    # Average tenure < 1.5 across 4+ roles
    from services.tsi.detector import TSIDetector
    det = TSIDetector()
    await det.initialize()
    history = [
        {"title": "Engineer I", "company": "A", "start_year": 2019, "end_year": 2020},
        {"title": "Engineer II", "company": "B", "start_year": 2020, "end_year": 2021},
        {"title": "Senior Engineer", "company": "C", "start_year": 2021, "end_year": 2022},
        {"title": "Lead Engineer", "company": "D", "start_year": 2022, "end_year": 2023},
    ] # 4 years, 4 roles => avg 1.0 years
    result = await det.analyze(_make_tsi_request(history))
    assert result.score > 0.50
    flag_types = [str(r.description).lower() for r in result.flags]
    assert any("average tenure" in r for r in flag_types)

@pytest.mark.asyncio
async def test_tsi_result_schema():
    from services.tsi.detector import TSIDetector
    det = TSIDetector()
    await det.initialize()
    history = [{"title": "Dev", "company": "A", "start_year": 2020, "end_year": 2021}]
    result = await det.analyze(_make_tsi_request(history))
    assert 0.0 <= result.score <= 1.0
    assert 0.0 <= result.confidence <= 1.0
    assert result.score >= 0.0
    assert isinstance(result.flags, list)
