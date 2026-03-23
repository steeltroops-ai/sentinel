# tests/test_ccs.py
# Full test coverage for CCS (Cross-Candidate Similarity)

import pytest
from kive.shared.schemas import SignalRequest, ScreeningResponse


def _make_candidate(cid: str, answers: list[tuple[str, str]]) -> SignalRequest:
    """answers = [(answer_text, question_id)]."""
    return SignalRequest(
        candidate_id=cid,
        profile={},
        screening_responses=[
            ScreeningResponse(
                question_id=qid, answer=text, latency_ms=5000, topic="core"
            )
            for text, qid in answers
        ],
    )


@pytest.mark.asyncio
async def test_ccs_no_responses():
    """No data -> neutral."""
    from services.ccs.detector import CCSDetector
    det = CCSDetector()
    await det.initialize()
    result = await det.analyze(SignalRequest(candidate_id="x", profile={}))
    assert result.score == 0.10
    assert result.confidence == 0.10


@pytest.mark.asyncio
async def test_ccs_first_candidate_no_cluster():
    """First candidate in empty index -> no cluster possible."""
    from services.ccs.detector import CCSDetector
    det = CCSDetector()
    await det.initialize()
    result = await det.analyze(_make_candidate("c1", [
        ("Kubernetes is a container orchestration platform used in production.", "q1")
    ]))
    assert result.cluster_size == 0
    assert len(result.flags) == 0


@pytest.mark.asyncio
async def test_ccs_template_cluster_detection():
    """4 candidates submit near-identical answers -> cluster flag."""
    from services.ccs.detector import CCSDetector
    det = CCSDetector()
    await det.initialize()

    template_answer = (
        "Kubernetes provides a robust container orchestration platform that enables "
        "automated deployment, scaling, and management of containerized applications. "
        "Key features include self-healing, load balancing, and rolling updates."
    )

    # Submit 4 near-identical answers
    for i in range(4):
        # Minor variations to simulate slight edits
        answer = template_answer if i < 3 else template_answer.replace("robust", "powerful")
        await det.analyze(_make_candidate(f"c{i}", [(answer, "q1")]))

    # 5th candidate with same template
    result = await det.analyze(_make_candidate("c4", [(template_answer, "q1")]))
    assert result.cluster_size >= 3
    assert len(result.flags) > 0
    flag_types = [f.type for f in result.flags]
    assert "template_cluster_detected" in flag_types
    assert "q1" in result.flagged_questions


@pytest.mark.asyncio
async def test_ccs_unique_answers_no_cluster():
    """Different candidates with unique answers -> no cluster."""
    from services.ccs.detector import CCSDetector
    det = CCSDetector()
    await det.initialize()

    unique_answers = [
        "I prefer using Helm charts for K8s deployments because the templating gives me version control.",
        "We migrated to ArgoCD for GitOps after our Jenkins pipeline kept failing on rollbacks.",
        "Our team uses Kustomize overlays instead of Helm because we find YAML more transparent.",
        "I set up K8s from scratch on bare metal for our on-prem datacenter using kubeadm.",
    ]

    for i, answer in enumerate(unique_answers):
        await det.analyze(_make_candidate(f"c{i}", [(answer, "q1")]))

    result = await det.analyze(_make_candidate("c_unique", [
        ("My approach to Kubernetes is to use managed GKE with Terraform for infra-as-code.", "q1")
    ]))
    assert result.cluster_size == 0
    assert len(result.flagged_questions) == 0


@pytest.mark.asyncio
async def test_ccs_jaccard_similarity():
    """Internal Jaccard similarity should be correct."""
    from services.ccs.detector import CCSDetector
    det = CCSDetector()
    a = {1, 2, 3, 4, 5}
    b = {3, 4, 5, 6, 7}
    sim = det._jaccard_similarity(a, b)
    # Intersection: {3,4,5} = 3, Union: {1,2,3,4,5,6,7} = 7
    assert abs(sim - 3/7) < 0.01


@pytest.mark.asyncio
async def test_ccs_jaccard_empty_sets():
    """Empty set -> 0 similarity."""
    from services.ccs.detector import CCSDetector
    det = CCSDetector()
    assert det._jaccard_similarity(set(), {1, 2, 3}) == 0.0
    assert det._jaccard_similarity(set(), set()) == 0.0


@pytest.mark.asyncio
async def test_ccs_shingle_function():
    """Shingling should produce consistent sets."""
    from services.ccs.detector import CCSDetector
    det = CCSDetector()
    s1 = det._shingle("hello world")
    s2 = det._shingle("hello world")
    assert s1 == s2
    s3 = det._shingle("completely different text")
    # Should have low overlap
    intersection = len(s1 & s3)
    union = len(s1 | s3)
    assert intersection / max(union, 1) < 0.3


@pytest.mark.asyncio
async def test_ccs_result_schema():
    """CCSResult fields validation."""
    from services.ccs.detector import CCSDetector
    det = CCSDetector()
    await det.initialize()
    result = await det.analyze(_make_candidate("test", [
        ("This is a test answer for schema validation.", "q_schema")
    ]))
    assert 0.0 <= result.score <= 1.0
    assert 0.0 <= result.confidence <= 1.0
    assert result.max_similarity >= 0.0
    assert result.cluster_size >= 0
    assert isinstance(result.flagged_questions, list)
    assert isinstance(result.flags, list)
