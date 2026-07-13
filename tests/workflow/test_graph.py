"""Graph routing logic and compiled topology (no LLM calls)."""

from paranoid_qa.aggregate.graph import _after_verify as aggregate_after_verify
from paranoid_qa.config import settings
from paranoid_qa.contracts.specific import SpecificAnswer, SpecificClaim, SpecificClaimVerdict
from paranoid_qa.contracts.state import GraphState
from paranoid_qa.specific.graph import _after_grade, _after_verify
from paranoid_qa.workflow.graph import build_graph


def _verdict(kind: str) -> SpecificClaimVerdict:
    return SpecificClaimVerdict(verdict=kind, explanation="e")  # type: ignore[arg-type]


def test_after_grade_generates_when_relevant():
    assert _after_grade({"specific_grade": "yes"}) == "generate"


def test_after_grade_rewrites_then_generates_when_budget_spent(monkeypatch):
    monkeypatch.setattr(settings, "max_attempts", 2)
    assert _after_grade({"specific_grade": "no", "specific_retrieval_attempts": 0}) == "rewrite"
    assert _after_grade({"specific_grade": "no", "specific_retrieval_attempts": 2}) == "generate"


def test_after_verify_accepts_when_passed():
    assert _after_verify({"verification_passed": True}) == "accept"


def test_after_verify_revises_then_abstains_on_separate_budget(monkeypatch):
    monkeypatch.setattr(settings, "max_attempts", 2)
    answer = SpecificAnswer(claims=[SpecificClaim(text="a", quote="q")])
    state: GraphState = {
        "verification_passed": False,
        "answer": answer,
        "specific_verdicts": [_verdict("unsupported")],
        "specific_revision_attempts": 0,
    }
    assert _after_verify(state) == "revise"
    state["specific_revision_attempts"] = 2
    assert _after_verify(state) == "abstain"


def test_after_verify_abstains_on_all_irrelevant():
    answer = SpecificAnswer(claims=[SpecificClaim(text="a", quote="q")])
    state: GraphState = {
        "verification_passed": False,
        "answer": answer,
        "specific_verdicts": [_verdict("irrelevant")],
        "specific_revision_attempts": 0,
    }
    assert _after_verify(state) == "abstain"


def test_after_verify_abstains_on_empty_claims():
    state: GraphState = {
        "verification_passed": False,
        "answer": SpecificAnswer(claims=[]),
        "specific_verdicts": [],
    }
    assert _after_verify(state) == "abstain"


def test_retrieval_and_revision_budgets_are_separate(monkeypatch):
    # Retrieval budget spent must not force an abstention on the revision loop and vice versa.
    monkeypatch.setattr(settings, "max_attempts", 2)
    answer = SpecificAnswer(claims=[SpecificClaim(text="a", quote="q")])
    state: GraphState = {
        "verification_passed": False,
        "answer": answer,
        "specific_verdicts": [_verdict("unsupported")],
        "specific_retrieval_attempts": 5,  # retrieval budget blown
        "specific_revision_attempts": 0,  # revision budget fresh
    }
    assert _after_verify(state) == "revise"


def test_aggregate_after_verify():
    assert aggregate_after_verify({"verification_passed": True}) == "accept"
    assert aggregate_after_verify({"verification_passed": False}) == "abstain"


def _edges(verify_enabled: bool) -> set[tuple[str, str]]:
    g = build_graph(verify_enabled=verify_enabled).get_graph()
    return {(e.source, e.target) for e in g.edges}


def test_both_path_entries_registered():
    nodes = build_graph().get_graph().nodes
    assert "specific_retrieve" in nodes
    assert "aggregate_answer" in nodes


def test_successful_verification_routes_through_accept_node():
    g = build_graph().get_graph()
    assert "accept" in g.nodes and "abstain" in g.nodes
    edges = {(e.source, e.target) for e in g.edges}
    assert ("specific_verify", "accept") in edges
    assert ("aggregate_verify", "accept") in edges


def test_verify_disabled_still_routes_generation_to_accept():
    edges = _edges(verify_enabled=False)
    nodes = build_graph(verify_enabled=False).get_graph().nodes
    assert "specific_verify" not in nodes
    assert ("specific_generate", "accept") in edges
    assert ("aggregate_answer", "accept") in edges
