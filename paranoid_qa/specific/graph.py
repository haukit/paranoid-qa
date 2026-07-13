"""Specific-path graph registration and routing.

Owns the specific nodes, the retrieval-rewrite loop, and the verify/revise/abstain loop. Target
node names for the shared terminals are passed in (not imported) to keep the dependency pointing
outward: the path does not import the workflow package.
"""

from __future__ import annotations

from langgraph.graph import StateGraph

from paranoid_qa.config import settings
from paranoid_qa.contracts.state import GraphState
from paranoid_qa.specific.nodes import generate, grade, retrieve, rewrite
from paranoid_qa.specific.verification import specific_verify

ENTRY = "specific_retrieve"


def _after_grade(state: GraphState) -> str:
    """Answer, or rewrite the question and retrieve again until the retrieval budget runs out."""
    if state["specific_grade"] == "yes":
        return "generate"
    if state.get("specific_retrieval_attempts", 0) >= settings.max_attempts:
        return "generate"  # retrieval budget spent; answer with what we have
    return "rewrite"


def _after_verify(state: GraphState) -> str:
    """Accept, revise, or abstain based on the verdicts and the revision budget."""
    if state["verification_passed"]:
        return "accept"

    answer = state.get("answer")
    if answer is None or not answer.claims:
        return "abstain"  # nothing to ground

    verdicts = state.get("specific_verdicts") or []
    if verdicts and all(v.verdict == "irrelevant" for v in verdicts):
        # every quote is off-subject: the question's premise isn't in the corpus,
        # so regenerating from the same retrieved docs won't help
        return "abstain"

    if state.get("specific_revision_attempts", 0) >= settings.max_attempts:
        return "abstain"  # revision budget spent; don't emit an unsupported answer

    # An unfaithful verdict on already-graded retrieval is a generation fault -> regenerate.
    return "revise"


def add_specific_path(
    graph: StateGraph,
    *,
    verify_enabled: bool,
    accept_target: str,
    abstain_target: str,
) -> str:
    """Register the specific path's nodes and edges; return its entry node name."""
    graph.add_node("specific_retrieve", retrieve)
    graph.add_node("specific_grade", grade)
    graph.add_node("specific_rewrite", rewrite)
    graph.add_node("specific_generate", generate)

    graph.add_edge("specific_retrieve", "specific_grade")
    graph.add_conditional_edges(
        "specific_grade",
        _after_grade,
        {"generate": "specific_generate", "rewrite": "specific_rewrite"},
    )
    graph.add_edge("specific_rewrite", "specific_retrieve")

    if verify_enabled:
        graph.add_node("specific_verify", specific_verify)
        graph.add_edge("specific_generate", "specific_verify")
        graph.add_conditional_edges(
            "specific_verify",
            _after_verify,
            {
                "accept": accept_target,
                "revise": "specific_generate",
                "abstain": abstain_target,
            },
        )
    else:
        # No verification: still route through the accept terminal so status is set.
        graph.add_edge("specific_generate", accept_target)

    return ENTRY
