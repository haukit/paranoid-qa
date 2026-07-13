"""Aggregate-path graph registration and routing.

Current flow: aggregate_answer -> aggregate_verify -> accept | abstain. A future corrective loop
belongs here (verify -> answer), addable without touching the workflow composition root.
"""

from __future__ import annotations

from langgraph.graph import StateGraph

from paranoid_qa.aggregate.nodes import aggregate_answer
from paranoid_qa.aggregate.verification import aggregate_verify
from paranoid_qa.contracts.state import GraphState

ENTRY = "aggregate_answer"


def _after_verify(state: GraphState) -> str:
    """Accept when verified, otherwise abstain (no corrective loop yet)."""
    return "accept" if state["verification_passed"] else "abstain"


def add_aggregate_path(
    graph: StateGraph,
    *,
    verify_enabled: bool,
    accept_target: str,
    abstain_target: str,
) -> str:
    """Register the aggregate path's nodes and edges; return its entry node name."""
    graph.add_node("aggregate_answer", aggregate_answer)

    if verify_enabled:
        graph.add_node("aggregate_verify", aggregate_verify)
        graph.add_edge("aggregate_answer", "aggregate_verify")
        graph.add_conditional_edges(
            "aggregate_verify",
            _after_verify,
            {"accept": accept_target, "abstain": abstain_target},
        )
    else:
        graph.add_edge("aggregate_answer", accept_target)

    return ENTRY
