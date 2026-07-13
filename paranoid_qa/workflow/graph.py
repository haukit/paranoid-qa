"""The composition root: assemble one compiled LangGraph from the two path fragments.

START -> route
    -> "specific"  -> specific_retrieve -> specific_grade
        -> "yes" -> specific_generate -> specific_verify -> accept | revise | abstain
        -> "no"  -> specific_rewrite -> specific_retrieve
    -> "aggregate" -> aggregate_answer -> aggregate_verify -> accept | abstain

accept -> status="answered" (verification passed) -> END
abstain -> status="abstained" (couldn't ground within budget) -> END

This is the only module that imports both path packages. Each path registers its own nodes and
loops via a path-local wiring function; no compiled subgraphs are used.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from paranoid_qa.aggregate.graph import add_aggregate_path
from paranoid_qa.contracts.state import GraphState
from paranoid_qa.llm.policy import validate_model_policy
from paranoid_qa.specific.graph import add_specific_path
from paranoid_qa.workflow.outcomes import abstain, accept
from paranoid_qa.workflow.routing import route


def build_graph(*, verify_enabled: bool = True):
    """Validate model policy and compile the routed specific + aggregate graph."""
    validate_model_policy()

    graph = StateGraph(GraphState)
    graph.add_node("route", route)
    graph.add_node("accept", accept)
    graph.add_node("abstain", abstain)

    specific_entry = add_specific_path(
        graph, verify_enabled=verify_enabled, accept_target="accept", abstain_target="abstain"
    )
    aggregate_entry = add_aggregate_path(
        graph, verify_enabled=verify_enabled, accept_target="accept", abstain_target="abstain"
    )

    graph.add_edge(START, "route")
    graph.add_conditional_edges(
        "route",
        lambda state: state["route"],
        {"specific": specific_entry, "aggregate": aggregate_entry},
    )
    graph.add_edge("accept", END)
    graph.add_edge("abstain", END)

    return graph.compile()
