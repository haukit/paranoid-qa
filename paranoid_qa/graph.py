"""Assemble the LangGraph app.

START -> retrieve -> grade
    -> "yes" -> generate -> verify
        -> accept -> END
        -> revise -> generate
        -> re_retrieve -> retrieve
    -> "no" -> rewrite -> retrieve

All correction loops (rewrite / revise / re_retrieve) are bounded by settings.max_attempts.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from paranoid_qa.aggregate import aggregate_answer
from paranoid_qa.config import settings
from paranoid_qa.nodes import generate, grade, retrieve, rewrite
from paranoid_qa.router import route
from paranoid_qa.schemas import GraphState
from paranoid_qa.verify import verify, verify_aggregate


def _decide(state: GraphState) -> str:
    """Conditional edge after `grade`: answer, or loop back to rewrite question until the budget runs out."""
    if state["grade"] == "yes":
        return "generate"
    if state.get("attempts", 0) >= settings.max_attempts:
        return "generate"  # retries exhausted, answer with what we have
    return "rewrite"


def _verify_route(state: GraphState) -> str:
    if state["faithful"]:
        return "accept"
    if state.get("attempts", 0) >= settings.max_attempts:
        # Budget spent -> return best-effort answer
        return "accept"
    if any(v.verdict == "fabricated" for v in state["verdicts"]):
        # Quotes are not in the docs -> get docs again
        return "re_retrieve"

    # Quotes real but do not support claim -> regenerate
    return "revise"


def build_graph():
    g = StateGraph(GraphState)

    g.add_node("route", route)
    g.add_node("retrieve", retrieve)
    g.add_node("grade", grade)
    g.add_node("rewrite", rewrite)
    g.add_node("generate", generate)
    g.add_node("verify", verify)
    g.add_node("aggregate", aggregate_answer)
    g.add_node("verify_aggregate", verify_aggregate)

    # classify, then branch
    g.add_edge(START, "route")
    g.add_conditional_edges(
        "route", lambda s: s["route"], {"specific": "retrieve", "aggregate": "aggregate"}
    )

    # specific path
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", _decide, {"generate": "generate", "rewrite": "rewrite"})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", "verify")
    g.add_conditional_edges(
        "verify",
        _verify_route,
        {
            "accept": END,
            "revise": "generate",  # back to the generator, same docs
            "re_retrieve": "retrieve",  # back to the retrieval for fresh docs
        },
    )

    # aggregate path
    g.add_edge("aggregate", "verify_aggregate")
    g.add_edge("verify_aggregate", END)

    return g.compile()
