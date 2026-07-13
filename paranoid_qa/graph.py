"""Assemble the LangGraph app.

START -> route
    -> "specific" -> retrieve -> grade
        -> "yes" -> generate -> verify -> accept|revise|abstain
        -> "no"  -> rewrite -> retrieve
    -> "aggregate" -> aggregate -> verify -> accept|abstain

verify ends at one of two terminals, both -> END:
    accept  -> status="answered"  (verification passed; accept <=> faithful)
    abstain -> status="abstained" (couldn't ground the answer within the retry budget)

All correction loops (rewrite / revise) are bounded by settings.max_attempts.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from paranoid_qa.aggregate import aggregate_answer
from paranoid_qa.config import settings
from paranoid_qa.nodes import generate, grade, retrieve, rewrite
from paranoid_qa.router import route
from paranoid_qa.schemas import GraphState
from paranoid_qa.verify import abstain, accept, verify


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

    answer = state.get("answer")
    if answer is None or not answer.claims:
        # nothing to ground
        return "abstain"

    verdicts = state.get("verdicts") or []
    if verdicts and all(v.verdict == "irrelevant" for v in verdicts):
        # every quote is off-subject: the question's premise isn't in the corpus,
        # so regenerating from the same retrieved docs won't help
        return "abstain"

    if state.get("attempts", 0) >= settings.max_attempts:
        # specific path: retries exhausted, don't emit an unsupported answer
        return "abstain"
    if state["route"] == "aggregate":
        # no corrective loop on the aggregate path
        return "abstain"

    # Specific path: retrieval quality is already gated by `grade`, so any unfaithful verdict
    # is a generation fault -> regenerate with feedback.
    return "revise"


def build_graph(verify_enabled: bool = True):
    g = StateGraph(GraphState)

    g.add_node("route", route)
    g.add_node("retrieve", retrieve)
    g.add_node("grade", grade)
    g.add_node("rewrite", rewrite)
    g.add_node("generate", generate)
    g.add_node("verify", verify)
    g.add_node("accept", accept)
    g.add_node("abstain", abstain)
    g.add_node("aggregate", aggregate_answer)

    # classify, then branch
    g.add_edge(START, "route")
    g.add_conditional_edges(
        "route", lambda s: s["route"], {"specific": "retrieve", "aggregate": "aggregate"}
    )

    # prompt rewrite in specific path
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", _decide, {"generate": "generate", "rewrite": "rewrite"})
    g.add_edge("rewrite", "retrieve")

    # verify + correction routing
    if verify_enabled:
        g.add_edge("generate", "verify")
        g.add_edge("aggregate", "verify")
        g.add_conditional_edges(
            "verify",
            _verify_route,
            {
                "accept": END,
                "revise": "generate",  # back to the generator, same docs
                "abstain": "abstain",
            },
        )
        g.add_edge("accept", END)
        g.add_edge("abstain", END)
    else:
        g.add_edge("generate", END)
        g.add_edge("aggregate", END)

    return g.compile()
