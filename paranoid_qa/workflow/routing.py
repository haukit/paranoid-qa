"""The router: classify a question as specific or aggregate, then branch.

Uses the shared model factory but imports neither path package.
"""

from __future__ import annotations

from typing import cast

from pydantic import BaseModel, Field

from paranoid_qa.contracts.common import RouteKind
from paranoid_qa.contracts.state import GraphState
from paranoid_qa.llm.factory import make_router

ROUTER_SYSTEM = """Classify the user's question into one of two strategies.

- "specific": about a particular event, document, or detail – answerable from a few retrieved
  passages. Examples: "What caused the X accident?", "Where did flight Y crash?"
- "aggregate": about the corpus as a whole – counts, trends, themes, or comparisons across
  many documents. Examples: "How many accidents involved Z?", "What factors recur across
  these reports?"

Choose "aggregate" only when answering requires looking across the whole corpus at once."""


class Route(BaseModel):
    """Router decision: which path answers this question."""

    kind: RouteKind = Field(
        description=(
            "'specific' = about a particular event/document/detail, answerable from a few "
            "retrieved passages; 'aggregate' = corpus-level counts, trends, themes, or "
            "comparisons across many documents."
        )
    )


async def route(state: GraphState) -> GraphState:
    """Classify the question."""
    classifier = make_router(Route)
    messages = [("system", ROUTER_SYSTEM), ("human", state["question"])]
    result = cast(Route, await classifier.ainvoke(messages))
    return {"route": result.kind}
