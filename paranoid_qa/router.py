from __future__ import annotations

from typing import cast

from paranoid_qa.models import make_structured_llm
from paranoid_qa.schemas import GraphState, Route

ROUTER_SYSTEM = """Classify the user's question into one of two strategies.

- "specific": about a particular event, document, or detail – answerable from a few retrieved
  passages. Examples: "What caused the X accident?", "Where did flight Y crash?"
- "aggregate": about the corpus as a whole – counts, trends, themes, or comparisons across
  many documents. Examples: "How many accidents involved Z?", "What factors recur across
  these reports?"

Choose "aggregate" only when answering requires looking across the whole corpus at once."""


async def route(state: GraphState) -> GraphState:
    """Classify the question."""
    classifier = make_structured_llm(Route)
    messages = [
        ("system", ROUTER_SYSTEM),
        ("human", state["question"]),
    ]
    result = cast(Route, await classifier.ainvoke(messages))
    return {"route": result.kind}
