"""Aggregate-path answer node: query LightRAG, then decompose the prose into checkable claims."""

from __future__ import annotations

from typing import cast

from paranoid_qa.aggregate.retrieval import query
from paranoid_qa.contracts.aggregate import AggregateAnswer, AggregateClaim
from paranoid_qa.contracts.state import GraphState
from paranoid_qa.llm.factory import make_generator

DECOMPOSE_SYSTEM = """You are given an ANSWER to a corpus-level question. Break it into atomic
factual claims: each a single, self-contained statement that can be checked on its own. Copy the
meaning faithfully — do not add, remove, or embellish information, and ignore any inline citation
markers like [1]. Return only the claims."""


def _decompose(answer_text: str) -> list[AggregateClaim]:
    """Split a corpus-level prose answer into atomic claims (grounded by references, not quotes)."""
    gen = make_generator(AggregateAnswer)
    messages = [("system", DECOMPOSE_SYSTEM), ("human", answer_text)]
    decomposed = cast(AggregateAnswer, gen.invoke(messages))
    return decomposed.claims


def aggregate_answer(state: GraphState) -> GraphState:
    """Answer a corpus-level question, then structure the prose into verifiable claims."""
    result = query(state["question"])
    answer = AggregateAnswer(
        claims=_decompose(result.answer_text),
        references=result.references,
    )
    return {"answer": answer, "aggregate_context": result.context}
