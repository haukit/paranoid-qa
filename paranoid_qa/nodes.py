"""Core LangGraph nodes for the retrieval-and-answer loop."""

from __future__ import annotations

from typing import cast

from paranoid_qa.index import get_retriever
from paranoid_qa.models import make_llm, make_structured_llm
from paranoid_qa.schemas import Answer, Grade, GraphState, RetrievedChunk

GENERATE_SYSTEM = """You answer questions strictly from the provided sources.

Rules:
- Use ONLY the sources. Do not add outside knowledge or assumptions.
- Break your answer into atomic claims: each claim's `text` is ONE self-contained fact.
- For each claim, set `quote` to a span copied VERBATIM (character-for-character) from the
  sources that supports `text`. Never paraphrase the quote — copy it exactly.
- If the sources do not answer the question, return an empty list of claims."""

GRADE_SYSTEM = """You assess whether retrieved documents are relevant to a user's question.
Set relevant=true ONLY if the documents contain information needed to answer the question.
Otherwise set relevant=false."""

REWRITE_SYSTEM = """The retrieved documents did not answer the user's question well.
Rewrite the question to be clearer and more retrievable: expand abbreviations, add likely
synonyms/keywords, and make the information need explicit. Return ONLY the rewritten question."""


def _format_chunks(chunks: list[RetrievedChunk]) -> str:
    """Join the retrieved chunks' text into one context block."""
    return "\n\n---\n\n".join(c["text"] for c in chunks)


def retrieve(state: GraphState) -> GraphState:
    """Fetch candidate chunks for the question"""
    nodes = get_retriever().retrieve(state["question"])
    chunks: list[RetrievedChunk] = [
        {
            "text": n.text,
            "document": n.metadata.get("file_name", "?"),
            "page": n.metadata.get("page_label"),
        }
        for n in nodes
    ]
    return {"chunks": chunks}


def grade(state: GraphState) -> GraphState:
    """Judge whether the retrieved docs are good enough to answer"""
    question = state["question"]
    context = _format_chunks(state["chunks"])

    grader = make_structured_llm(Grade)
    messages = [
        ("system", GRADE_SYSTEM),
        ("human", f"Documents:\n{context}\n\nQuestion: {question}"),
    ]
    result = cast(Grade, grader.invoke(messages))
    return {"grade": "yes" if result.relevant else "no"}


def rewrite(state: GraphState) -> GraphState:
    """Reformulate the question for another retrieval pass; count the attempt."""
    llm = make_llm()
    messages = [
        ("system", REWRITE_SYSTEM),
        ("human", f"Original question: {state['question']}\n\nRewritten question:"),
    ]
    rewritten = str(llm.invoke(messages).content)
    return {"question": rewritten, "attempts": state.get("attempts", 0) + 1}


def generate(state: GraphState) -> GraphState:
    """Produce a grounded Answer (claims + verbatim quotes) from chunks."""
    question = state["question"]
    context = _format_chunks(state["chunks"])

    structured = make_structured_llm(Answer)

    messages = [
        ("system", GENERATE_SYSTEM),
        ("human", f"Sources:\n{context}\n\nQuestion: {question}"),
    ]
    answer = structured.invoke(messages)
    return {"answer": cast(Answer, answer)}
