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
- For each claim, set `quote` to a SINGLE CONTINUOUS span copied VERBATIM
  (character-for-character) from ONE place in the sources. You must be able to find it
  with ctrl-F. Never paraphrase, never reword connectors, and never stitch together
  text from different sentences or passages.
- If the sources do not answer the question, return an empty list of claims."""

GRADE_SYSTEM = """You assess whether retrieved documents are relevant to a user's question.
Set relevant=true ONLY if the documents contain information needed to answer the question.
Otherwise set relevant=false."""

REWRITE_SYSTEM = """The retrieved documents did not answer the user's question well.
Rewrite the question to be clearer and more retrievable: expand abbreviations, add likely
synonyms/keywords, and make the information need explicit. Return ONLY the rewritten question."""

REVISE_GUIDANCE = """Your previous answer was fact-checked and some claims were rejected.
Correct them: give each rejected claim a span copied VERBATIM from the sources that genuinely
supports it, or drop the claim if the sources do not support it. Return the full corrected answer."""


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


def _feedback(state: GraphState) -> str:
    """Render critic feedback from the last verify pass; '' on the first generate."""
    answer = state.get("answer")
    verdicts = state.get("verdicts") or []
    if not answer or not verdicts:
        return ""
    rejected = [
        f'- claim: "{c.text}"\n  quote: "{c.quote}"\n  rejected ({v.verdict}): {v.explanation}'
        for c, v in zip(answer.claims, verdicts)
        if v.verdict != "supported"
    ]
    if not rejected:
        return ""
    return REVISE_GUIDANCE + "\n\nRejected claims:\n" + "\n".join(rejected)


def generate(state: GraphState) -> GraphState:
    """Produce a grounded Answer (claims + verbatim quotes) from chunks,
    incorporating critic feedback on a revise pass."""
    question = state["question"]
    context = _format_chunks(state["chunks"])

    structured = make_structured_llm(Answer)

    human = f"Sources:\n{context}\n\nQuestion: {question}"
    feedback = _feedback(state)
    if feedback:
        human += f"\n\n{feedback}"

    messages = [
        ("system", GENERATE_SYSTEM),
        ("human", human),
    ]
    answer = structured.invoke(messages)
    return {"answer": cast(Answer, answer)}
