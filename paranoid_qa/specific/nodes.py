"""Specific-path LangGraph nodes: retrieve, grade, rewrite, generate.

Nodes are pure: state in, partial-state dict out. Retrieval rewriting increments
`specific_retrieval_attempts`; answer revision is counted separately in the verifier.
"""

from __future__ import annotations

from typing import cast

from paranoid_qa.contracts.specific import Grade, RetrievedChunk, SpecificAnswer
from paranoid_qa.contracts.state import GraphState
from paranoid_qa.llm.factory import make_chat_model, make_generator
from paranoid_qa.specific.retrieval import retrieve_chunks

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


async def retrieve(state: GraphState) -> GraphState:
    """Fetch candidate chunks for the question."""
    chunks = await retrieve_chunks(state["question"])
    return {"specific_chunks": chunks}


async def grade(state: GraphState) -> GraphState:
    """Judge whether the retrieved docs are good enough to answer."""
    context = _format_chunks(state.get("specific_chunks", []))

    grader = make_generator(Grade)
    messages = [
        ("system", GRADE_SYSTEM),
        ("human", f"Documents:\n{context}\n\nQuestion: {state['question']}"),
    ]
    result = cast(Grade, await grader.ainvoke(messages))
    return {"specific_grade": "yes" if result.relevant else "no"}


async def rewrite(state: GraphState) -> GraphState:
    """Reformulate the question for another retrieval pass; count the retrieval attempt."""
    llm = make_chat_model(role="generator")
    messages = [
        ("system", REWRITE_SYSTEM),
        ("human", f"Original question: {state['question']}\n\nRewritten question:"),
    ]
    rewritten = str((await llm.ainvoke(messages)).content)
    return {
        "question": rewritten,
        "specific_retrieval_attempts": state.get("specific_retrieval_attempts", 0) + 1,
    }


def _feedback(state: GraphState) -> str:
    """Render critic feedback from the last verify pass; '' on the first generate."""
    answer = state.get("answer")
    verdicts = state.get("specific_verdicts") or []
    if not isinstance(answer, SpecificAnswer) or not verdicts:
        return ""
    rejected = [
        f'- claim: "{c.text}"\n  quote: "{c.quote}"\n  rejected ({v.verdict}): {v.explanation}'
        for c, v in zip(answer.claims, verdicts)
        if v.verdict != "supported"
    ]
    if not rejected:
        return ""
    return REVISE_GUIDANCE + "\n\nRejected claims:\n" + "\n".join(rejected)


async def generate(state: GraphState) -> GraphState:
    """Produce a grounded SpecificAnswer (claims + verbatim quotes), applying critic feedback."""
    context = _format_chunks(state.get("specific_chunks", []))

    structured = make_generator(SpecificAnswer)
    human = f"Sources:\n{context}\n\nQuestion: {state['question']}"
    feedback = _feedback(state)
    if feedback:
        human += f"\n\n{feedback}"

    messages = [("system", GENERATE_SYSTEM), ("human", human)]
    answer = await structured.ainvoke(messages)
    return {"answer": cast(SpecificAnswer, answer)}
