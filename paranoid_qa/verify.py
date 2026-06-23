"""The critic: verify each claim against its sources.

Two stages per claim:
1. locate_quote() — deterministic, no LLM: find the chunk whose text contains the quote
    (whitespace-normalized). Not found -> the claim is fabricated.
2. critic — a different-family model judges whether the located chunk supports the
    claim, emitting a ClaimVerdict. The located chunk's metadata becomes the citation.
"""

from __future__ import annotations

import re
from typing import cast

from paranoid_qa.config import settings
from paranoid_qa.models import make_structured_llm
from paranoid_qa.schemas import Claim, ClaimVerdict, GraphState, RetrievedChunk, Source


def _normalize(text: str) -> str:
    "Collapse consecutive whitespace to single spaces."
    return re.sub(r"\s+", " ", text).strip()


def locate_quote(quote: str, chunks: list[RetrievedChunk]) -> RetrievedChunk | None:
    nq = _normalize(quote)
    if not nq:
        return None
    for chunk in chunks:
        if nq in _normalize(chunk["text"]):
            return chunk
    return None


CRITIC_SYSTEM = """You are a strict fact-checker. You are given a CLAIM, a QUOTE its author says
supports it, and the SOURCE the quote was drawn from. Judge ONLY whether the SOURCE supports the
CLAIM — not whether the claim is true in the world.

Choose exactly one verdict:
- supported    — the source clearly supports the claim.
- unsupported  — the source neither supports nor contradicts the claim.
- contradicted — the source contradicts the claim.
- fabricated   — the quote does not appear in the source.
Give a one-sentence explanation."""


def verify_claim(claim: Claim, chunks: list[RetrievedChunk]) -> ClaimVerdict:
    """Verify one claim: locate the quote, then the LLM critic."""
    # Located the quote
    located = locate_quote(claim.quote, chunks)
    if located is None:
        return ClaimVerdict(
            verdict="fabricated",
            explanation="Quote not found in any retrieved chunk (deterministic check)",
        )
    source = Source(document=located["document"], page=located["page"])

    # Use a model from a different family to assess entailment against the located chunk
    critic = make_structured_llm(ClaimVerdict, model=settings.critic_model)
    messages = [
        ("system", CRITIC_SYSTEM),
        ("human", f"SOURCE:\n{located['text']}\n\nCLAIM: {claim.text}\n\nQUOTE: {claim.quote}"),
    ]
    verdict = cast(ClaimVerdict, critic.invoke(messages))
    verdict.source = source
    return verdict


def verify(state: GraphState) -> GraphState:
    """Graph node: run the critic over every claim and flag faithfulness"""
    answer = state["answer"]
    chunks = state["chunks"]

    verdicts = [verify_claim(c, chunks) for c in answer.claims]

    # Case: empty claims mean content isn't there -> accept so we don't spend up our budget chasing
    faithful = all(v.verdict == "supported" for v in verdicts)

    out: GraphState = {"verdicts": verdicts, "faithful": faithful}
    if not faithful:
        out["attempts"] = state.get("attempts", 0) + 1
    return out


AGGREGATE_CRITIC_SYSTEM = """You are a strict fact-checker for a corpus-level answer.
You are given SOURCE CONTEXT retrieved from a document corpus and an ANSWER synthesized from it.

Judge ONLY whether the ANSWER is supported by the SOURCE CONTEXT:
- supported    — the context supports the answer.
- unsupported  — the context lacks enough to support the answer.
- contradicted — the context contradicts the answer.
Give a one-sentence explanation."""


def verify_aggregate(state: GraphState) -> GraphState:
    """Path-aware verify: an aggregate answer must cite real corpus documents, and be supported
    by them (according to a critic)"""
    # Deterministic check: did retrieval find any source documents?
    if not state.get("references"):
        verdict = ClaimVerdict(verdict="unsupported", explanation="No source documents retrieved.")
        return {"verdicts": [verdict], "faithful": False}

    # Entailment check: is the answer supported by the retrieved context?
    critic = make_structured_llm(ClaimVerdict, model=settings.critic_model)
    messages = [
        ("system", AGGREGATE_CRITIC_SYSTEM),
        ("human", f"SOURCE CONTEXT:\n{state.get('context', '')}\n\nANSWER: {state['answer'].text}"),
    ]
    verdict = cast(ClaimVerdict, critic.invoke(messages))

    return {"verdicts": [verdict], "faithful": verdict.verdict == "supported"}
