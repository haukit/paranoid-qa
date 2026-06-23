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

from paranoid_qa.config import CORPUS_DIR, settings
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


def verify_aggregate(state: GraphState) -> GraphState:
    """Path-aware verify: an aggregate answer must cite real corpus documents"""
    references = state.get("references", [])
    corpus_files = {p.name for p in CORPUS_DIR.glob("*")}
    faithful = bool(references) and all(s.document in corpus_files for s in references)
    return {"faithful": faithful}
