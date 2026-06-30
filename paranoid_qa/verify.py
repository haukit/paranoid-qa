"""The critic: verify each claim against its sources.

Two stages per claim:
1. locate_quote() — deterministic, no LLM: find the chunk whose text contains the quote
    (whitespace-normalized). Not found -> the claim is fabricated.
2. critic — a different-family model judges whether the located chunk supports the
    claim, emitting a ClaimVerdict. The located chunk's metadata becomes the citation.
"""

from __future__ import annotations

import re
import unicodedata
from typing import cast

from rapidfuzz import fuzz

from paranoid_qa.config import settings
from paranoid_qa.models import make_structured_llm
from paranoid_qa.schemas import Claim, ClaimVerdict, GraphState, RetrievedChunk, Source

_QUOTE_MATCH_THRESHOLD = 90  # rapidfuzz partial_ratio
_ELLIPSIS = re.compile(r"\s*(?:\.\.\.|…)\s*")


def _normalize(text: str) -> str:
    "Lowercase, NFKC-normalise unicode, and collapse whitespace."
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", " ", text).strip()


def locate_quote(quote: str, chunks: list[RetrievedChunk]) -> RetrievedChunk | None:
    """Return the chunk that contains `quote` (fuzzy matching).

    The quote is split on ellipses into segments, each segment is normalised
    (case / unicode / whitespace), and each must fuzzy-match a span of the chunk
    (partial_ratio >= threshold)."""

    segments = [s for s in (_normalize(seg) for seg in _ELLIPSIS.split(quote)) if s]
    if not segments:
        return None
    for chunk in chunks:
        chunk_test = _normalize(chunk["text"])
        all_segments_present = all(
            fuzz.partial_ratio(seg, chunk_test) >= _QUOTE_MATCH_THRESHOLD for seg in segments
        )
        if all_segments_present:
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


async def verify_claim(
    claim: Claim,
    chunks: list[RetrievedChunk],
    context: str = "",
    references: list[Source] | None = None,
) -> ClaimVerdict:
    """Verify one claim against its evidence.

    - `quote` present -> specific: the quote must locate verbatim in a retrieved chunk,
      then the critic judges entailment against that chunk; the chunk's metadata becomes the citation.
    - `quote` absent -> aggregate: the claim is judged against the retrieved content, gated on
      retrieval having found any source documents (references)."""

    if claim.quote is not None:
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
        verdict = cast(ClaimVerdict, await critic.ainvoke(messages))
        verdict.source = source
        return verdict

    # aggregate
    if not references:
        return ClaimVerdict(verdict="unsupported", explanation="No source documents retrieved.")
    critic = make_structured_llm(ClaimVerdict, model=settings.critic_model)
    messages = [
        ("system", AGGREGATE_CRITIC_SYSTEM),
        ("human", f"SOURCE CONTENT:\n{context}\n\nCLAIM: {claim.text}"),
    ]
    return cast(ClaimVerdict, await critic.ainvoke(messages))


async def verify(state: GraphState) -> GraphState:
    """Graph node: run the critic over every claim and flag faithfulness"""
    answer = state["answer"]
    verdicts = [
        await verify_claim(c, state.get("chunks", []), state.get("context", ""), answer.references)
        for c in answer.claims
    ]
    faithful = all(v.verdict == "supported" for v in verdicts)

    out: GraphState = {"verdicts": verdicts, "faithful": faithful}
    if not faithful:
        out["attempts"] = state.get("attempts", 0) + 1
    return out


AGGREGATE_CRITIC_SYSTEM = """You are a strict fact-checker for a corpus-level answer.
You are given SOURCE CONTEXT retrieved from a document corpus and a CLAIM drawn from an answer
synthesized over it.

Judge ONLY whether the CLAIM is supported by the SOURCE CONTEXT:
- supported    — the context supports the claim.
- unsupported  — the context lacks enough to support the claim.
- contradicted — the context contradicts the claim.
Give a one-sentence explanation."""
