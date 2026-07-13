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

from paranoid_qa.aggregate import document_text
from paranoid_qa.config import settings
from paranoid_qa.models import make_structured_llm
from paranoid_qa.schemas import (
    Claim,
    ClaimVerdict,
    GraphState,
    RelevanceVerdict,
    RetrievedChunk,
    Source,
)

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

# opening slice of a source doc, enough to clear a cover page
_IDENTITY_CHARS = 800


RELEVANCE_SYSTEM = """You check whether a piece of evidence is even about the subject a QUESTION asks about.
This runs BEFORE any check of whether the evidence supports a claim.

You are given the QUESTION and a SOURCE: the opening text of the document the evidence was drawn from,
which identifies what that document is about.

Set relevant=false ONLY if the source is about a DIFFERENT subject than the question asks about: for
example, the question asks about a specific named subject (a particular entity, event, case, person, or
item) and the source document is about a different one, or the question's subject does not appear in the
source at all.

Otherwise set relevant=true. If the source plausibly concerns the question's subject, or you are unsure,
set relevant=true. Give a one-sentence explanation."""


async def _is_relevant(question: str, document: str) -> RelevanceVerdict:
    """Judge whether `document` is about the question's subject, from its opening text.

    Sets relevant=true when the document identity can't be read."""
    try:
        identity = (document_text(document) or "")[:_IDENTITY_CHARS]
    except FileNotFoundError:
        identity = ""

    if not identity:
        return RelevanceVerdict(
            relevant=True, explanation="No document identity available; not gating."
        )

    judge = make_structured_llm(RelevanceVerdict, model=settings.critic_model)
    messages = [
        ("system", RELEVANCE_SYSTEM),
        ("human", f"QUESTION: {question}\n\nSOURCE (opening of the source document):\n{identity}"),
    ]
    return cast(RelevanceVerdict, await judge.ainvoke(messages))


async def verify_claim(
    claim: Claim,
    chunks: list[RetrievedChunk],
    question: str,
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

        # Relevance gate: is this source about the question's subject?
        relevance = await _is_relevant(question, located["document"])
        if not relevance.relevant:
            return ClaimVerdict(
                verdict="irrelevant",
                explanation=relevance.explanation,
                source=source,
            )

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
    question = state["question"]
    verdicts = [
        await verify_claim(
            c, state.get("chunks", []), question, state.get("context", ""), answer.references
        )
        for c in answer.claims
    ]
    faithful = bool(verdicts) and all(v.verdict == "supported" for v in verdicts)

    out: GraphState = {"verdicts": verdicts, "faithful": faithful}
    if not faithful:
        out["attempts"] = state.get("attempts", 0) + 1
    return out


def accept(state: GraphState) -> GraphState:
    """Terminal node: verification passed, the answer is grounded."""
    return {"status": "answered"}


def abstain(state: GraphState) -> GraphState:
    """Terminal node: verification could not ground the answer within the retry budget."""
    return {"status": "abstained"}


AGGREGATE_CRITIC_SYSTEM = """You are a strict fact-checker for a corpus-level answer.
You are given SOURCE CONTEXT retrieved from a document corpus and a CLAIM drawn from an answer
synthesized over it.

Judge ONLY whether the CLAIM is supported by the SOURCE CONTEXT:
- supported    — the context supports the claim.
- unsupported  — the context lacks enough to support the claim.
- contradicted — the context contradicts the claim.
Give a one-sentence explanation."""
