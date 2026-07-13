"""Specific-path verification: locate the quote, gate on relevance, then judge support.

For each claim:
1. locate_quote() — deterministic, no LLM: find the chunk containing the quote. Missing -> fabricated.
2. relevance gate — a critic-family model checks the quote's source document is about the
   question's subject (identity read via corpus.repository). Off-subject -> irrelevant.
3. support critic — a different-family model judges whether the located chunk supports the claim.

Depends on contracts.specific, corpus.repository, llm.factory, config. Never on the aggregate path.
"""

from __future__ import annotations

import re
import unicodedata
from typing import cast

from rapidfuzz import fuzz

from paranoid_qa.contracts.common import SourceRef, document_id_from_filename
from paranoid_qa.contracts.specific import (
    RelevanceVerdict,
    RetrievedChunk,
    SpecificAnswer,
    SpecificClaim,
    SpecificClaimVerdict,
)
from paranoid_qa.contracts.state import GraphState
from paranoid_qa.corpus.repository import get_document_text
from paranoid_qa.llm.factory import make_critic

_QUOTE_MATCH_THRESHOLD = 90  # rapidfuzz partial_ratio
_ELLIPSIS = re.compile(r"\s*(?:\.\.\.|…)\s*")
_IDENTITY_CHARS = 800  # opening slice of a source doc, enough to clear a cover page


def _normalize(text: str) -> str:
    "Lowercase, NFKC-normalise unicode, and collapse whitespace."
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", " ", text).strip()


def locate_quote(quote: str, chunks: list[RetrievedChunk]) -> RetrievedChunk | None:
    """Return the chunk that contains `quote` (fuzzy matching).

    The quote is split on ellipses into segments; each normalised segment must fuzzy-match a
    span of the chunk (partial_ratio >= threshold)."""
    segments = [s for s in (_normalize(seg) for seg in _ELLIPSIS.split(quote)) if s]
    if not segments:
        return None
    for chunk in chunks:
        chunk_test = _normalize(chunk["text"])
        if all(fuzz.partial_ratio(seg, chunk_test) >= _QUOTE_MATCH_THRESHOLD for seg in segments):
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

    Sets relevant=true when the document identity can't be read (fails open)."""
    try:
        identity = (get_document_text(document) or "")[:_IDENTITY_CHARS]
    except FileNotFoundError:
        identity = ""

    if not identity:
        return RelevanceVerdict(
            relevant=True, explanation="No document identity available; not gating."
        )

    judge = make_critic(RelevanceVerdict)
    messages = [
        ("system", RELEVANCE_SYSTEM),
        ("human", f"QUESTION: {question}\n\nSOURCE (opening of the source document):\n{identity}"),
    ]
    return cast(RelevanceVerdict, await judge.ainvoke(messages))


async def verify_specific_claim(
    claim: SpecificClaim, chunks: list[RetrievedChunk], question: str
) -> SpecificClaimVerdict:
    """Verify one quote-backed claim: locate the quote, gate on relevance, then judge support."""
    located = locate_quote(claim.quote, chunks)
    if located is None:
        return SpecificClaimVerdict(
            verdict="fabricated",
            explanation="Quote not found in any retrieved chunk (deterministic check)",
        )

    source = SourceRef(
        document_id=document_id_from_filename(located["document"]),
        filename=located["document"],
        page=located["page"],
    )

    # Relevance gate: is this source about the question's subject?
    relevance = await _is_relevant(question, located["document"])
    if not relevance.relevant:
        return SpecificClaimVerdict(
            verdict="irrelevant", explanation=relevance.explanation, source=source
        )

    # Different-family critic assesses entailment against the located chunk.
    critic = make_critic(SpecificClaimVerdict)
    messages = [
        ("system", CRITIC_SYSTEM),
        ("human", f"SOURCE:\n{located['text']}\n\nCLAIM: {claim.text}\n\nQUOTE: {claim.quote}"),
    ]
    verdict = cast(SpecificClaimVerdict, await critic.ainvoke(messages))
    verdict.source = source
    return verdict


async def specific_verify(state: GraphState) -> GraphState:
    """Graph node: verify every specific claim; count a revision when verification fails."""
    answer = cast(SpecificAnswer, state["answer"])
    question = state["question"]
    verdicts = [
        await verify_specific_claim(c, state.get("specific_chunks", []), question)
        for c in answer.claims
    ]
    passed = bool(verdicts) and all(v.verdict == "supported" for v in verdicts)

    out: GraphState = {"specific_verdicts": verdicts, "verification_passed": passed}
    if not passed:
        out["specific_revision_attempts"] = state.get("specific_revision_attempts", 0) + 1
    return out
