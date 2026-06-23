from __future__ import annotations

from typing import Literal, TypedDict

from pydantic import BaseModel, Field


class Grade(BaseModel):
    relevant: bool = Field(
        description="True if the documents contain the information needed to answer this question."
    )


class Route(BaseModel):
    """Router decision: which path answers this question."""

    kind: Literal["specific", "aggregate"] = Field(
        description=(
            "'specific' = about a particular event/document/detail, answerable from a few "
            "retrieved passages; 'aggregate' = corpus-level counts, trends, themes, or "
            "comparisons across many documents."
        )
    )


class Claim(BaseModel):
    """One atomic factual statement plus the evidence backing it.

    Grounding depends on path:
    - specific:     `quote` is a verbatim span from a retrieved chunk.
    - aggregate:    `quote` is None since a corpus-level synthesis usually cannot be quoted
                    directly; the backing for this is in the Answer's `references` instead."""

    text: str = Field(description="A single, atomic factual statement.")
    quote: str | None = Field(
        default=None,
        description="Verbatim span copied from the retrieved documents that supports `text`.",
    )


class Source(BaseModel):
    """A citation, i.e., a (document, page) pair."""

    document: str
    page: str | None = None

    def __str__(self) -> str:
        return f"{self.document} p.{self.page}" if self.page else self.document


class RetrievedChunk(TypedDict):
    """One retrieved piece of text + its provenance"""

    text: str
    document: str  # the source file this chunk came from
    page: str | None  # page number within the document


class Answer(BaseModel):
    """The generator's structured output: an answer decomposed into cited claims,
    plus source refs for aggregate answers."""

    claims: list[Claim] = Field(default_factory=list)
    references: list[Source] = Field(default_factory=list)

    @property
    def text(self) -> str:
        """Render the prose answer from its claims."""
        return " ".join(c.text for c in self.claims)


class ClaimVerdict(BaseModel):
    """The critic's independent judgment of a single (claim, quote) pair."""

    verdict: Literal["supported", "unsupported", "contradicted", "fabricated"]
    explanation: str = Field(description="Why the cited source does or does not support the claim.")
    source: Source | None = None  # derived by the verifier when the quote is located


# Specify as TypedDict, instead of Pydantic model, for LangGraph dict-merge;
# also not LLM-facing so no validation needed.
class GraphState(TypedDict, total=False):
    """State passed between LangGraph nodes."""

    question: str
    route: str  # "specific" | "aggregate" (set by the router)
    chunks: list[RetrievedChunk]
    grade: str  # "yes" | "no" from the relevance grader
    answer: Answer
    verdicts: list[ClaimVerdict]
    faithful: bool
    attempts: int
    context: str
