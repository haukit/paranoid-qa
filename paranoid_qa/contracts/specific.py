"""Specific-path contracts: quote-grounded claims, answers, and verdicts.

A specific claim is always backed by a verbatim quote, so `SpecificClaim.quote`
is required. This makes "a specific claim without a quote" unrepresentable, and
removes the runtime dispatch that used a missing quote to select an algorithm.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from pydantic import BaseModel, Field

from paranoid_qa.contracts.common import SourceRef


class Grade(BaseModel):
    relevant: bool = Field(
        description="True if the documents contain the information needed to answer this question."
    )


class RetrievedChunk(TypedDict):
    """One retrieved piece of text plus its provenance."""

    text: str
    document: str  # the source filename this chunk came from
    page: str | None  # page label within the document


class SpecificClaim(BaseModel):
    """One atomic factual statement backed by a verbatim quote from a retrieved chunk."""

    text: str = Field(description="A single, atomic factual statement.")
    quote: str = Field(
        description="Verbatim span copied from the retrieved documents that supports `text`."
    )


class SpecificAnswer(BaseModel):
    """The specific generator's structured output: an answer decomposed into quoted claims."""

    kind: Literal["specific"] = "specific"
    claims: list[SpecificClaim] = Field(default_factory=list)

    @property
    def text(self) -> str:
        """Render the prose answer from its claims."""
        return " ".join(claim.text for claim in self.claims)


class SpecificClaimVerdict(BaseModel):
    """The critic's independent judgment of a single specific (claim, quote) pair."""

    verdict: Literal["supported", "unsupported", "contradicted", "fabricated", "irrelevant"]
    explanation: str = Field(description="Why the cited source does or does not support the claim.")
    source: SourceRef | None = None  # derived by the verifier when the quote is located


class RelevanceVerdict(BaseModel):
    """Whether a quote's source document is about the question's subject (checked before support)."""

    relevant: bool = Field(
        description="True if the source document concerns the subject the question asks about."
    )
    explanation: str = Field(description="Why the source is or isn't about the question's subject.")
