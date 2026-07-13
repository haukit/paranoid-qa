"""Aggregate-path contracts: reference-grounded claims, answers, and verdicts.

An aggregate claim is a corpus-level synthesis that usually cannot be quoted from
any single chunk, so it carries no `quote` field. Its grounding lives in the
answer's `references`.

The aggregate verdict is intentionally a separate type from the specific verdict,
even where the initial literals overlap, so future aggregate-only concepts
(coverage, invalid reference, aggregation error) do not have to borrow the
specific quote-verification vocabulary.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from paranoid_qa.contracts.common import SourceRef


class AggregateClaim(BaseModel):
    """One atomic factual statement drawn from a corpus-level synthesis (no verbatim quote)."""

    text: str = Field(description="A single, atomic factual statement.")


class AggregateAnswer(BaseModel):
    """The aggregate path's structured output: synthesized claims plus source references."""

    kind: Literal["aggregate"] = "aggregate"
    claims: list[AggregateClaim] = Field(default_factory=list)
    references: list[SourceRef] = Field(default_factory=list)

    @property
    def text(self) -> str:
        """Render the prose answer from its claims."""
        return " ".join(claim.text for claim in self.claims)


class AggregateClaimVerdict(BaseModel):
    """The critic's judgment of a single aggregate claim against the retrieved context."""

    verdict: Literal["supported", "unsupported", "contradicted"]
    explanation: str = Field(description="Why the context does or does not support the claim.")
