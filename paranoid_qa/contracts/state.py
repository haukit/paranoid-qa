"""The single LangGraph state passed between every node.

Fields are path-prefixed so a name never changes meaning by route, and the
retry counters are separated so one correction loop cannot silently consume
another's budget.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from paranoid_qa.contracts.aggregate import AggregateAnswer, AggregateClaimVerdict
from paranoid_qa.contracts.common import RouteKind, RunStatus
from paranoid_qa.contracts.specific import RetrievedChunk, SpecificAnswer, SpecificClaimVerdict


# Not-required access is fine: nodes read fields written earlier in the run.
class GraphState(TypedDict, total=False):
    """State passed between LangGraph nodes."""

    question: str
    route: RouteKind

    # Specific path.
    specific_chunks: list[RetrievedChunk]
    specific_grade: Literal["yes", "no"]
    specific_verdicts: list[SpecificClaimVerdict]
    specific_retrieval_attempts: int
    specific_revision_attempts: int

    # Aggregate path.
    aggregate_context: str
    aggregate_verdicts: list[AggregateClaimVerdict]
    aggregate_correction_attempts: int

    # Shared answer + terminal outcome.
    answer: SpecificAnswer | AggregateAnswer | None
    verification_passed: bool
    status: RunStatus
    outcome_reason: str
