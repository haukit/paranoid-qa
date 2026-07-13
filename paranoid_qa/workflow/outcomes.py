"""Terminal outcome nodes shared by both paths.

These are workflow outcomes, not verification primitives, so they live here rather than in a
path's verifier. `accept` records a grounded answer; `abstain` records a refusal and clears the
answer so an unsupported draft is never exposed.
"""

from __future__ import annotations

from paranoid_qa.contracts.state import GraphState

_ABSTAIN_REASON = "The answer could not be verified against the source documents."


def accept(state: GraphState) -> GraphState:
    """Terminal: verification passed, the answer is grounded."""
    return {"status": "answered"}


def abstain(state: GraphState) -> GraphState:
    """Terminal: the answer could not be grounded; refuse rather than emit an unsupported claim."""
    return {"status": "abstained", "answer": None, "outcome_reason": _ABSTAIN_REASON}
