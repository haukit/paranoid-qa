"""Aggregate-path verification: judge synthesized claims against the retrieved context.

Structurally independent from specific quote verification: no quote-locator is imported.
Future work (do not implement yet): reference-based entailment, coverage checks, an
aggregate-specific critic prompt, and a corrective loop.
"""

from __future__ import annotations

from typing import cast

from paranoid_qa.contracts.aggregate import AggregateAnswer, AggregateClaim, AggregateClaimVerdict
from paranoid_qa.contracts.common import SourceRef
from paranoid_qa.contracts.state import GraphState
from paranoid_qa.llm.factory import make_critic

AGGREGATE_CRITIC_SYSTEM = """You are a strict fact-checker for a corpus-level answer.
You are given SOURCE CONTEXT retrieved from a document corpus and a CLAIM drawn from an answer
synthesized over it.

Judge ONLY whether the CLAIM is supported by the SOURCE CONTEXT:
- supported    — the context supports the claim.
- unsupported  — the context lacks enough to support the claim.
- contradicted — the context contradicts the claim.
Give a one-sentence explanation."""


async def verify_aggregate_claim(
    claim: AggregateClaim, context: str, references: list[SourceRef]
) -> AggregateClaimVerdict:
    """Judge one aggregate claim against the retrieved context, gated on references existing."""
    if not references:
        return AggregateClaimVerdict(
            verdict="unsupported", explanation="No source documents retrieved."
        )
    critic = make_critic(AggregateClaimVerdict)
    messages = [
        ("system", AGGREGATE_CRITIC_SYSTEM),
        ("human", f"SOURCE CONTENT:\n{context}\n\nCLAIM: {claim.text}"),
    ]
    return cast(AggregateClaimVerdict, await critic.ainvoke(messages))


async def aggregate_verify(state: GraphState) -> GraphState:
    """Graph node: verify every aggregate claim against the context and its references."""
    answer = cast(AggregateAnswer, state["answer"])
    context = state.get("aggregate_context", "")
    verdicts = [
        await verify_aggregate_claim(c, context, answer.references) for c in answer.claims
    ]
    passed = bool(verdicts) and all(v.verdict == "supported" for v in verdicts)
    return {"aggregate_verdicts": verdicts, "verification_passed": passed}
