"""In-process token/cost accumulation from OpenInference LLM spans."""

from __future__ import annotations

import threading
from collections import defaultdict

from openinference.semconv.trace import SpanAttributes
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor

from paranoid_qa.config import token_cost


class Usage:
    """Running token and cost totals for one trace."""

    # Fixed record created once per in-flight trace; __slots__ drops the
    # per-instance __dict__ to keep it lean.
    __slots__ = ("tokens_in", "tokens_out", "cost")

    def __init__(self) -> None:
        self.tokens_in = 0
        self.tokens_out = 0
        self.cost = 0.0


class TokenCostProcessor(SpanProcessor):
    """Sums LLM token counts and cost per trace; read out when the request finishes."""

    def __init__(self) -> None:
        super().__init__()

        # trace id -> that request's running token/cost totals
        self._usage_by_trace: dict[int, Usage] = defaultdict(Usage)

        # on_end can fire on several threads at once (concurrent requests end spans in parallel),
        # so guard the read-modify-write on the shared map and its Usage records.
        self._lock = threading.Lock()

    def on_end(self, span: ReadableSpan) -> None:
        """Add an LLM span's token counts and cost to its trace's running total."""
        attrs = span.attributes or {}

        # Only chat/LLM spans carry these token counts. Embedding spans use a different
        # convention and we skip over them now; query-embedding cost should be negligible anyway.
        raw_in = attrs.get(SpanAttributes.LLM_TOKEN_COUNT_PROMPT)
        raw_out = attrs.get(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION)

        if raw_in is None and raw_out is None:
            return  # not an LLM span
        if span.context is None:
            return

        tin = int(raw_in) if isinstance(raw_in, (int, float)) else 0
        tout = int(raw_out) if isinstance(raw_out, (int, float)) else 0
        model = str(attrs.get(SpanAttributes.LLM_MODEL_NAME, ""))
        with self._lock:
            usage = self._usage_by_trace[span.context.trace_id]
            usage.tokens_in += tin
            usage.tokens_out += tout
            usage.cost += token_cost(model, tin, tout)

    def pop(self, trace_id: int) -> Usage:
        """Remove and return the accumulated usage for a finished trace (empty if unseen)."""
        with self._lock:
            return self._usage_by_trace.pop(trace_id, Usage())
