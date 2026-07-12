"""OpenTelemetry tracing: always-on spans, with optional Phoenix export."""

from __future__ import annotations

import os

from openinference.instrumentation.langchain import LangChainInstrumentor
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider


def setup_tracing(project_name: str = "paranoid-qa") -> TracerProvider:
    """Set up a TracerProvider to let spans always exist in-process.

    Phoenix export is added only when PHOENIX_COLLECTOR_ENDPOINT is set."""
    if os.getenv("PHOENIX_COLLECTOR_ENDPOINT"):
        from phoenix.otel import register

        tracer_provider = register(project_name=project_name)
    else:
        tracer_provider = TracerProvider()
        trace.set_tracer_provider(tracer_provider)

    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
    return tracer_provider
