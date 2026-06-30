"""OpenTelemetry tracing into Phoenix."""

from __future__ import annotations

from openinference.instrumentation.langchain import LangChainInstrumentor
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from phoenix.otel import register


def setup_tracing(project_name: str = "paranoid-qa") -> None:
    tracer_provider = register(project_name=project_name)
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
