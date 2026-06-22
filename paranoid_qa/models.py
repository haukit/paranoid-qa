"""Model factory. Consolidates model/provider-specific syntax and logic.

Switch providers by setting `settings.provider` (+ provider-specific model names in config)
and installing that provider's extra dependencies:
    uv sync --extra openai      # langchain-openai + llama-index-embeddings-openai
    uv sync --extra anthropic   # langchain-anthropic (no first-party embeddings)
    uv sync --extra cohere      # langchain-cohere + llama-index-embeddings-cohere

NOTE: changing the embedder invalidates the persisted index, so you should delete settings.STORAGE_DIR and rebuild.
"""

# Provider extras (openai/anthropic/cohere) are optional and installed on demand, so their
# imports won't resolve in the base environment -> expected here, not an error.
# pyright: reportMissingImports=false

from __future__ import annotations

from typing import Literal

from paranoid_qa.config import settings


def make_llm(
    model: str | None = None, *, provider: str | None = None, temperature: float | None = None
):
    """Return a LangChain chat model for the provider.

    `provider` can be overridden per call so the generator and critic may live on different
    providers, e.g. gen on OpenAI, critic on Anthropic.
    """
    name = model or settings.gen_model
    prov = provider or settings.provider
    temp = settings.temperature if temperature is None else temperature

    if prov == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=name, base_url=settings.ollama_host, temperature=temp, reasoning=False
        )  # qwen3 thinks by default -> constrained decoding doesn't work
    if prov == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=name, temperature=temp)
    if prov == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=name, temperature=temp)
    if prov == "cohere":
        from langchain_cohere import ChatCohere

        return ChatCohere(model=name, temperature=temp)

    raise ValueError(
        f"No chat model configured for provider={prov!r}. "
        "Add a branch in paranoid_qa.models.make_llm and install its extra."
    )


def _ollama_structured_method(model: str) -> Literal["function_calling", "json_schema"]:
    """Set structured-output method based on model"""
    return "json_schema" if "gemma" in model.lower() else "function_calling"


def make_structured_llm(
    schema,
    *,
    model: str | None = None,
    provider: str | None = None,
    temperature: float | None = None,
):
    """A chat model wired to return `schema` as validated structured output.

    Primarily used for local Ollama models where tool/function calling is required
    for them to return reliable structured data."""
    prov = provider or settings.provider
    name = model or settings.gen_model
    llm = make_llm(model=model, provider=prov, temperature=temperature)
    if prov == "ollama":
        return llm.with_structured_output(schema, method=_ollama_structured_method(name))

    return llm.with_structured_output(schema)


def make_li_llm(
    model: str | None = None, *, provider: str | None = None, temperature: float | None = None
):
    """Return a LlamaIndex LLM for LLMRerank"""
    name = model or settings.gen_model
    prov = provider or settings.provider
    temp = settings.temperature if temperature is None else temperature

    if prov == "ollama":
        from llama_index.llms.ollama import Ollama

        return Ollama(
            model=name,
            base_url=settings.ollama_host,
            temperature=temp,
            request_timeout=600.0,
        )

    raise ValueError(
        f"No LlamaIndex LLM configured for provider={prov!r}. "
        "Add a branch in paranoid_qa.models.make_li_llm."
    )


def make_embedder(model: str | None = None, *, provider: str | None = None):
    """Return a LlamaIndex embedding model for the configured provider.

    LlamaIndex has no single init function, so we dispatch on the provider explicitly.
    """
    name = model or settings.embed_model
    prov = provider or settings.provider

    if prov == "ollama":
        from llama_index.embeddings.ollama import OllamaEmbedding

        return OllamaEmbedding(model_name=name, base_url=settings.ollama_host)
    if prov == "openai":
        from llama_index.embeddings.openai import OpenAIEmbedding

        return OpenAIEmbedding(model=name)
    if prov == "cohere":
        from llama_index.embeddings.cohere import CohereEmbedding

        return CohereEmbedding(model_name=name)
    # Anthropic has no first-party embeddings so we pair its chat models with another embedder.

    raise ValueError(
        f"No embedder configured for provider={prov!r}. "
        "Add a branch in paranoid_qa.models.make_embedder and install its extra."
    )
