"""Model factory. Consolidates model/provider-specific construction behind role-aware helpers.

Switch providers by setting `settings.provider` (+ provider-specific model names in config)
and installing that provider's extra dependencies:
    uv sync --extra openai      # langchain-openai + llama-index-embeddings-openai
    uv sync --extra anthropic   # langchain-anthropic (no first-party embeddings)
    uv sync --extra cohere      # langchain-cohere + llama-index-embeddings-cohere

NOTE: changing the embedder invalidates the persisted index, so delete the specific index and
rebuild.

Prefer the role-aware helpers (`make_generator`, `make_critic`, `make_router`, `make_chat_model`)
so that the generator/critic split is expressed once, here, rather than by threading model names
through every call site.
"""

# Provider extras (openai/anthropic/cohere) are optional and installed on demand, so their
# imports won't resolve in the base environment -> expected here, not an error.
# pyright: reportMissingImports=false

from __future__ import annotations

from typing import Literal

from paranoid_qa.config import settings

ModelRole = Literal["generator", "critic", "router"]


def _model_for_role(role: ModelRole) -> str:
    """The configured model name for a role (the router shares the generator model)."""
    if role == "critic":
        return settings.critic_model
    return settings.gen_model


def _chat(model: str, *, provider: str | None = None, temperature: float | None = None):
    """Low-level LangChain chat-model construction for a given provider."""
    prov = provider or settings.provider
    temp = settings.temperature if temperature is None else temperature

    if prov == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model, base_url=settings.ollama_host, temperature=temp, reasoning=False
        )  # qwen3 thinks by default -> constrained decoding doesn't work
    if prov == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=temp)
    if prov == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, temperature=temp)
    if prov == "cohere":
        from langchain_cohere import ChatCohere

        return ChatCohere(model=model, temperature=temp)

    raise ValueError(
        f"No chat model configured for provider={prov!r}. "
        "Add a branch in paranoid_qa.llm.factory._chat and install its extra."
    )


def _ollama_structured_method(model: str) -> Literal["function_calling", "json_schema"]:
    """Pick the structured-output method Ollama needs for this model."""
    return "json_schema" if "gemma" in model.lower() else "function_calling"


def make_chat_model(
    *, role: ModelRole = "generator", provider: str | None = None, temperature: float | None = None
):
    """Return a plain LangChain chat model for a role."""
    return _chat(_model_for_role(role), provider=provider, temperature=temperature)


def make_structured(
    schema,
    *,
    model: str,
    provider: str | None = None,
    temperature: float | None = None,
):
    """Return a chat model on an explicit `model`, wired to emit `schema` as structured output.

    Low-level escape hatch for callers that need a specific model (e.g. eval gold generation on a
    stronger model). Application code should prefer the role helpers below.
    """
    prov = provider or settings.provider
    llm = _chat(model, provider=prov, temperature=temperature)
    if prov == "ollama":
        return llm.with_structured_output(schema, method=_ollama_structured_method(model))
    return llm.with_structured_output(schema)


def make_structured_model(
    schema,
    *,
    role: ModelRole,
    provider: str | None = None,
    temperature: float | None = None,
):
    """Return a chat model for a role, wired to emit `schema` as validated structured output."""
    return make_structured(
        schema, model=_model_for_role(role), provider=provider, temperature=temperature
    )


def make_generator(schema):
    """Structured generator model (the answer/decomposition/grade role)."""
    return make_structured_model(schema, role="generator")


def make_critic(schema):
    """Structured critic model (a different family from the generator, enforced by llm.policy)."""
    return make_structured_model(schema, role="critic")


def make_router(schema):
    """Structured router model (shares the generator model)."""
    return make_structured_model(schema, role="router")


def make_llamaindex_llm(model: str | None = None, *, provider: str | None = None):
    """Return a LlamaIndex LLM (used by the specific-path reranker)."""
    name = model or settings.gen_model
    prov = provider or settings.provider
    temp = settings.temperature

    if prov == "ollama":
        from llama_index.llms.ollama import Ollama

        return Ollama(model=name, base_url=settings.ollama_host, temperature=temp, request_timeout=600.0)
    if prov == "openai":
        from llama_index.llms.openai import OpenAI

        return OpenAI(model=name, temperature=temp)

    raise ValueError(
        f"No LlamaIndex LLM configured for provider={prov!r}. "
        "Add a branch in paranoid_qa.llm.factory.make_llamaindex_llm."
    )


def make_embedder(model: str | None = None, *, provider: str | None = None):
    """Return a LlamaIndex embedding model for the configured provider.

    LlamaIndex has no single init function, so we dispatch on the provider explicitly.
    """
    name = model or settings.embed_model
    prov = provider or settings.embed_provider

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
        "Add a branch in paranoid_qa.llm.factory.make_embedder and install its extra."
    )
