"""LightRAG lifecycle, query calls, and reference discovery for the aggregate path.

Only this module knows LightRAG's query parameters and event-loop constraints.
"""

from __future__ import annotations

import asyncio
from typing import cast

from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
from pydantic import BaseModel, Field

from paranoid_qa.config import LIGHTRAG_DIR, settings
from paranoid_qa.contracts.common import SourceRef
from paranoid_qa.corpus.repository import list_documents

_EMBED_DIM, _EMBED_MAX_TOKENS = 1536, 8191  # text-embedding-3-small

_loop: asyncio.AbstractEventLoop | None = None


def _run(coro):
    """Run a coroutine on a single, reused event loop.

    LightRAG's locks bind to whichever loop first created them, so every LightRAG call in the
    process must share the same loop."""
    global _loop
    if _loop is None or _loop.is_closed():
        _loop = asyncio.new_event_loop()
    return _loop.run_until_complete(coro)


async def _llm(prompt, system_prompt=None, history_messages=None, **kwargs):
    # LightRAG requires its own OpenAI adapter; this path is OpenAI-specific by design.
    return await openai_complete_if_cache(
        settings.gen_model,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages or [],
        **kwargs,
    )


async def _embed(texts):
    return await openai_embed(texts, model=settings.embed_model)


async def _make_rag() -> LightRAG:
    """Construct and initialize a LightRAG over LIGHTRAG_DIR (loads the graph if it exists)."""
    rag = LightRAG(
        working_dir=str(LIGHTRAG_DIR),
        llm_model_func=_llm,
        embedding_func=EmbeddingFunc(_EMBED_DIM, _embed, _EMBED_MAX_TOKENS),
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag


class AggregateRetrievalResult(BaseModel):
    """One aggregate query: the synthesized prose, the retrieved context, and cited references."""

    answer_text: str
    context: str
    references: list[SourceRef] = Field(default_factory=list)


def _references_from_context(context: str) -> list[SourceRef]:
    # TODO: replace substring scanning with typed LightRAG evidence (or Postgres-backed references).
    try:
        docs = list_documents()
    except FileNotFoundError:
        return []
    return [doc for doc in docs if doc.filename in context]


def query(question: str) -> AggregateRetrievalResult:
    """Answer a corpus-level question and return the answer, its context, and references.

    Runs two queries on the same question: one synthesizes the prose answer, one returns the
    retrieved context. They are separate aquery calls but return the same context because
    LightRAG caches its keyword-extraction step (keyed by prompt, cache on by default). So the
    critic's context reflects what the answer was synthesized from."""

    async def _query():
        rag = await _make_rag()
        answer = await rag.aquery(question, param=QueryParam(mode="hybrid"))
        context = await rag.aquery(
            question, param=QueryParam(mode="hybrid", only_need_context=True)
        )
        return answer, context

    raw, context = _run(_query())
    context_text = cast(str, context)
    return AggregateRetrievalResult(
        answer_text=cast(str, raw),
        context=context_text,
        references=_references_from_context(context_text),
    )
