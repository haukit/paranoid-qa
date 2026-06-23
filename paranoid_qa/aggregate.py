"""The aggregate path: corpus-level QA via a LightRAG knowledge graph."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import cast

from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
from llama_index.core import SimpleDirectoryReader

from paranoid_qa.config import CORPUS_DIR, PROJECT_ROOT, settings
from paranoid_qa.schemas import Answer, Claim, GraphState, Source

LIGHTRAG_DIR = PROJECT_ROOT / ".lightrag"  # persisted graph
_EMBED_DIM, _EMBED_MAX_TOKENS = 1536, 8191  # text-embedding-3-small


_loop: asyncio.AbstractEventLoop | None = None


def _run(coro):
    """Run a coroutine on a single, reused event loop.
    LightRAG's locks bind to whichever loop first created them, so every LightRAG
    call in the process must share the same loop."""
    global _loop
    if _loop is None or _loop.is_closed():
        _loop = asyncio.new_event_loop()
    return _loop.run_until_complete(coro)


async def _llm(prompt, system_prompt=None, history_messages=None, **kwargs):
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


def build_lightrag() -> None:
    """Extract entities/relationships from the corpus; run once."""
    LIGHTRAG_DIR.mkdir(parents=True, exist_ok=True)
    docs = SimpleDirectoryReader(str(CORPUS_DIR)).load_data()  # PDFs -> page texts

    # Regroup pages per file so that we can add file names as references during the index build below
    by_file: dict[str, list[str]] = defaultdict(list)
    for d in docs:
        by_file[d.metadata.get("file_name", "?")].append(d.text)
    names = list(by_file)
    texts = ["\n".join(by_file[n]) for n in names]  # one big string per doc

    async def _build():
        rag = await _make_rag()
        await rag.ainsert(texts, file_paths=names)  # file names as references

    _run(_build())


def _references_from_context(context: str) -> list[Source]:
    cited = sorted(f.name for f in CORPUS_DIR.glob("*") if f.is_file() and f.name in context)
    return [Source(document=name) for name in cited]


def aggregate_answer(state: GraphState) -> GraphState:
    """Answer a corpus-level question; derive references + context from the graph for verify.

    Runs two queries on the same question: one synthesizes the prose answer, one returns the
    retrieved context. They are separate aquery calls but return the same context because
    LightRAG caches its keyword-extraction step (keyed by prompt, cache on by default). So
    the critic's context reflects what the answer was synthesized from."""

    async def _query():
        rag = await _make_rag()
        answer = await rag.aquery(state["question"], param=QueryParam(mode="hybrid"))
        context = await rag.aquery(
            state["question"], param=QueryParam(mode="hybrid", only_need_context=True)
        )
        return answer, context

    raw, context = _run(_query())
    answer_text, context = cast(str, raw), cast(str, context)

    return {
        "answer": Answer(claims=[Claim(text=answer_text, quote="")]),
        "references": _references_from_context(context),
        "context": context,
    }
