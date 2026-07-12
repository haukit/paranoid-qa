"""The aggregate path: corpus-level QA via a LightRAG knowledge graph."""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from functools import lru_cache
from typing import cast

from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
from llama_index.core import SimpleDirectoryReader

from paranoid_qa.config import CORPUS_DIR, LIGHTRAG_DIR, settings
from paranoid_qa.models import make_structured_llm
from paranoid_qa.schemas import Answer, Claim, GraphState, Source

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


@lru_cache(maxsize=1)
def corpus_filenames() -> tuple[str, ...]:
    """Filenames the LightRAG graph was built from."""
    status = json.loads((LIGHTRAG_DIR / "kv_store_doc_status.json").read_text())
    return tuple(sorted({v["file_path"] for v in status.values() if v.get("file_path")}))


@lru_cache(maxsize=1)
def _docs_by_filename() -> dict[str, str]:
    """Map each corpus filename to its extracted text."""
    docs = json.loads((LIGHTRAG_DIR / "kv_store_full_docs.json").read_text())
    return {
        v["file_path"]: v["content"]
        for v in docs.values()
        if isinstance(v, dict) and "file_path" in v and "content" in v
    }


def document_text(name: str) -> str | None:
    """Return a corpus document's extracted text by filename, or None if unknown."""
    return _docs_by_filename().get(name)


def _references_from_context(context: str) -> list[Source]:
    return [Source(document=name) for name in corpus_filenames() if name in context]


DECOMPOSE_SYSTEM = """You are given an ANSWER to a corpus-level question. Break it into atomic
factual claims: each a single, self-contained statement that can be checked on its own. Copy the
meaning faithfully — do not add, remove, or embellish information, and ignore any inline citation
markers like [1]. Return only the claims."""


def _decompose(answer_text: str) -> list[Claim]:
    """Split a corpus-level prose answer into atomic claims.

    Aggregate claims carry no verbatim quote (a corpus-level synthesis isn't quotable from any
    single chunk); they're grounded by the Answer's source references instead.
    """
    gen = make_structured_llm(Answer, model=settings.gen_model)
    messages = [("system", DECOMPOSE_SYSTEM), ("human", answer_text)]
    decomposed = cast(Answer, gen.invoke(messages))
    return [Claim(text=c.text) for c in decomposed.claims]  # rebuild so quote stays None


def aggregate_answer(state: GraphState) -> GraphState:
    """Answer a corpus-level question, then structure the prose into verificable claims.

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

    answer = Answer(
        claims=_decompose(answer_text),
        references=_references_from_context(context),
    )

    return {
        "answer": answer,
        "context": context,
    }
