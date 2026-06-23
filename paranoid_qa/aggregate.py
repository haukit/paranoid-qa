"""The aggregate path: corpus-level QA via a LightRAG knowledge graph."""

from __future__ import annotations

import asyncio
import re
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

    asyncio.run(_build())


def _split_references(text: str) -> tuple[str, list[Source]]:
    """Split LightRAG's response into prose body and the list of cited source files."""
    parts = re.split(r"\n#+\s*References\s*\n", text, maxsplit=1)
    names = re.findall(r"-\s*\[\d+\]\s*(.+)", parts[1]) if len(parts) > 1 else []
    return parts[0].strip(), [Source(document=n.strip()) for n in names]


def aggregate_answer(state: GraphState) -> GraphState:
    """Answer a corpus-level question from the graph; return prose + Source references."""

    async def _query():
        rag = await _make_rag()
        return await rag.aquery(
            state["question"], param=QueryParam(mode="hybrid", include_references=True)
        )

    body, references = _split_references(cast(str, asyncio.run(_query())))
    return {"answer": Answer(claims=[Claim(text=body, quote="")]), "references": references}
