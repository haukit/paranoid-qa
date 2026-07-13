"""Build-time construction of the aggregate path's LightRAG knowledge graph.

Future home (do not implement yet): Postgres-backed LightRAG storage.
"""

from __future__ import annotations

from paranoid_qa.aggregate.retrieval import _make_rag, _run
from paranoid_qa.config import LIGHTRAG_DIR
from paranoid_qa.corpus.loader import load_full_documents


def build_aggregate_index() -> None:
    """Extract entities/relationships from the corpus into a LightRAG graph; run once."""
    LIGHTRAG_DIR.mkdir(parents=True, exist_ok=True)

    docs = load_full_documents()
    names = [doc.filename for doc in docs]
    texts = [doc.text for doc in docs]  # one whole-document string per file

    async def _build():
        rag = await _make_rag()
        await rag.ainsert(texts, file_paths=names)  # filenames become source references

    _run(_build())
