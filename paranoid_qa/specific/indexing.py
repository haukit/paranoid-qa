"""Build-time construction of the specific path's LlamaIndex vector/doc store.

Future homes (do not implement yet): per-chunk contextual blurbs (contextual retrieval),
indexed metadata for filtering, and pgvector-backed persistence.
"""

from __future__ import annotations

from llama_index.core import Settings as LlamaSettings
from llama_index.core import VectorStoreIndex

from paranoid_qa.config import STORAGE_DIR
from paranoid_qa.corpus.loader import load_pages
from paranoid_qa.llm.factory import make_embedder


def build_specific_index() -> None:
    """Load the corpus, embed it, and persist the specific retrieval index."""
    LlamaSettings.embed_model = make_embedder()
    LlamaSettings.llm = None  # type: ignore[assignment]

    index = VectorStoreIndex.from_documents(load_pages())
    index.storage_context.persist(persist_dir=str(STORAGE_DIR))
