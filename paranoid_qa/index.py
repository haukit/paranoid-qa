"""Ingestion & retrieval."""

from __future__ import annotations

from llama_index.core import Settings as LlamaSettings
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.schema import QueryBundle
from llama_index.retrievers.bm25 import BM25Retriever

from paranoid_qa.config import CORPUS_DIR, STORAGE_DIR, settings
from paranoid_qa.models import make_embedder, make_li_llm


def build_index():
    """Load the corpus, embed it, and create the index."""
    LlamaSettings.embed_model = make_embedder()
    LlamaSettings.llm = None  # type: ignore

    documents = SimpleDirectoryReader(str(CORPUS_DIR)).load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=str(STORAGE_DIR))
    return index


def _load_index():
    """Load the persisted index."""
    LlamaSettings.embed_model = make_embedder()  # retriever uses same embed model for query
    LlamaSettings.llm = None  # type: ignore
    storage_context = StorageContext.from_defaults(persist_dir=str(STORAGE_DIR))
    return load_index_from_storage(storage_context)


def get_retriever():
    index = _load_index()

    dense = index.as_retriever(similarity_top_k=settings.top_k)
    bm25 = BM25Retriever.from_defaults(docstore=index.docstore, similarity_top_k=settings.top_k)

    fusion = QueryFusionRetriever(
        [dense, bm25],
        similarity_top_k=settings.top_k,
        num_queries=1,
        mode=FUSION_MODES.RECIPROCAL_RANK,
        use_async=False,
    )

    reranker = LLMRerank(
        llm=make_li_llm(), top_n=settings.rerank_top_n, choice_batch_size=settings.top_k
    )

    return _RerankingRetriever(fusion, reranker)


class _RerankingRetriever:
    def __init__(self, base, reranker):
        self._base = base
        self._reranker = reranker

    def retrieve(self, query: str):
        nodes = self._base.retrieve(query)
        return self._reranker.postprocess_nodes(nodes, query_bundle=QueryBundle(query))

    async def aretrieve(self, query: str):
        nodes = await self._base.aretrieve(query)
        return await self._reranker.apostprocess_nodes(nodes, query_bundle=QueryBundle(query))
