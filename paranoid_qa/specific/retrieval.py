"""Query-time retrieval for the specific path: dense + BM25 fusion, then LLM reranking.

Returns typed `RetrievedChunk` values so the graph node never touches LlamaIndex node internals.
Future homes (do not implement yet): structured metadata filters, pgvector queries.
"""

from __future__ import annotations

from llama_index.core import Settings as LlamaSettings
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.schema import QueryBundle
from llama_index.retrievers.bm25 import BM25Retriever

from paranoid_qa.config import STORAGE_DIR, settings
from paranoid_qa.contracts.specific import RetrievedChunk
from paranoid_qa.llm.factory import make_embedder, make_llamaindex_llm


def _load_index():
    """Load the persisted specific index."""
    LlamaSettings.embed_model = make_embedder()  # query uses the same embed model
    LlamaSettings.llm = None  # type: ignore[assignment]
    storage_context = StorageContext.from_defaults(persist_dir=str(STORAGE_DIR))
    return load_index_from_storage(storage_context)


class _RerankingRetriever:
    """Fuses dense + BM25 candidates, then reranks with an LLM."""

    def __init__(self, base, reranker):
        self._base = base
        self._reranker = reranker

    def retrieve(self, query: str):
        nodes = self._base.retrieve(query)
        return self._reranker.postprocess_nodes(nodes, query_bundle=QueryBundle(query))

    async def aretrieve(self, query: str):
        nodes = await self._base.aretrieve(query)
        return await self._reranker.apostprocess_nodes(nodes, query_bundle=QueryBundle(query))


def get_retriever() -> _RerankingRetriever:
    """Build the fused, reranking retriever over the persisted index."""
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
        llm=make_llamaindex_llm(), top_n=settings.rerank_top_n, choice_batch_size=settings.top_k
    )
    return _RerankingRetriever(fusion, reranker)


async def retrieve_chunks(question: str) -> list[RetrievedChunk]:
    """Retrieve and rerank chunks for a question as typed `RetrievedChunk` values."""
    nodes = await get_retriever().aretrieve(question)
    return [
        {
            "text": n.text,
            "document": n.metadata.get("file_name", "?"),
            "page": n.metadata.get("page_label"),
        }
        for n in nodes
    ]
