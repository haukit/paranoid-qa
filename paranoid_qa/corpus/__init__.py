"""Canonical corpus access: source-document loading and full-text reads.

`loader` extracts source files into LlamaIndex documents (shared by both paths' index builds).
`repository` owns read access to canonical full-document text and the document list.
Neither knows about vector stores, BM25, LightRAG query internals, or LangGraph.
"""
