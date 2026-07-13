"""The specific QA path: quote-grounded answers to questions about particular documents.

Owns its LlamaIndex index (build + retrieval), its LangGraph nodes, its quote-based verifier,
and its graph fragment. Must not import the aggregate path.
"""
