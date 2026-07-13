"""The aggregate QA path: corpus-level synthesis via a LightRAG knowledge graph.

Owns its LightRAG index (build + retrieval), its answer/verification nodes, and its graph
fragment. Grounding is by source reference, not verbatim quote. Must not import the specific path.
"""
