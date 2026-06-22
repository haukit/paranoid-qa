from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = Path(
    os.getenv("PARANOID_QA_CORPUS", PROJECT_ROOT / "data" / "corpus" / "refund_policy")
)
STORAGE_DIR = PROJECT_ROOT / ".storage"  # persisted vector index (gitignored)


@dataclass(frozen=True)
class Settings:
    provider: str = "ollama"  # "ollama" | "openai" | "anthropic" | "cohere"

    gen_model: str = "qwen3.5:9b"  # the generator
    critic_model: str = "gemma4:12b"  # the critic, ideally from different family for decorrelation
    embed_model: str = "bge-m3"

    top_k: int = 10  # candidates retrieved (dense + BM25) before reranking
    rerank_top_n: int = 4  # kept after reranking -> passed to the generator

    max_attempts: int = 2  # hard cap on rewrite / revise / re_retrieve loops

    ollama_host: str = "localhost:11434"
    temperature: float = 0.0


settings = Settings()
