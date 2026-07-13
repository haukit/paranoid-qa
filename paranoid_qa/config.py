from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# USD per token (input, output). Need to update when OpenAI changes prices.
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15e-6, 0.60e-6),
    "gpt-5.4-mini": (0.75e-6, 4.50e-6),
    "gpt-5.4-nano": (0.20e-6, 1.25e-6),
    "text-embedding-3-small": (0.02e-6, 0.0),
    "text-embedding-3-large": (0.13e-6, 0.0),
}


def token_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    """Return the USD cost of a call, pricing input and output tokens by model (prefix-matched)."""
    prices = MODEL_PRICES.get(model)
    if prices is None:  # match dated variants like gpt-4o-mini-2024-07-18
        prices = next((p for k, p in MODEL_PRICES.items() if model.startswith(k)), (0.0, 0.0))
    return tokens_in * prices[0] + tokens_out * prices[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PARANOID_QA_",
        env_file=".env",
        extra="ignore",
    )

    provider: str = "openai"
    embed_provider: str = "openai"

    gen_model: str = "gpt-4o-mini"  # the generator (local: "qwen3.5:9b")
    critic_model: str = "gpt-5.4-nano"  # different family from the generator (local: "gemma4:12b")
    embed_model: str = "text-embedding-3-small"  # local: "bge-m3"

    # The critic must be a different model family from the generator so their errors decorrelate.
    # Families are configured explicitly (never inferred from model-name string parsing) and
    # validated at startup by llm.policy.validate_model_policy.
    gen_model_family: str = "gpt-4o"  # local: "qwen"
    critic_model_family: str = "gpt-5.4"  # local: "gemma"

    corpus: Path = PROJECT_ROOT / "data" / "corpus" / "sample"
    storage: Path = PROJECT_ROOT / ".storage"
    lightrag_dir: Path = PROJECT_ROOT / ".lightrag"

    top_k: int = 10  # candidates retrieved (dense + BM25) before reranking
    rerank_top_n: int = 4  # kept after reranking -> passed to the generator

    max_attempts: int = 2  # hard cap on rewrite / revise / re_retrieve loops
    max_verification_retries: int = 1

    ollama_host: str = "localhost:11434"
    temperature: float = 0.0

    request_timeout_seconds: int = 90
    max_concurrent_requests: int = 2
    max_query_chars: int = 1000

    # Demo access control.
    demo_require_access: bool = True
    demo_invite_code: str | None = None
    demo_session_days: int = 7
    demo_questions_per_session: int = 20
    demo_global_daily_limit: int = 100
    demo_disabled: bool = False
    demo_secret_key: str | None = None

    cors_allow_origins: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

CORPUS_DIR = settings.corpus
STORAGE_DIR = settings.storage
LIGHTRAG_DIR = settings.lightrag_dir
