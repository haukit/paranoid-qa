# paranoid-qa

A grounded, self-verifying question-answering service over a document corpus.

Every answer is decomposed into atomic claims, each backed by a verbatim quote from the sources. A separate critic (a different model), excluded from the generator's reasoning,  independently verifies that each quote (a) actually exists in the retrieved documents and (b) genuinely supports its claim, attaching the source citation (document + page). Unsupported claims are regenerated; fabricated quotes trigger re-retrieval. All correction loops are bounded by a retry budget.

Stack: LangGraph (orchestration), LlamaIndex (hybrid retrieval + reranking), Pydantic (typed structured outputs), OpenAI (also supports local models via Ollama)

## Quickstart

```bash
uv sync
export OPENAI_API_KEY=sk-...        # the default provider is OpenAI

# Put your documents (PDF / Markdown / text) in data/corpus, or point elsewhere:
#   export PARANOID_QA_CORPUS=/path/to/your/corpus
uv run python -c "from paranoid_qa.index import build_index; build_index()"   # one-time index build
uv run python -m paranoid_qa "your question here"
```

Run fully locally instead: `uv sync --extra ollama`, set `provider` / `embed_provider` to `"ollama"` in config, and `ollama pull qwen3.5:9b gemma4:12b bge-m3`.

## How it works

```
START -> retrieve -> grade
    -> "yes" -> generate -> verify
        -> accept -> END
        -> revise -> generate
        -> re_retrieve -> retrieve
    -> "no" -> rewrite -> retrieve
```

- retrieve: hybrid dense + BM25 retrieval, fused (reciprocal rank fusion) and reranked.
- grade: is the retrieved context relevant? If not, reformulate the query and retry.
- generate: emit the answer as `Claim{text, quote}` pairs, grounded strictly in the sources.
- verify: a deterministic check locates each quote in the sources (and mints its citation), then a different-family critic judges whether the quote actually supports the claim.

## Configuration

All models, paths, and budgets live in [`paranoid_qa/config.py`](paranoid_qa/config.py). Models can be swapped easily ([`paranoid_qa/models.py`](paranoid_qa/models.py)), so switching providers (to local Ollama, or to Anthropic / Cohere) is a config change plus the matching extra (`uv sync --extra ollama|anthropic|cohere`).
