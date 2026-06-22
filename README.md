# paranoid-qa

A grounded, self-verifying question-answering service over a document corpus.

Every answer is decomposed into atomic claims, each backed by a verbatim quote from the sources. A separate critic (a different model), excluded from the generator's reasoning,  independently verifies that each quote (a) actually exists in the retrieved documents and (b) genuinely supports its claim, attaching the source citation (document + page). Unsupported claims are regenerated; fabricated quotes trigger re-retrieval. All correction loops are bounded by a retry budget.

Stack: LangGraph (orchestration), LlamaIndex (hybrid retrieval + reranking), Pydantic (typed structured outputs), Ollama (local models)

## Quickstart

```bash
uv sync
ollama serve &
ollama pull qwen3.5:9b && ollama pull gemma4:12b && ollama pull bge-m3

# Put your documents (PDF / Markdown / text) in data/corpus, or point elsewhere:
#   export PARANOID_QA_CORPUS=/path/to/your/corpus
uv run python -c "from paranoid_qa.index import build_index; build_index()"   # one-time index build
uv run python -m paranoid_qa "your question here"
```

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

All models, paths, and budgets live in [`paranoid_qa/config.py`](paranoid_qa/config.py). Models can be swapped easily ([`paranoid_qa/models.py`](paranoid_qa/models.py)), so switching from Ollama to a hosted provider is a config change plus the matching extra (`uv sync --extra openai|anthropic|cohere`).
