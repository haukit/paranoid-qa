# paranoid-qa

A grounded, self-verifying question-answering service over a document corpus.

Every answer is decomposed into atomic claims, each backed by a verbatim quote from the sources. A separate critic (a different model), excluded from the generator's reasoning,  independently verifies that each quote (a) actually exists in the retrieved documents and (b) genuinely supports its claim, attaching the source citation (document + page). Unsupported claims are regenerated; fabricated quotes trigger re-retrieval. All correction loops are bounded by a retry budget.

Stack: LangGraph (orchestration), LlamaIndex (hybrid retrieval + reranking), LightRAG (graph-based corpus-level retrieval), Pydantic (typed structured outputs), OpenAI (also supports local models via Ollama)

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

A router sends each question down one of two paths, each with its own grounding guarantee:

```
START -> route
    -> "specific"  -> retrieve -> grade
        -> "yes" -> generate -> verify
            -> accept       -> END
            -> revise       -> generate
            -> re_retrieve  -> retrieve
        -> "no"  -> rewrite -> retrieve
    -> "aggregate" -> lightrag -> verify_aggregate -> END
```

- route: classify the question as **specific** (about particular documents / events) or **aggregate** (corpus-level counts, trends, themes), then branch.

Specific path — grounded by verbatim quotes:

- retrieve: hybrid dense + BM25 retrieval, fused (reciprocal rank fusion) and reranked.
- grade: is the retrieved context relevant? If not, reformulate the query and retry.
- generate: emit the answer as `Claim{text, quote}` pairs, grounded strictly in the sources.
- verify: a deterministic check locates each quote in the sources (and mints its citation), then a different-family critic judges whether the quote actually supports the claim.

Aggregate path — grounded by source references:

- lightrag: a knowledge graph built over the whole corpus answers corpus-level questions, such as counts and trends across many documents.
- verify_aggregate: grounds the answer against the documents it cited

## Configuration

All models, paths, and budgets live in [`paranoid_qa/config.py`](paranoid_qa/config.py). Models can be swapped easily ([`paranoid_qa/models.py`](paranoid_qa/models.py)), so switching providers (to local Ollama, or to Anthropic / Cohere) is a config change plus the matching extra (`uv sync --extra ollama|anthropic|cohere`).

## TODO

- Router: add few-shot examples to reduce phrasing-sensitive misroutes (e.g. "which report involves X" vs "how many reports involve X").
- Retrieval recall: raise `rerank_top_n` / `top_k`, or swap the LLM reranker for a cross-encoder (should be faster).
- Rewrite node: anchor to the original question, show the rejected docs, keep a rewrite history; route `re_retrieve` through reformulation so it fetches different chunks.
- Aggregate eval: rephrase questions to be more unambiguous and grade on the cited references rather than prose substrings.
- `verify_aggregate`: add an LLM entailment pass (does the answer follow from the cited documents?) on top of reference existence.
- Evaluation: RAGAS faithfulness/context, retrieval ranking (MRR / NDCG / hit-rate), and critic precision/recall on labeled sets.
- Serving: FastAPI `/ask` endpoint, Phoenix tracing, Dockerfile.
