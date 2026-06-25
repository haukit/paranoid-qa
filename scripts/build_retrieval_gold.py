#!/usr/bin/env python3
"""Generate retrieval_gold.jsonl: a question paired with the chunk that answers it.

For each sampled chunk, we ask a generator model to write a specific question whose
answer lives in that chunk; the chunk's `document#page` is the gold answer.

We do stratified sampling here (sample same number of chunks per doc) as different docs
have different number of pages/chunks."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import cast

from llama_index.core import StorageContext
from llama_index.core.schema import BaseNode
from pydantic import BaseModel, Field

from paranoid_qa.config import STORAGE_DIR
from paranoid_qa.models import make_structured_llm

SEED = 23
PER_DOC = 6  # questions per document; sparse reports are capped at what they have
MIN_CHARS = 500  # skip covers, toc since those make poor questions
GEN_MODEL = "gpt-4o"  # pick a stronger model than the actual judge model
OUT_PATH = Path("evals/data/retrieval_gold.jsonl")


class GoldQA(BaseModel):
    question: str = Field(description="A specific question answerable ONLY from the passage.")
    reference_answer: str = Field(description="The answer, stated from the passage.")


def chunk_id(node: BaseNode) -> str:
    m = node.metadata
    return f"{m['file_name']}#{m.get('page_label')}"


def load_chunks() -> list[BaseNode]:
    storage = StorageContext.from_defaults(persist_dir=str(STORAGE_DIR))
    return list(storage.docstore.docs.values())


def sample_chunks(nodes: list[BaseNode], per_doc: int, seed: int, min_chars: int) -> list[BaseNode]:
    rng = random.Random(seed)
    by_doc: defaultdict[str, list[BaseNode]] = defaultdict(list)
    for n in nodes:
        if len(n.get_content()) >= min_chars:
            by_doc[n.metadata["file_name"]].append(n)
    out: list[BaseNode] = []
    for doc in sorted(by_doc):
        chunks = by_doc[doc]
        out.extend(rng.sample(chunks, k=min(per_doc, len(chunks))))

    return out


SYS = """You write evaluation questions for a retrieval system over NTSB accident reports.
Given ONE passage, write a single specific question whose answer is found ONLY in this passage
(a particular cause, finding, name, date, count, or recommendation unique to it).

Rules:
- The question must require this passage; never write something answerable from general aviation knowledge.
- Paraphrase. Do NOT copy distinctive phrases verbatim, or retrieval becomes a trivial keyword match.
- The reference answer must be fully supported by the passage."""


def make_row(i: int, node: BaseNode, qa: GoldQA) -> dict:
    return {
        "id": f"rg_{i:04d}",
        "question": qa.question,
        "gold_chunk_ids": [chunk_id(node)],
        "reference_answer": qa.reference_answer,
        "path": "specific",
    }


def main() -> None:
    sample = sample_chunks(load_chunks(), PER_DOC, SEED, MIN_CHARS)
    gen = make_structured_llm(GoldQA, model=GEN_MODEL, temperature=0)

    rows = []
    for i, node in enumerate(sample, 1):
        qa = cast(GoldQA, gen.invoke([("system", SYS), ("human", node.get_content())]))
        rows.append(make_row(i, node, qa))
        print(f"  rg_{i:04d}  {chunk_id(node):24s}  {qa.question}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {len(rows)} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
