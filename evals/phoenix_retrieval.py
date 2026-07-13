#!/usr/bin/env python3
"""Recall@k for the specific-path retriever.

Task: Run the retriever on each gold question and return the delivered
chunk ids (document#page).

Evaluators: recall vs gold_chunk_ids, and MRR of the first gold hit.
k is settings.rerank_top_n: the final set of chunks the generator actually sees.

Involves LLM calls for embedding the questions."""

from __future__ import annotations

import json
from pathlib import Path

from phoenix.client import Client
from phoenix.client.experiments import run_experiment

from paranoid_qa.config import settings
from paranoid_qa.specific.retrieval import get_retriever

GOLD = Path("evals/data/retrieval_gold.jsonl")


def chunk_id(n) -> str:
    return f"{n.metadata.get('file_name', '?')}#{n.metadata.get('page_label')}"


def load_rows() -> list[dict]:
    return [json.loads(line) for line in GOLD.read_text().splitlines() if line.strip()]


def recall(output, expected) -> float:
    """Fraction of gold chunks present in the retrieved set."""
    gold = set(expected["gold_chunk_ids"])
    return len(gold & set(output)) / len(gold) if gold else float("nan")


def mrr(output, expected) -> float:
    """Reciprocal rank of the first retrieved gold chunk (0 if none retrieved)."""
    gold = set(expected["gold_chunk_ids"])
    for rank, cid in enumerate(output, start=1):  # top retrieved chunk is rank 1
        if cid in gold:
            return 1.0 / rank
    return 0.0


def main() -> None:
    # Create dataset from dicts instead of a dataframe since the dataframe
    # path stringifies list cells, which throws the equivalence check off
    rows = load_rows()
    dataset = Client().datasets.create_dataset(
        name="retrieval-gold",
        inputs=[{"question": r["question"]} for r in rows],
        outputs=[
            {"gold_chunk_ids": r["gold_chunk_ids"], "reference_answer": r["reference_answer"]}
            for r in rows
        ],
        metadata=[{"id": r["id"], "path": r["path"]} for r in rows],
    )

    retriever = get_retriever()

    def task(input) -> list[str]:
        return [chunk_id(n) for n in retriever.retrieve(input["question"])]

    run_experiment(
        dataset=dataset,
        task=task,
        evaluators=[recall, mrr],
        experiment_name=f"retrieval-baseline-k{settings.rerank_top_n}",
    )


if __name__ == "__main__":
    main()
