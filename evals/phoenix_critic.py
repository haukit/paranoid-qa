#!/usr/bin/env python3
"""Critic precision/recall over critic_gold.jsonl, as a Phoenix experiment.

Task: run verify_claim on each labeled (claim, quote, source) triple and return its verdict.

Evaluators: verdict_exact (4-way accuracy) and confusion_cell (binary
supported-vs-not).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Mapping

from phoenix.client import Client
from phoenix.client.experiments import run_experiment

from paranoid_qa.schemas import Claim, RetrievedChunk
from paranoid_qa.verify import verify_claim

GOLD = Path("evals/data/critic_gold.jsonl")


def load_rows() -> list[dict]:
    return [json.loads(line) for line in GOLD.read_text().splitlines() if line.strip()]


def task(input) -> dict:
    claim = Claim(text=input["claim"], quote=input["quote"])
    chunk: RetrievedChunk = {
        "text": input["source_text"],
        "document": input["document"],
        "page": input["page"],
    }
    v = verify_claim(claim, [chunk])
    return {"verdict": v.verdict, "explanation": v.explanation}


def verdict_exact(output, expected) -> float:
    """Did the critic name the exact right verdict, out of all four?

    Tests whether the critic understands which kind of failure it is looking at, e.g.
    telling `contradicted` apart from `unsupported`."""
    return float(output["verdict"] == expected["gold"])


def confusion_cell(output, expected):
    """Did the critic make the right accept-or-reject decision?

    Collapses the four verdicts into two: `supported` is positive, everything else
    (`unsupported`, `contradicted`, `fabricated`) is negative. Returns (score, label, explanation)."""
    gold_pos = expected["gold"] == "supported"
    pred_pos = output["verdict"] == "supported"
    cell = (
        "TP"
        if gold_pos and pred_pos
        else "FP"
        if not gold_pos and pred_pos
        else "FN"
        if gold_pos and not pred_pos
        else "TN"
    )
    return (1.0 if gold_pos == pred_pos else 0.0, cell, None)


def main() -> None:
    rows = load_rows()
    dataset = Client().datasets.create_dataset(
        name="critic-gold",
        inputs=[
            {
                "claim": c["claim"],
                "quote": c["quote"],
                "source_text": c["source_text"],
                "document": c["document"],
                "page": c["page"],
            }
            for c in rows
        ],
        outputs=[{"gold": c["gold"]} for c in rows],
        metadata=[{"id": c["id"], "origin": c["origin"]} for c in rows],
    )

    exp = run_experiment(
        dataset=dataset,
        task=task,
        evaluators=[verdict_exact, confusion_cell],
        experiment_name="critic-baseline",
    )

    cells = Counter(
        r.result["label"]
        for r in exp["evaluation_runs"]
        if r.name == "confusion_cell" and isinstance(r.result, Mapping)
    )
    tp, fp, fn = cells["TP"], cells["FP"], cells["FN"]
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    exact = [
        r.result["score"]
        for r in exp["evaluation_runs"]
        if r.name == "verdict_exact" and isinstance(r.result, Mapping)
    ]
    accuracy = sum(exact) / len(exact) if exact else float("nan")

    print(f"precision={precision:.3f}  recall={recall:.3f}")
    print(f"confusion: TP={tp} FP={fp} FN={fn} TN={cells['TN']}")
    print(f"4-way accuracy={accuracy:.3f}  ({int(sum(exact))}/{len(exact)})")


if __name__ == "__main__":
    main()
