#!/usr/bin/env python3
"""Measure the critics' precision/recall on labelled sets.

Two routes, dispatched by case shape (presence of "source_text"):
- specific: does a verbatim quote support a claim?
- aggregate: is a synthesized answer supported by the retrieved context?

Case schema:
    specific:
        {
          "id": "kingair_cause",            # unique label
          "gold": "supported",              # supported|unsupported|contradicted|fabricated
          "claim": "The probable cause ...",  # the atomic statement under test
          "quote": "the probable cause ...",  # span the claim cites (verbatim if grounded)
          "source_text": "National ...",      # the chunk the quote should live in
          "document": "AAR0301.pdf",          # provenance shown in the citation
          "page": "viii"
        }
    aggregate:
        {
          "id": "vfr_imc_theme",            # unique label
          "gold": "supported",              # supported|unsupported|contradicted
          "context": "In the Safari ...",     # retrieved evidence the answer is judged against
          "answer": "At least two ...",       # synthesized corpus-level answer under test
          "references": ["AIR2205.pdf", ...]  # cited docs; [] exercises the empty-refs gate
        }

Confusion matrix:
- positive: critic returns "supported" -> TP: gold is supported, FP: gold is unsupported
- negative: critic returns "unsupported" -> TN: gold is unsupported, FN: gold is supported

Usage:
    uv run python evals/critic.py data/eval/critic_cases.json
    uv run python evals/critic.py data/eval/aggregate_critic_cases.json
    ... <cases.json> --only id_a,id_b      # specific cases
    ... <cases.json> --gold supported      # one gold class
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from paranoid_qa.schemas import Claim, RetrievedChunk, Source
from paranoid_qa.verify import verify_claim


def run_case(case: dict) -> dict:
    """Dispatch by case shape; return a record of gold vs. predicted verdict."""
    if "source_text" in case:  # specific-path case
        claim = Claim(text=case["claim"], quote=case["quote"])
        chunk: RetrievedChunk = {
            "text": case["source_text"],
            "document": case["document"],
            "page": case["page"],
        }
        verdict = asyncio.run(verify_claim(claim, [chunk]))
        is_gate = case["gold"] == "fabricated"
    else:  # aggregate-path case
        claim = Claim(text=case["answer"])  # no quote
        references = [Source(document=d) for d in case["references"]]
        verdict = asyncio.run(verify_claim(claim, [], case["context"], references))
        is_gate = not case["references"]
    return {
        "id": case["id"],
        "gold": case["gold"],
        "pred": verdict.verdict,
        "gold_pos": case["gold"] == "supported",
        "pred_pos": verdict.verdict == "supported",
        "is_gate": is_gate,
        "explanation": verdict.explanation,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", help="path to a *_cases.json file")
    ap.add_argument("--only", help="comma-separated case ids")
    ap.add_argument("--gold", help="filter to one gold class")
    args = ap.parse_args()

    cases = json.loads(Path(args.cases).read_text())
    if args.only:
        keep = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in keep]
    if args.gold:
        cases = [c for c in cases if c["gold"] == args.gold]

    tp = fp = fn = tn = exact = 0
    gate_total = gate_ok = 0
    false_positives = []

    for c in cases:
        r = run_case(c)
        if r["gold_pos"] and r["pred_pos"]:
            tp += 1
        elif not r["gold_pos"] and r["pred_pos"]:
            fp += 1
            false_positives.append(r)
        elif r["gold_pos"] and not r["pred_pos"]:
            fn += 1
        else:
            tn += 1
        exact += r["gold"] == r["pred"]
        if r["is_gate"]:
            gate_total += 1
            gate_ok += r["gold"] == r["pred"]
        mark = "ok " if r["gold_pos"] == r["pred_pos"] else "XX "
        print(f"{mark}{r['id']:42s} gold={r['gold']:13s} pred={r['pred']:13s}")

    n = len(cases)
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else float("nan")

    print(f"cases: {n}   exact-match (verdict): {exact}/{n} = {exact / n:.0%}")
    print("confusion (positive = 'supported'):")
    print(f"    TP={tp}  FP={fp}")
    print(f"    FN={fn}  TN={tn}")
    print(f"precision={prec:.3f}  recall={rec:.3f}  f1={f1:.3f}")
    if gate_total:
        print(f"deterministic gate: {gate_ok}/{gate_total} resolved without an LLM call")
    if false_positives:
        print("\nFALSE POSITIVES:")
        for r in false_positives:
            print(f"  {r['id']} (gold={r['gold']}): {r['explanation']}")


if __name__ == "__main__":
    main()
