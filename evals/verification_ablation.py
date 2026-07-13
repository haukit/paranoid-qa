#!/usr/bin/env python3
"""Verification ablation: does the verify/revise/re_retrieve loop reduce the
final-answer fabrication rate?"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from paranoid_qa.specific.verification import locate_quote
from paranoid_qa.workflow.graph import build_graph

GOLD = Path("evals/data/retrieval_gold.jsonl")
OUT = Path("evals/studies/verification_ablation_results.jsonl")


def load_questions() -> list[str]:
    rows = [json.loads(line) for line in GOLD.read_text().splitlines() if line.strip()]
    return [r["question"] for r in rows if r.get("path", "specific") == "specific"]


def record(question: str, mode: str, state: dict) -> dict:
    answer = state.get("answer")
    chunks = state.get("specific_chunks", [])
    claims = [
        {"text": c.text, "quote": c.quote, "located": locate_quote(c.quote, chunks) is not None}
        for c in (answer.claims if answer else [])
        if getattr(c, "quote", None) is not None
    ]
    return {
        "question": question,
        "mode": mode,
        "attempts": state.get("specific_revision_attempts", 0),
        "verdicts": [
            {"verdict": v.verdict, "explanation": v.explanation}
            for v in state.get("specific_verdicts", [])
        ],
        "n_quoted": len(claims),
        "fabricated": sum(1 for c in claims if not c["located"]),
        "claims": claims,
        "chunks": [
            {"id": f"{ch.get('document', '?')}#{ch.get('page')}", "text": ch.get("text", "")}
            for ch in chunks
        ],
    }


def main() -> None:
    questions = load_questions()
    if len(sys.argv) > 1:
        questions = questions[: int(sys.argv[1])]

    records: list[dict] = []
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for verify_enabled in (False, True):
            mode = "on" if verify_enabled else "off"
            app = build_graph(verify_enabled=verify_enabled)
            for q in questions:
                rec = record(q, mode, asyncio.run(app.ainvoke({"question": q})))
                records.append(rec)
                f.write(json.dumps(rec) + "\n")
                f.flush()
                print(f"  [{mode}] fab={rec['fabricated']}/{rec['n_quoted']}  {q[:60]}")

    for mode in ("off", "on"):
        rs = [r for r in records if r["mode"] == mode]
        n = sum(r["n_quoted"] for r in rs)
        fab = sum(r["fabricated"] for r in rs)
        print(
            f"{mode.upper():4} fabrication={fab / n:.1%}  ({fab}/{n})"
            if n
            else f"{mode}: no claims"
        )
    print(f"\nper-question records -> {OUT}")


if __name__ == "__main__":
    main()
