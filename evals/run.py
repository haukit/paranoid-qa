#!/usr/bin/env python3
"""Run an eval suite through the graph and check each case against its expectations.

Case schema:
    [
      {
        "id": "probable_cause",
        "category": "specific",
        "question": "What was the probable cause of the accident?",
        "expect_pass": true,
        "checks": {
          "faithful": true,
          "empty_answer": false,
          "answer_contains": ["disorientation"],
          "cites": "report.pdf"
        }
      }
    ]

Fields: `id` (unique label), `category` (free-form tag, filtered by --category), `question`,
`expect_pass` (are the checks expected to hold?). Every key in `checks` is optional; only
those present are applied:
    faithful         state["faithful"] must equal this bool
    empty_answer     true => the answer must have zero claims
    answer_contains  every term must appear in the answer text (case-insensitive)
    cites            this document must appear in some claim's citation

Usage (the corpus must match the persisted index):
    PARANOID_QA_CORPUS=/path/to/corpus uv run python evals/run.py <cases.json>
    ... evals/run.py <cases.json> --only id_a,id_b      # run specific cases
    ... evals/run.py <cases.json> --category specific   # run one category
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from paranoid_qa.graph import build_graph


def check_case(case: dict, state: dict) -> list[str]:
    """Return a list of failure messages (empty == the case passed)."""
    checks = case.get("checks", {})
    answer = state.get("answer")
    claims = answer.claims if answer else []
    text = (answer.text if answer else "").lower()
    cited = {v.source.document for v in state.get("verdicts", []) if v.source}
    fails = []

    if "faithful" in checks and state.get("faithful") != checks["faithful"]:
        fails.append(f"faithful={state.get('faithful')} (want {checks['faithful']})")
    if "empty_answer" in checks and (len(claims) == 0) != checks["empty_answer"]:
        fails.append(f"claims={len(claims)} (want empty={checks['empty_answer']})")
    for term in checks.get("answer_contains", []):
        if term.lower() not in text:
            fails.append(f"answer missing {term!r}")
    if "cites" in checks and checks["cites"] not in cited:
        fails.append(f"{checks['cites']} not cited (cited: {sorted(cited)})")
    return fails


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", type=Path, help="JSON file of eval cases")
    ap.add_argument("--only", help="comma-separated case ids to run")
    ap.add_argument("--category", help="run only cases in this category")
    args = ap.parse_args()

    cases = json.loads(args.cases.read_text())
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in wanted]
    if args.category:
        cases = [c for c in cases if c.get("category") == args.category]

    app = build_graph()
    as_expected = 0
    for case in cases:
        state = app.invoke({"question": case["question"]})
        fails = check_case(case, state)
        passed = not fails
        expected = case.get("expect_pass", True)
        ok = passed == expected
        as_expected += ok

        mark = "  " if ok else "!!"
        result = "PASS" if passed else "FAIL"
        print(
            f"{mark} [{case.get('category', '?'):11}] {case['id']:24} {result}  (expect {'PASS' if expected else 'FAIL'})"
        )
        for f in fails:
            print(f"         - {f}")

    print(f"\n{as_expected}/{len(cases)} cases behaved as expected.")
    sys.exit(0 if as_expected == len(cases) else 1)


if __name__ == "__main__":
    main()
