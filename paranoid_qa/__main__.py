"""CLI entry point: `python -m paranoid_qa "your question"`."""

from __future__ import annotations

import sys

from paranoid_qa.graph import build_graph


def main() -> None:
    if len(sys.argv) < 2:
        print('usage: python -m paranoid_qa "your question"')
        raise SystemExit(1)
    question = sys.argv[1]

    app = build_graph()
    result = app.invoke({"question": question})
    answer = result["answer"]

    print(answer.text, "\n")
    for c in answer.claims:
        line = f"- {c.text}"
        if c.quote is not None:
            line += f"\n    quote: {c.quote!r}"
        print(line)
    if answer.references:
        print("\nsources:", ", ".join(str(s) for s in answer.references))


if __name__ == "__main__":
    main()
