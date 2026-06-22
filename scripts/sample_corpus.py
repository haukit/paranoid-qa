#!/usr/bin/env python3
"""Copy N random PDFs from a source directory into a small sample corpus.

Usage:
    uv run python scripts/sample_corpus.py SOURCE --n 100
    uv run python scripts/sample_corpus.py SOURCE --n 50 --dest DEST --seed 0

Then point the pipeline at the sample:
    export PARANOID_QA_CORPUS=DEST
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Copy N random PDFs into a sample corpus.")
    parser.add_argument(
        "source", type=Path, help="Directory to sample PDFs from (searched recursively)."
    )
    parser.add_argument("--n", type=int, default=100, help="How many PDFs to copy (default: 100).")
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("data/sample_corpus"),
        help="Destination directory (default: data/sample_corpus).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed(default: 0).",
    )
    args = parser.parse_args(argv)

    if not args.source.is_dir():
        print(f"error: source is not a directory: {args.source}", file=sys.stderr)
        return 2

    pdfs = sorted(p for p in args.source.rglob("*.pdf") if p.is_file())
    if not pdfs:
        print(f"error: no .pdf files found under {args.source}", file=sys.stderr)
        return 1

    n = min(args.n, len(pdfs))
    random.seed(args.seed)
    chosen = random.sample(pdfs, n)

    args.dest.mkdir(parents=True, exist_ok=True)
    for src in chosen:
        target = args.dest / src.name
        shutil.copy2(src, target)

    print(f"copied {n} of {len(pdfs)} PDFs -> {args.dest}")
    print("point the pipeline at it:")
    print(f"  export PARANOID_QA_CORPUS={args.dest.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
