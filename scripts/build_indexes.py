#!/usr/bin/env python3
"""Build both path-owned indexes: the specific vector/doc store and the aggregate graph.

Each path owns its own index build; this is a thin coordinator for building both at once.
"""

from __future__ import annotations

from paranoid_qa.aggregate.indexing import build_aggregate_index
from paranoid_qa.specific.indexing import build_specific_index


def main() -> None:
    build_specific_index()
    build_aggregate_index()


if __name__ == "__main__":
    main()
