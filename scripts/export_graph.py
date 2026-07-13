#!/usr/bin/env python3
"""Export a Mermaid visualization of the compiled LangGraph to docs/graph.mmd.

uv run python scripts/export_graph.py
"""

from __future__ import annotations

from pathlib import Path

from paranoid_qa.workflow.graph import build_graph

OUT = Path("docs/graph.mmd")


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build_graph().get_graph().draw_mermaid())
    print(f"wrote Mermaid diagram -> {OUT}")


if __name__ == "__main__":
    main()
