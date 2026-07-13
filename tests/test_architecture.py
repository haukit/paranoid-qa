"""Architecture boundary tests: enforce the path dependency rules via import scanning.

Uses the standard library only (no new architecture-testing dependency).
"""

from __future__ import annotations

import ast
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent / "paranoid_qa"


def _imports(pyfile: Path) -> set[str]:
    tree = ast.parse(pyfile.read_text())
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
    return mods


def _under(subpkg: str) -> list[Path]:
    return list((PKG / subpkg).rglob("*.py"))


def _imports_prefix(mods: set[str], prefix: str) -> bool:
    return any(m == prefix or m.startswith(prefix + ".") for m in mods)


def test_specific_does_not_import_aggregate():
    for f in _under("specific"):
        assert not _imports_prefix(_imports(f), "paranoid_qa.aggregate"), f


def test_aggregate_does_not_import_specific():
    for f in _under("aggregate"):
        assert not _imports_prefix(_imports(f), "paranoid_qa.specific"), f


def test_corpus_imports_neither_path():
    for f in _under("corpus"):
        mods = _imports(f)
        assert not _imports_prefix(mods, "paranoid_qa.specific"), f
        assert not _imports_prefix(mods, "paranoid_qa.aggregate"), f


def test_only_workflow_graph_imports_both_paths():
    for f in PKG.rglob("*.py"):
        mods = _imports(f)
        if _imports_prefix(mods, "paranoid_qa.specific") and _imports_prefix(
            mods, "paranoid_qa.aggregate"
        ):
            assert f.relative_to(PKG).as_posix() == "workflow/graph.py", (
                f"{f} imports both paths but is not the composition root"
            )
