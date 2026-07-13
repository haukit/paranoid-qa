"""Stable cross-path primitives.

These types are genuinely shared by both QA paths and change rarely.
Depends only on the standard library and Pydantic.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

RouteKind = Literal["specific", "aggregate"]
RunStatus = Literal["answered", "abstained"]


def document_id_from_filename(filename: str) -> str:
    """Derive a stable document identifier from a filename.

    Transitional: the identifier is currently the filename, which is what the
    persisted indexes key on. Callers must not assume this stays true once a
    Postgres-backed corpus assigns its own ids.
    """
    return filename


class SourceRef(BaseModel):
    """A citation: a reference to a document (and optionally a page within it)."""

    document_id: str
    filename: str
    page: str | None = None

    def __str__(self) -> str:
        return f"{self.filename} p.{self.page}" if self.page else self.filename
