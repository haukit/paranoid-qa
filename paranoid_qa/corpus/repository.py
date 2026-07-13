"""Read access to canonical full-document text and the corpus document list.

TEMPORARY file-backed implementation: full document text and filenames are read from the
LightRAG JSON key-value stores that ship inside the persisted index. This is the transitional
compromise from the refactor plan: it fixes the dependency direction now (the specific verifier
reads document text through here, not through the aggregate path) without bundling the Postgres
migration. Only this module may know these JSON filenames; a Postgres corpus repository will
replace it later behind the same two public functions.
"""

from __future__ import annotations

import json
from functools import lru_cache

from paranoid_qa.config import LIGHTRAG_DIR
from paranoid_qa.contracts.common import SourceRef, document_id_from_filename

_DOC_STATUS_FILE = "kv_store_doc_status.json"
_FULL_DOCS_FILE = "kv_store_full_docs.json"


@lru_cache(maxsize=1)
def _filenames() -> tuple[str, ...]:
    """Filenames the index was built from (read from the LightRAG doc-status store)."""
    status = json.loads((LIGHTRAG_DIR / _DOC_STATUS_FILE).read_text())
    return tuple(sorted({v["file_path"] for v in status.values() if v.get("file_path")}))


@lru_cache(maxsize=1)
def _text_by_filename() -> dict[str, str]:
    """Map each corpus filename to its full extracted text (from the LightRAG full-docs store)."""
    docs = json.loads((LIGHTRAG_DIR / _FULL_DOCS_FILE).read_text())
    return {
        v["file_path"]: v["content"]
        for v in docs.values()
        if isinstance(v, dict) and "file_path" in v and "content" in v
    }


def list_documents() -> tuple[SourceRef, ...]:
    """Return a reference for every document in the corpus.

    Raises FileNotFoundError if the index store is absent (e.g. stub mode); callers decide
    whether to treat that as an empty corpus.
    """
    return tuple(
        SourceRef(document_id=document_id_from_filename(name), filename=name)
        for name in _filenames()
    )


def get_document_text(document_id: str) -> str | None:
    """Return a document's full extracted text by id, or None if unknown.

    Raises FileNotFoundError if the index store is absent.
    """
    return _text_by_filename().get(document_id)
