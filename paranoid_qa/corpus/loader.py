"""Source-document loading, shared by both paths' index builds.

Centralizes the extraction previously duplicated between the specific and aggregate index
builders. Knows how to read the corpus files and preserve filename/page metadata; it does not
know about vector stores, BM25, LightRAG, or LangGraph.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document

from paranoid_qa.config import CORPUS_DIR


@dataclass(frozen=True)
class CorpusDocument:
    """A whole source document: its filename and full extracted text."""

    filename: str
    text: str


def load_pages() -> list[Document]:
    """Load the corpus as page-level LlamaIndex documents (filename/page metadata preserved)."""
    return SimpleDirectoryReader(str(CORPUS_DIR)).load_data()


def group_pages_by_document(pages: list[Document]) -> dict[str, list[Document]]:
    """Group page documents by their source filename."""
    by_file: dict[str, list[Document]] = defaultdict(list)
    for page in pages:
        by_file[page.metadata.get("file_name", "?")].append(page)
    return dict(by_file)


def load_full_documents() -> list[CorpusDocument]:
    """Load each source file as one whole-document record (pages concatenated)."""
    grouped = group_pages_by_document(load_pages())
    return [
        CorpusDocument(filename=name, text="\n".join(page.text for page in pages))
        for name, pages in grouped.items()
    ]
