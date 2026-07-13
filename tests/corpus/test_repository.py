import json

import pytest

from paranoid_qa.corpus import repository


@pytest.fixture
def store(tmp_path, monkeypatch):
    (tmp_path / "kv_store_doc_status.json").write_text(
        json.dumps({"d1": {"file_path": "a.pdf"}, "d2": {"file_path": "b.pdf"}})
    )
    (tmp_path / "kv_store_full_docs.json").write_text(
        json.dumps(
            {
                "d1": {"file_path": "a.pdf", "content": "alpha text"},
                "d2": {"file_path": "b.pdf", "content": "beta text"},
            }
        )
    )
    monkeypatch.setattr(repository, "LIGHTRAG_DIR", tmp_path)
    repository._filenames.cache_clear()
    repository._text_by_filename.cache_clear()
    yield tmp_path
    repository._filenames.cache_clear()
    repository._text_by_filename.cache_clear()


def test_list_documents(store):
    docs = repository.list_documents()
    assert {d.filename for d in docs} == {"a.pdf", "b.pdf"}
    assert all(d.document_id == d.filename for d in docs)  # transitional identity


def test_get_document_text(store):
    assert repository.get_document_text("a.pdf") == "alpha text"


def test_unknown_document_returns_none(store):
    assert repository.get_document_text("missing.pdf") is None


def test_missing_store_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(repository, "LIGHTRAG_DIR", tmp_path)  # empty dir, no store files
    repository._filenames.cache_clear()
    repository._text_by_filename.cache_clear()
    with pytest.raises(FileNotFoundError):
        repository.list_documents()
    repository._filenames.cache_clear()
    repository._text_by_filename.cache_clear()
