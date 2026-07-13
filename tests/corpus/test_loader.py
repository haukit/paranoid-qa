from llama_index.core.schema import Document

from paranoid_qa.corpus import loader


def _pages():
    return [
        Document(text="p1", metadata={"file_name": "a.pdf"}),
        Document(text="p2", metadata={"file_name": "a.pdf"}),
        Document(text="q1", metadata={"file_name": "b.pdf"}),
    ]


def test_group_pages_by_document():
    grouped = loader.group_pages_by_document(_pages())
    assert set(grouped) == {"a.pdf", "b.pdf"}
    assert len(grouped["a.pdf"]) == 2
    assert len(grouped["b.pdf"]) == 1


def test_load_full_documents_concatenates_pages(monkeypatch):
    monkeypatch.setattr(loader, "load_pages", _pages)
    docs = {d.filename: d.text for d in loader.load_full_documents()}
    assert docs == {"a.pdf": "p1\np2", "b.pdf": "q1"}
