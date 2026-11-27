import pytest

from app.ingestion import (
    IngestionError,
    clean_pages,
    extract_pages,
    ingest_pdf,
    prepare_pdf,
    split_text,
    validate_pdf,
)
from tests.conftest import PAPER_ALPHA, make_pdf


def test_validate_rejects_bad_inputs():
    with pytest.raises(IngestionError) as e:
        validate_pdf("notes.txt", b"%PDF-1.4", "application/pdf", 1)
    assert e.value.status_code == 415
    with pytest.raises(IngestionError) as e:
        validate_pdf("fake.pdf", b"hello world", "application/pdf", 1)
    assert e.value.status_code == 415
    with pytest.raises(IngestionError) as e:
        validate_pdf("big.pdf", b"%PDF-" + b"0" * (1024 * 1024 + 1), "application/pdf", 1)
    assert e.value.status_code == 413
    with pytest.raises(IngestionError):
        validate_pdf("empty.pdf", b"", "application/pdf", 1)
    with pytest.raises(IngestionError):
        validate_pdf("x.pdf", b"%PDF-1.4", "text/html", 1)


def test_extract_pages_one_based_metadata(settings):
    doc = prepare_pdf("alpha.pdf", make_pdf(PAPER_ALPHA), settings)
    assert doc.pages == 2
    assert {c.page for c in doc.chunks} == {1, 2}
    assert "TileAttn" in " ".join(c.text for c in doc.chunks if c.page == 1)
    c = doc.chunks[0]
    assert c.document_id == doc.document_id and c.filename == "alpha.pdf"
    assert c.chunk_id == f"{doc.document_id}:p{c.page}:c{c.chunk_index}"


def test_ids_are_deterministic(settings):
    data = make_pdf(PAPER_ALPHA)
    a, b = prepare_pdf("alpha.pdf", data, settings), prepare_pdf("renamed.pdf", data, settings)
    assert a.document_id == b.document_id
    assert [c.chunk_id for c in a.chunks] == [c.chunk_id for c in b.chunks]


def test_split_text_size_and_overlap():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = split_text(text, 200, 50)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    # consecutive chunks overlap
    assert chunks[0].split()[-1] in chunks[1]


def test_clean_removes_headers_footers_and_references():
    pages = [f"Journal of Things 2024\nBody text page {i} about attention.\nPage {i}" for i in range(1, 6)]
    pages[4] = "Journal of Things 2024\nConclusion text.\nReferences\n[1] Someone. A paper.\nPage 5"
    cleaned = clean_pages(pages, drop_references=True)
    assert all("Journal of Things" not in p for p in cleaned)
    assert "Body text page 1" in cleaned[0]
    assert "Conclusion text." in cleaned[4] and "Someone" not in cleaned[4]


def test_scanned_pdf_rejected(settings):
    blank = make_pdf([""])
    assert extract_pages(blank) == [""]
    with pytest.raises(IngestionError) as e:
        prepare_pdf("scan.pdf", blank, settings)
    assert e.value.status_code == 422


def test_reupload_creates_no_duplicates_and_delete(settings, store, embedder):
    data = make_pdf(PAPER_ALPHA)
    first = ingest_pdf("alpha.pdf", data, settings, store, embedder)
    second = ingest_pdf("alpha.pdf", data, settings, store, embedder)
    assert not first.duplicate and second.duplicate
    assert second.document_id == first.document_id
    assert len(list(store.iter_chunks())) == first.chunks_indexed
    assert store.delete_document(first.document_id)
    assert list(store.iter_chunks()) == []
    assert not store.delete_document(first.document_id)
