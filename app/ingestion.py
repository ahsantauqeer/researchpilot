"""PDF ingestion: validate -> extract (per page) -> clean -> chunk -> embed -> upsert."""

from __future__ import annotations

import hashlib
import io
import re
from collections import Counter
from dataclasses import dataclass

from app.config import Settings
from app.schemas import Chunk, UploadResponse


class IngestionError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


# ---------------------------------------------------------------- validate
ALLOWED_CONTENT_TYPES = {"application/pdf", "application/x-pdf", "application/octet-stream", None, ""}


def validate_pdf(filename: str, data: bytes, content_type: str | None, max_mb: int) -> None:
    if not filename or not filename.lower().endswith(".pdf"):
        raise IngestionError("Only .pdf files are accepted.", 415)
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise IngestionError(f"Unexpected content type {content_type!r}; expected application/pdf.", 415)
    if len(data) == 0:
        raise IngestionError("The uploaded file is empty.", 400)
    if len(data) > max_mb * 1024 * 1024:
        raise IngestionError(f"File is larger than the {max_mb} MB limit.", 413)
    if not data[:1024].lstrip().startswith(b"%PDF-"):
        raise IngestionError("File content is not a PDF (missing %PDF- header).", 415)


# ---------------------------------------------------------------- extract
def extract_pages(data: bytes) -> list[str]:
    """Return text per page; index 0 is page 1."""
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:  # noqa: BLE001
                raise IngestionError("Encrypted PDFs are not supported.", 422) from exc
        return [(page.extract_text() or "") for page in reader.pages]
    except IngestionError:
        raise
    except (PdfReadError, Exception) as exc:  # noqa: BLE001 - pypdf raises many types on bad input
        raise IngestionError(f"Could not read PDF: {exc}", 422) from exc


# ---------------------------------------------------------------- clean
def _norm_line(line: str) -> str:
    # Page numbers change on every page; normalise digits so "Page 3" == "Page 4".
    return re.sub(r"\d+", "#", line.strip().lower())


def remove_repeated_headers_footers(pages: list[str], edge_lines: int = 2) -> list[str]:
    """Drop lines that appear at the top/bottom of most pages (running headers, footers)."""
    if len(pages) < 3:
        return pages
    def edge_indices(lines: list[str]) -> set[int]:
        non_empty = [i for i, ln in enumerate(lines) if ln.strip()]
        n = edge_lines if len(non_empty) > 2 * edge_lines else 1  # short pages: only first/last line
        return set(non_empty[:n] + non_empty[-n:])

    counts: Counter[str] = Counter()
    for text in pages:
        lines = text.splitlines()
        counts.update({_norm_line(lines[i]) for i in edge_indices(lines)})
    threshold = max(3, int(0.5 * len(pages)))
    repeated = {ln for ln, c in counts.items() if c >= threshold and ln}
    cleaned = []
    for text in pages:
        lines = text.splitlines()
        edge_idx = edge_indices(lines)
        cleaned.append(
            "\n".join(ln for i, ln in enumerate(lines) if not (i in edge_idx and _norm_line(ln) in repeated))
        )
    return cleaned


_REF_HEADING = re.compile(r"^\s*(\d+\.?\s*)?(references|bibliography|works cited)\s*$", re.IGNORECASE | re.MULTILINE)


def drop_reference_section(pages: list[str]) -> list[str]:
    """Remove everything after a 'References' heading in the second half of the document.

    Note: appendices that come after the references are dropped too. Turn this off with
    DROP_REFERENCES=false if your papers keep important content there.
    """
    start_page = len(pages) // 2
    for p in range(len(pages) - 1, start_page - 1, -1):
        matches = list(_REF_HEADING.finditer(pages[p]))
        if matches:
            cut = matches[-1].start()
            return pages[:p] + [pages[p][:cut]] + [""] * (len(pages) - p - 1)
    return pages


def normalise_whitespace(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # re-join hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_pages(pages: list[str], drop_references: bool = True) -> list[str]:
    pages = remove_repeated_headers_footers(pages)
    if drop_references:
        pages = drop_reference_section(pages)
    return [normalise_whitespace(p) for p in pages]


# ---------------------------------------------------------------- chunk
def split_text(text: str, size: int, overlap: int) -> list[str]:
    """Character-based splitting with overlap, preferring paragraph/sentence/word boundaries."""
    if overlap >= size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window = text[start:end]
            min_cut = int(size * 0.5)
            for sep in ("\n\n", ". ", "\n", " "):
                cut = window.rfind(sep)
                if cut >= min_cut:
                    end = start + cut + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        next_start = end - overlap
        # snap forward to a word boundary so chunks don't begin mid-word
        space = text.find(" ", next_start, end)
        start = space + 1 if space != -1 else next_start
    return chunks


def document_id_for(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def make_chunks(
    pages: list[str], document_id: str, filename: str, content_hash: str, size: int, overlap: int
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page_idx, text in enumerate(pages):
        page = page_idx + 1  # one-based page numbers in all citations
        for i, piece in enumerate(split_text(text, size, overlap)):
            if len(piece) < 30:  # page numbers, stray symbols
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{document_id}:p{page}:c{i}",
                    document_id=document_id,
                    filename=filename,
                    page=page,
                    chunk_index=i,
                    text=piece,
                    content_hash=content_hash,
                )
            )
    return chunks


# ---------------------------------------------------------------- pipeline
@dataclass
class PreparedDocument:
    document_id: str
    filename: str
    pages: int
    chunks: list[Chunk]


def prepare_pdf(filename: str, data: bytes, settings: Settings, content_type: str | None = None) -> PreparedDocument:
    validate_pdf(filename, data, content_type, settings.max_upload_mb)
    raw_pages = extract_pages(data)
    if not any(p.strip() for p in raw_pages):
        raise IngestionError(
            "No extractable text found. This is probably a scanned PDF; OCR is not supported yet.", 422
        )
    pages = clean_pages(raw_pages, settings.drop_references)
    content_hash = hashlib.sha256(data).hexdigest()
    doc_id = content_hash[:16]
    safe_name = re.sub(r"[^\w.\- ()]", "_", filename.replace("\\", "/").split("/")[-1])[:200]
    chunks = make_chunks(pages, doc_id, safe_name, content_hash, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise IngestionError("PDF produced no usable text chunks.", 422)
    return PreparedDocument(doc_id, safe_name, len(raw_pages), chunks)


def ingest_pdf(filename, data, settings, store, embedder, content_type=None) -> UploadResponse:
    doc = prepare_pdf(filename, data, settings, content_type)
    if store.document_exists(doc.document_id):  # idempotent: identical bytes -> same document_id
        info = next(d for d in store.list_documents() if d.document_id == doc.document_id)
        return UploadResponse(
            document_id=doc.document_id, filename=info.filename, pages_processed=info.pages,
            chunks_indexed=info.chunks, duplicate=True,
        )
    vectors = embedder.embed([c.text for c in doc.chunks])
    store.upsert_chunks(doc.chunks, vectors)
    return UploadResponse(
        document_id=doc.document_id, filename=doc.filename,
        pages_processed=doc.pages, chunks_indexed=len(doc.chunks),
    )
