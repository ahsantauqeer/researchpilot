"""Shared test fixtures: a tiny PDF writer, a deterministic fake embedder and a scripted fake LLM.

Unit tests never call a real LLM or download models.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable

import pytest

from app.config import Settings
from app.llm import LLMOutputError, LLMUsage
from app.retrieval import RetrievalConfig, Retriever
from app.vector_store import VectorStore


# ------------------------------------------------------------------ PDF writer
def make_pdf(pages: list[str]) -> bytes:
    """Write a minimal valid text PDF (one string per page, '\\n' separates lines)."""
    objs: list[bytes] = []
    n = len(pages)
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(n))
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode())
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for i, text in enumerate(pages):
        ops = ["BT", "/F1 10 Tf", "12 TL", "40 770 Td"]
        for line in text.split("\n"):
            esc = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            ops.append(f"({esc}) Tj T*")
        ops.append("ET")
        stream = "\n".join(ops).encode("latin-1")
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {5 + 2 * i} 0 R >>".encode()
        )
        objs.append(f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream")
    out = b"%PDF-1.4\n"
    offsets = []
    for idx, obj in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{idx} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out


# ------------------------------------------------------------------ fakes
class FakeEmbedder:
    """Hashed bag-of-words vectors: texts sharing words are close. Deterministic and fast."""

    model_name = "fake-hash-embedder"
    dimension = 256

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * self.dimension
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dimension] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out


class ScriptedLLM:
    """Fake LLM. `handlers` maps schema name -> callable(system, user) returning a model instance
    (or raising LLMOutputError). Records every call for assertions."""

    def __init__(self, handlers: dict[str, Callable] | None = None):
        self.handlers = handlers or {}
        self.calls: list[dict] = []

    def complete_json(self, system, user, schema, *, fast=False):
        self.calls.append({"schema": schema.__name__, "system": system, "user": user, "fast": fast})
        handler = self.handlers.get(schema.__name__)
        if handler is None:
            raise LLMOutputError(f"no handler for {schema.__name__}")
        return handler(system, user), LLMUsage(model="fake", prompt_tokens=10, completion_tokens=5)

    def count(self, schema_name: str) -> int:
        return sum(1 for c in self.calls if c["schema"] == schema_name)


# ------------------------------------------------------------------ fixtures
PAPER_ALPHA = [
    "Alpha: Efficient Attention\nAbstract. We introduce TileAttn, an attention algorithm that reduces GPU memory\n"
    "by computing attention in tiles that fit in on-chip SRAM. TileAttn never materialises the full attention matrix.",
    "Method. TileAttn uses tiling and recomputation in the backward pass. Memory usage is linear in sequence length\n"
    "instead of quadratic. On long sequences TileAttn is 2x faster than standard attention in our setup.",
]
PAPER_BETA = [
    "Beta: Low Rank Adapters\nAbstract. We propose RankAdapt, which freezes the pretrained weights and trains small\n"
    "low rank matrices. RankAdapt reduces the number of trainable parameters by 10000 times.",
    "Results. RankAdapt needs very little optimizer memory, which makes fine tuning possible on a single\n"
    "consumer GPU. Inference latency is unchanged because adapters are merged into the weights.",
]


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(_env_file=None, llm_api_key="test", qdrant_path=":memory:", chunk_size=300,
                    chunk_overlap=50, top_k=4, max_rounds=3, use_hybrid=True, use_rerank=False,
                    collection_name="test", pdf_dir=str(tmp_path / "pdfs"))


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def store(settings, embedder) -> VectorStore:
    return VectorStore.from_settings(settings, embedder.dimension, embedder.model_name)


@pytest.fixture
def retriever(store, embedder, settings) -> Retriever:
    return Retriever(store, embedder, RetrievalConfig(top_k=settings.top_k, hybrid=True))


@pytest.fixture
def indexed(settings, store, embedder):
    """Store with two small papers indexed. Returns {filename: document_id}."""
    from app.ingestion import ingest_pdf

    ids = {}
    for name, pages in (("alpha.pdf", PAPER_ALPHA), ("beta.pdf", PAPER_BETA)):
        ids[name] = ingest_pdf(name, make_pdf(pages), settings, store, embedder).document_id
    return ids
