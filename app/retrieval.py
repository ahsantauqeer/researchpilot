"""Retrieval: dense search, optional BM25 hybrid (RRF fusion), optional cross-encoder rerank.

Each improvement is a switch so the evaluation harness can measure them one at a time.

The BM25 index is built in-process over all chunks and rebuilt whenever the store changes.
That is simple and transparent for a personal corpus (hundreds of papers). For a large corpus,
move to Qdrant's native sparse vectors instead.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass

from app.embeddings import Embedder, Reranker
from app.schemas import Chunk, RetrievedChunk
from app.vector_store import VectorStore

_TOKEN = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    # Keeps identifiers like "gpt-4", "bert_base", "v1.2" as single tokens.
    return _TOKEN.findall(text.lower())


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal rank fusion: score(d) = sum over lists of 1 / (k + rank)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class BM25Index:
    def __init__(self, store: VectorStore):
        self.store = store
        self._version = -1
        self._lock = threading.Lock()
        self._chunks: list[Chunk] = []
        self._bm25 = None

    def _refresh(self) -> None:
        if self._version == self.store.version and self._bm25 is not None:
            return
        from rank_bm25 import BM25Okapi

        self._chunks = list(self.store.iter_chunks())
        corpus = [tokenize(c.text) for c in self._chunks] or [[""]]
        self._bm25 = BM25Okapi(corpus)
        self._version = self.store.version

    def search(self, query: str, k: int, document_ids: list[str] | None = None) -> list[RetrievedChunk]:
        with self._lock:
            self._refresh()
            if not self._chunks:
                return []
            q = tokenize(query)
            if not q:
                return []
            scores = self._bm25.get_scores(q)
            allowed = set(document_ids) if document_ids else None
            ranked = sorted(
                (i for i, c in enumerate(self._chunks) if allowed is None or c.document_id in allowed),
                key=lambda i: scores[i],
                reverse=True,
            )
            return [
                RetrievedChunk(**self._chunks[i].model_dump(), score=float(scores[i]))
                for i in ranked[:k]
                if scores[i] > 0
            ]


@dataclass
class RetrievalConfig:
    top_k: int = 5
    hybrid: bool = False
    rerank: bool = False
    rerank_candidates: int = 20


class Retriever:
    def __init__(self, store: VectorStore, embedder: Embedder, config: RetrievalConfig,
                 reranker: Reranker | None = None, bm25: BM25Index | None = None):
        if config.rerank and reranker is None:
            raise ValueError("rerank=True requires a reranker")
        self.store = store
        self.embedder = embedder
        self.config = config
        self.reranker = reranker
        self.bm25 = bm25 or (BM25Index(store) if config.hybrid else None)

    def search(self, query: str, k: int | None = None, document_ids: list[str] | None = None) -> list[RetrievedChunk]:
        k = k or self.config.top_k
        n = max(k, self.config.rerank_candidates) if self.config.rerank else k
        qv = self.embedder.embed([query])[0]
        dense = self.store.dense_search(qv, n, document_ids)

        if self.config.hybrid and self.bm25 is not None:
            sparse = self.bm25.search(query, n, document_ids)
            by_id = {c.chunk_id: c for c in dense + sparse}
            fused = rrf_fuse([[c.chunk_id for c in dense], [c.chunk_id for c in sparse]])[:n]
            candidates = [by_id[cid].model_copy(update={"score": s}) for cid, s in fused]
        else:
            candidates = dense

        if self.config.rerank and self.reranker and candidates:
            scores = self.reranker.score(query, [c.text for c in candidates])
            candidates = [
                c.model_copy(update={"score": s})
                for c, s in sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
            ]
        return candidates[:k]

    def search_per_document(self, query: str, document_ids: list[str], k: int | None = None) -> list[RetrievedChunk]:
        """Top-k inside each document, so one paper cannot crowd out the others."""
        out: list[RetrievedChunk] = []
        for doc_id in document_ids:
            out.extend(self.search(query, k, [doc_id]))
        return out


def dedupe(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Remove repeated chunks (same chunk_id), keeping first position and the best score."""
    seen: dict[str, RetrievedChunk] = {}
    for c in chunks:
        if c.chunk_id not in seen:
            seen[c.chunk_id] = c
        elif c.score > seen[c.chunk_id].score:
            seen[c.chunk_id] = seen[c.chunk_id].model_copy(update={"score": c.score})
    return list(seen.values())


def label_evidence(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    return [c.model_copy(update={"label": f"S{i}"}) for i, c in enumerate(chunks, start=1)]
