"""Qdrant wrapper. Embedded (on-disk) mode by default; set QDRANT_URL to use a server."""

from __future__ import annotations

import threading
import uuid
from collections import defaultdict

from qdrant_client import QdrantClient, models

from app.schemas import Chunk, DocumentInfo, RetrievedChunk


class EmbeddingModelMismatch(RuntimeError):
    pass


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def _doc_filter(document_ids: list[str] | None) -> models.Filter | None:
    if not document_ids:
        return None
    return models.Filter(must=[models.FieldCondition(key="document_id", match=models.MatchAny(any=document_ids))])


class VectorStore:
    def __init__(self, client: QdrantClient, collection: str, dimension: int, embedding_model: str):
        self.client = client
        self.collection = collection
        self.dimension = dimension
        self.embedding_model = embedding_model
        self._lock = threading.RLock()  # embedded Qdrant is not safe for concurrent writers
        self.version = 0  # bumped on every write; lets the BM25 index know when to rebuild
        self._ensure_collection()

    @classmethod
    def from_settings(cls, settings, dimension: int, embedding_model: str, path: str | None = None):
        if settings.qdrant_url:
            client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        elif (path or settings.qdrant_path) == ":memory:":
            client = QdrantClient(location=":memory:")
        else:
            client = QdrantClient(path=path or settings.qdrant_path)
        return cls(client, settings.collection_name, dimension, embedding_model)

    # ------------------------------------------------------------ setup
    def _ensure_collection(self) -> None:
        with self._lock:
            if not self.client.collection_exists(self.collection):
                self.client.create_collection(
                    self.collection,
                    vectors_config=models.VectorParams(size=self.dimension, distance=models.Distance.COSINE),
                )
                try:  # no-op (with a warning) in embedded mode; matters on a real server
                    self.client.create_payload_index(
                        self.collection, field_name="document_id", field_schema=models.PayloadSchemaType.KEYWORD
                    )
                except Exception:  # noqa: BLE001
                    pass
                return
            info = self.client.get_collection(self.collection)
            existing = info.config.params.vectors
            size = existing.size if hasattr(existing, "size") else None
            if size is not None and size != self.dimension:
                raise EmbeddingModelMismatch(
                    f"Collection '{self.collection}' has vectors of size {size} but the embedding model "
                    f"produces {self.dimension}. Use a new COLLECTION_NAME or delete the old index."
                )
            sample, _ = self.client.scroll(self.collection, limit=1, with_payload=["embedding_model"])
            if sample and sample[0].payload.get("embedding_model") not in (None, self.embedding_model):
                raise EmbeddingModelMismatch(
                    f"Collection '{self.collection}' was built with "
                    f"{sample[0].payload['embedding_model']!r}, but EMBEDDING_MODEL is "
                    f"{self.embedding_model!r}. Never mix embedding models in one collection."
                )

    # ------------------------------------------------------------ writes
    def upsert_chunks(self, chunks: list[Chunk], vectors: list[list[float]], batch_size: int = 128) -> None:
        assert len(chunks) == len(vectors)
        with self._lock:
            for i in range(0, len(chunks), batch_size):
                points = [
                    models.PointStruct(
                        id=point_id(c.chunk_id),
                        vector=v,
                        payload={**c.model_dump(), "embedding_model": self.embedding_model},
                    )
                    for c, v in zip(chunks[i : i + batch_size], vectors[i : i + batch_size])
                ]
                self.client.upsert(self.collection, points=points, wait=True)
            self.version += 1

    def delete_document(self, document_id: str) -> bool:
        with self._lock:
            if not self.document_exists(document_id):
                return False
            self.client.delete(
                self.collection,
                points_selector=models.FilterSelector(filter=_doc_filter([document_id])),
                wait=True,
            )
            self.version += 1
            return True

    # ------------------------------------------------------------ reads
    def document_exists(self, document_id: str) -> bool:
        with self._lock:
            pts, _ = self.client.scroll(self.collection, scroll_filter=_doc_filter([document_id]), limit=1)
            return bool(pts)

    def iter_chunks(self, document_ids: list[str] | None = None):
        with self._lock:
            offset = None
            while True:
                pts, offset = self.client.scroll(
                    self.collection, scroll_filter=_doc_filter(document_ids), limit=256,
                    offset=offset, with_payload=True, with_vectors=False,
                )
                for p in pts:
                    yield Chunk.model_validate(p.payload)
                if offset is None:
                    break

    def list_documents(self) -> list[DocumentInfo]:
        agg: dict[str, dict] = defaultdict(lambda: {"filename": "", "chunks": 0, "pages": set()})
        for c in self.iter_chunks():
            d = agg[c.document_id]
            d["filename"] = c.filename
            d["chunks"] += 1
            d["pages"].add(c.page)
        return sorted(
            (DocumentInfo(document_id=k, filename=v["filename"], chunks=v["chunks"], pages=max(v["pages"]))
             for k, v in agg.items()),
            key=lambda d: d.filename.lower(),
        )

    def dense_search(self, vector: list[float], k: int, document_ids: list[str] | None = None) -> list[RetrievedChunk]:
        with self._lock:
            res = self.client.query_points(
                self.collection, query=vector, limit=k, query_filter=_doc_filter(document_ids), with_payload=True
            )
        return [RetrievedChunk(**p.payload, score=float(p.score)) for p in res.points]
