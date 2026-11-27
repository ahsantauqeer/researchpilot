"""Local embedding model and optional cross-encoder reranker (Sentence Transformers)."""

from __future__ import annotations

from typing import Protocol


class Embedder(Protocol):
    model_name: str
    dimension: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        dim = getattr(self.model, "get_embedding_dimension", None) or getattr(
            self.model, "get_sentence_embedding_dimension", None
        )
        self.dimension = int(dim() if dim else len(self.model.encode(["x"])[0]))

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vecs = self.model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        return vecs.tolist()


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        return [float(s) for s in self.model.predict([(query, t) for t in texts], show_progress_bar=False)]
