"""Central configuration. All values come from environment variables or `.env`."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM provider: "openai" (any OpenAI-compatible endpoint) or "ollama" (Ollama cloud) ---
    llm_provider: Literal["openai", "ollama"] = "openai"

    # --- Ollama cloud (used when LLM_PROVIDER=ollama) ---
    ollama_api_key: str = Field(default="", description="Ollama cloud API key (Bearer token)")
    ollama_base_url: str = "https://ollama.com/v1"  # OpenAI-compatible endpoint
    ollama_model: str = "glm-5.3-flash"
    ollama_fast_model: str | None = None

    # --- LLM (any OpenAI-compatible endpoint: OpenAI, local Ollama, vLLM, LM Studio, ...) ---
    llm_api_key: str = Field(default="", description="API key for the LLM endpoint")
    llm_base_url: str | None = Field(default=None, description="Leave empty for api.openai.com")
    llm_model: str = "gpt-4o-mini"  # strong model for the final answer
    llm_fast_model: str | None = None  # cheaper model for plan/evaluate/refine; defaults to llm_model
    llm_temperature: float = 0.0
    llm_timeout_s: float = 60.0
    llm_json_mode: bool = True  # set false if your endpoint rejects response_format=json_object
    # e.g. "none" turns off hidden "thinking" on Ollama thinking models (qwen3.x) -> much faster.
    # Leave empty to use the model/provider default.
    llm_reasoning_effort: str | None = None
    judge_model: str | None = None  # evaluation only; must differ from llm_model

    # --- LangSmith tracing (optional; off by default) ---
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "researchpilot"
    langsmith_endpoint: str | None = None  # only for non-US regions, e.g. https://eu.api.smith.langchain.com

    # --- Embeddings / reranker (local, Sentence Transformers) ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Vector store ---
    qdrant_path: str = "data/qdrant"  # embedded, persistent on disk
    qdrant_url: str | None = None  # if set (e.g. http://localhost:6333), use a Qdrant server instead
    qdrant_api_key: str | None = None
    collection_name: str = "researchpilot"

    # --- Ingestion ---
    chunk_size: int = 800
    chunk_overlap: int = 150
    max_upload_mb: int = 25
    pdf_dir: str = "data/pdfs"  # copies of uploaded PDFs, served to the UI's page viewer
    drop_references: bool = True

    # --- Retrieval ---
    top_k: int = 5
    rerank_candidates: int = 20
    use_hybrid: bool = True
    use_rerank: bool = False
    max_context_chunks: int = 12

    # --- Agent ---
    max_rounds: int = 3
    max_queries: int = 3

    # Resolved values for whichever provider is active. Code should use these, not the raw fields.
    @property
    def active_api_key(self) -> str:
        return self.ollama_api_key if self.llm_provider == "ollama" else self.llm_api_key

    @property
    def active_base_url(self) -> str | None:
        return self.ollama_base_url if self.llm_provider == "ollama" else (self.llm_base_url or None)

    @property
    def active_model(self) -> str:
        return self.ollama_model if self.llm_provider == "ollama" else self.llm_model

    @property
    def fast_model(self) -> str:
        if self.llm_provider == "ollama":
            return self.ollama_fast_model or self.ollama_model
        return self.llm_fast_model or self.llm_model

    @property
    def llm_configured(self) -> bool:
        if self.llm_provider == "ollama":
            return bool(self.ollama_api_key)
        return bool(self.llm_api_key or self.llm_base_url)


@lru_cache
def get_settings() -> Settings:
    return Settings()
