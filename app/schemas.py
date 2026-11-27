"""Pydantic models: data passed between components, LLM output schemas, and the API contract."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


# ---------- internal data ----------
class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    page: int  # 1-based
    chunk_index: int
    text: str
    section: str | None = None
    content_hash: str = ""


class RetrievedChunk(Chunk):
    score: float = 0.0
    label: str = ""  # "S1", "S2", ... assigned when evidence is packaged


# ---------- LLM output schemas ----------
class QueryPlan(BaseModel):
    queries: list[str] = Field(default_factory=list)
    is_comparison: bool = False


class EvidenceVerdict(BaseModel):
    sufficient: bool = False
    missing_information: list[str] = Field(default_factory=list)
    relevant_source_ids: list[str] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"] = "low"


class Claim(BaseModel):
    text: str
    source_ids: list[str] = Field(default_factory=list)


class CitedAnswer(BaseModel):
    answer: str
    claims: list[Claim] = Field(default_factory=list)
    insufficient_evidence: bool = False
    missing_information: list[str] = Field(default_factory=list)


# ---------- API contract ----------
class SourceRef(BaseModel):
    label: str
    document_id: str
    filename: str
    page: int
    chunk_id: str
    snippet: str = ""


class AskRequest(BaseModel):
    question: str
    document_ids: list[str] | None = None
    include_trace: bool = False

    @field_validator("question")
    @classmethod
    def not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be empty")
        if len(v) > 4000:
            raise ValueError("question is too long (max 4000 characters)")
        return v


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceRef]
    retrieval_rounds: int
    insufficient_evidence: bool = False
    trace: list[dict] | None = None
    run_id: str | None = None
    trace_url: str | None = None  # LangSmith link, when tracing is enabled


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    pages_processed: int
    chunks_indexed: int
    duplicate: bool = False


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    chunks: int
    pages: int


class DeleteResponse(BaseModel):
    document_id: str
    deleted: bool
