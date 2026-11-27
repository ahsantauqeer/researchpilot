"""FastAPI service.

Run:  uvicorn app.main:app --reload
Docs: http://localhost:8000/docs

/ask is stateless: every question is answered independently (no chat memory).
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from app.agent import ResearchAgent
from app.answering import RunResult
from app.config import Settings, get_settings
from app.embeddings import Embedder
from app.ingestion import IngestionError, ingest_pdf
from app.llm import LLM
from app.retrieval import Retriever
from app.schemas import AskRequest, AskResponse, DeleteResponse, DocumentInfo, UploadResponse
from app.vector_store import VectorStore

log = logging.getLogger("researchpilot")


@dataclass
class Deps:
    settings: Settings
    embedder: Embedder
    store: VectorStore
    retriever: Retriever
    llm: LLM
    agent: ResearchAgent


def build_deps(settings: Settings | None = None) -> Deps:
    """Construct real components. Fails fast with a clear message if configuration is missing."""
    from app.embeddings import CrossEncoderReranker, SentenceTransformerEmbedder
    from app.llm import OpenAICompatibleLLM
    from app.retrieval import RetrievalConfig

    s = settings or get_settings()
    llm = OpenAICompatibleLLM(s)  # raises if LLM_API_KEY / LLM_BASE_URL are missing
    embedder = SentenceTransformerEmbedder(s.embedding_model)
    store = VectorStore.from_settings(s, embedder.dimension, embedder.model_name)
    reranker = CrossEncoderReranker(s.reranker_model) if s.use_rerank else None
    retriever = Retriever(store, embedder, RetrievalConfig(s.top_k, s.use_hybrid, s.use_rerank, s.rerank_candidates),
                          reranker)
    return Deps(s, embedder, store, retriever, llm, ResearchAgent(llm, retriever, store, s))


_DOC_ID = re.compile(r"^[0-9a-f]{16}$")  # document_id = first 16 hex chars of the file's SHA-256


def pdf_path(settings: Settings, document_id: str) -> Path:
    if not _DOC_ID.match(document_id):  # also blocks path traversal
        raise HTTPException(404, f"Unknown document_id: {document_id}")
    return Path(settings.pdf_dir) / f"{document_id}.pdf"


def to_response(result: RunResult, include_trace: bool, run_id: str | None = None,
                settings: Settings | None = None) -> AskResponse:
    from app.tracing import trace_url

    resp = _to_response(result, include_trace)
    if run_id and settings:
        resp.run_id = run_id
        resp.trace_url = trace_url(settings, run_id)
    return resp


def _to_response(result: RunResult, include_trace: bool) -> AskResponse:
    trace = None
    if include_trace:
        trace = result.trace + [{"node": "total", "llm_calls": len(result.usages),
                                 "prompt_tokens": sum(u.prompt_tokens for u in result.usages),
                                 "completion_tokens": sum(u.completion_tokens for u in result.usages)}]
    return AskResponse(answer=result.answer, sources=result.sources, retrieval_rounds=result.retrieval_rounds,
                       insufficient_evidence=result.insufficient_evidence, trace=trace)


def create_app(deps: Deps | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.deps = deps or build_deps()
        yield

    app = FastAPI(title="ResearchPilot", version="0.1.0", lifespan=lifespan,
                  description="Agentic RAG research assistant with verified citations.")

    def d(request: Request) -> Deps:
        return request.app.state.deps

    def check_document_ids(deps_: Deps, ids: list[str] | None) -> None:
        if ids:
            known = {doc.document_id for doc in deps_.store.list_documents()}
            unknown = [i for i in ids if i not in known]
            if unknown:
                raise HTTPException(404, f"Unknown document_ids: {unknown}")

    @app.exception_handler(IngestionError)
    async def ingestion_error(_: Request, exc: IngestionError):
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})

    @app.get("/")
    def root():
        return {"service": "ResearchPilot", "status": "ok", "docs": "/docs"}

    @app.get("/health")
    def health(request: Request):
        deps_ = d(request)
        checks = {"llm_provider": deps_.settings.llm_provider, "llm_model": deps_.settings.active_model,
                  "llm_configured": deps_.settings.llm_configured,
                  "tracing": bool(deps_.settings.langsmith_tracing and deps_.settings.langsmith_api_key),
                  "embedding_model": deps_.embedder.model_name}
        try:
            checks["documents"] = len(deps_.store.list_documents())
            checks["qdrant"] = "ok"
        except Exception as exc:  # noqa: BLE001
            checks["qdrant"] = f"error: {exc}"
        ok = checks["qdrant"] == "ok" and checks["llm_configured"]
        return JSONResponse(status_code=200 if ok else 503, content={"ready": ok, **checks})

    @app.post("/upload", response_model=UploadResponse)
    def upload(request: Request, file: UploadFile = File(...)):
        deps_ = d(request)
        limit = deps_.settings.max_upload_mb * 1024 * 1024
        data = file.file.read(limit + 1)  # read at most limit+1 bytes; validate_pdf rejects oversize
        result = ingest_pdf(file.filename or "", data, deps_.settings, deps_.store, deps_.embedder,
                            content_type=file.content_type)
        # Keep a copy for the page viewer (also restores it for documents indexed before this existed).
        path = pdf_path(deps_.settings, result.document_id)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return result

    @app.get("/documents/{document_id}/pdf")
    def document_pdf(request: Request, document_id: str):
        path = pdf_path(d(request).settings, document_id)
        if not path.exists():
            raise HTTPException(404, "PDF file not stored for this document. Re-upload it to enable the viewer.")
        # inline (not attachment) so the browser's viewer can open it at #page=N
        return FileResponse(path, media_type="application/pdf", content_disposition_type="inline",
                            filename=f"{document_id}.pdf")

    @app.get("/documents", response_model=list[DocumentInfo])
    def documents(request: Request):
        return d(request).store.list_documents()

    @app.delete("/documents/{document_id}", response_model=DeleteResponse)
    def delete_document(request: Request, document_id: str):
        if not d(request).store.delete_document(document_id):
            raise HTTPException(404, f"Unknown document_id: {document_id}")
        pdf_path(d(request).settings, document_id).unlink(missing_ok=True)
        return DeleteResponse(document_id=document_id, deleted=True)

    @app.post("/ask", response_model=AskResponse)
    def ask(request: Request, body: AskRequest):
        deps_ = d(request)
        check_document_ids(deps_, body.document_ids)
        run_id = str(uuid.uuid4())
        try:
            result = deps_.agent.run(body.question, body.document_ids, run_id=run_id)
        except Exception as exc:  # noqa: BLE001 - surface provider/network errors as 502
            log.exception("agent failed")
            raise HTTPException(502, f"Answer generation failed: {type(exc).__name__}: {exc}") from exc
        return to_response(result, body.include_trace, run_id, deps_.settings)

    @app.post("/ask/stream")
    def ask_stream(request: Request, body: AskRequest):
        """Server-sent events: one 'progress' event per agent step, then a 'result' event."""
        deps_ = d(request)
        check_document_ids(deps_, body.document_ids)

        run_id = str(uuid.uuid4())

        def events():
            try:
                for kind, payload in deps_.agent.stream(body.question, body.document_ids, run_id=run_id):
                    if kind == "result":
                        payload = to_response(payload, body.include_trace, run_id, deps_.settings).model_dump()
                    yield f"event: {kind}\ndata: {json.dumps(payload, default=str)}\n\n"
            except Exception as exc:  # noqa: BLE001
                log.exception("agent failed")
                yield f"event: error\ndata: {json.dumps({'detail': str(exc)})}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


# Components (models, Qdrant, LLM client) are built in the lifespan hook at server start, so
# importing this module does not load models or require an API key.
app = create_app()
