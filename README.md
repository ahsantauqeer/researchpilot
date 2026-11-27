# ResearchPilot — Agentic RAG Research Assistant

Ask questions across your PDFs. ResearchPilot plans focused searches, retrieves passages, judges
whether the evidence is enough, refines its search within a hard round limit, and answers with
citations (filename + page) that are **verified in code** — or says plainly that the documents do
not contain the answer.

**Stack:** Python 3.11 · FastAPI · LangGraph · Qdrant (embedded) · Sentence Transformers · BM25 ·
cross-encoder reranker · any OpenAI-compatible LLM · Streamlit

> Status: implemented and unit-tested. **No evaluation results are claimed yet** — see
> [Results](#results).

## How it works

```
INGEST  PDF -> validate (ext, %PDF magic, size) -> extract per page (pypdf) -> drop headers/footers/references
            -> chunk (800 chars / 150 overlap) -> embed -> upsert to Qdrant (deterministic IDs, no duplicates)

ASK     plan (<=3 queries) -> retrieve (dense [+BM25 RRF] [+rerank], per-document for comparisons)
            -> evaluate (schema-validated verdict) --insufficient & rounds left--> refine -> retrieve ...
            -> answer (only from labelled evidence [S1].., citations checked in code)
```

| Guarantee | How |
|---|---|
| Bounded loop | `attempts` increments in retrieve; router stops at `MAX_ROUNDS`, then answers with an explicit limitation |
| No fabricated citations | every `[S#]` must map to a passage given to the model; unknown labels → one regeneration → stripped. Filenames/pages come from metadata, never model text |
| Abstention | empty retrieval → "not enough evidence" without calling the answer model |
| Malformed LLM output | evaluator → treated as insufficient; planner → falls back to the raw question |
| Prompt injection | document text only inside escaped `<evidence>` blocks in the user message; agent has no tools; injection PDF in tests |

`/ask` is **stateless** — each question is independent (no chat memory, by design for the MVP).

## Setup (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt          # or requirements.lock.txt for exact versions
copy .env.example .env                    # then edit .env: LLM_API_KEY, LLM_MODEL
pytest -q                                 # offline tests, no API key needed
```

## Run

```powershell
uvicorn app.main:app --reload             # API on http://localhost:8000  (docs: /docs)
streamlit run ui/streamlit_app.py         # UI on http://localhost:8501   (second terminal)
```

API: `GET /health` · `POST /upload` (multipart PDF) · `GET /documents` · `DELETE /documents/{id}` ·
`POST /ask` `{"question": "...", "document_ids": [...], "include_trace": true}` · `POST /ask/stream` (SSE).

## Evaluate

1. Put 5-10 papers in `data/papers/`.
2. Check extraction: `python scripts/inspect_pdf.py data/papers/*.pdf`
3. Write 30-50 questions in `evaluation/questions.jsonl` (format: `evaluation/questions.example.jsonl`).
   Find page numbers with `python scripts/inspect_pdf.py data/papers/*.pdf --grep "phrase"`.
   Include comparison, multi-evidence and **unanswerable** questions; mark some `"split": "test"` and never tune on them.
4. Run:
   ```powershell
   python -m evaluation.run_evaluation --systems A,E --limit 3        # smoke test
   python -m evaluation.run_evaluation --split dev                   # all systems A-F
   python -m evaluation.run_evaluation --split test --judge --price-in 0.15 --price-out 0.6
   ```
   Output goes to `evaluation/results/<timestamp>/` (`summary.md`, `summary.json`, `per_question.jsonl`).

| ID | System | Question it answers |
|---|---|---|
| A | one-pass dense | baseline |
| B | one-pass hybrid (BM25 + RRF) | does keyword search help? |
| C | one-pass + rerank | does reranking help? |
| D | multi-query, no evaluator | is the loop needed, or just more queries? |
| E | agentic (dense) | does evidence-aware retrying help? |
| F | agentic + hybrid + rerank | best combination, at what cost? |

Metrics: recall@5, context recall, MRR, citation precision (cited pages ∈ gold pages), correct
abstention / false refusal, latency, LLM calls, tokens, cost; with `--judge` (a different model):
faithfulness, answer relevance, correctness. All with bootstrap 95% CIs. LLM calls are cached
(temperature 0), so re-runs give identical numbers.

## Results

_TBD — paste `summary.md` here after running the evaluation. Report negative results too._

## Layout

```
app/        config, schemas, llm, ingestion, embeddings, vector_store, retrieval, prompts,
            answering (citation verification), agent (LangGraph), pipelines (baselines), main (FastAPI)
tests/      ingestion, retrieval, agent, api, security (fake LLM + fake embedder, offline)
evaluation/ run_evaluation.py, questions.example.jsonl
scripts/    inspect_pdf.py
ui/         streamlit_app.py
```

## Limitations / next steps

Text PDFs only (no OCR) · in-process BM25 suits hundreds of papers, not millions (move to Qdrant
sparse vectors) · no auth/multi-tenancy · no chat memory · optional claim-verification pass and
section-aware chunking not yet implemented.
