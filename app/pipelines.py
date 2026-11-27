"""Non-agentic baselines used for the evaluation comparison (systems A-D)."""

from __future__ import annotations

import time

from langsmith import traceable  # no-op unless LangSmith tracing is enabled

from app import prompts
from app.answering import RunResult, generate_answer
from app.llm import LLM, LLMOutputError
from app.retrieval import Retriever, dedupe, label_evidence
from app.schemas import QueryPlan


@traceable(name="one_pass baseline", run_type="chain")
def one_pass(question: str, retriever: Retriever, llm: LLM, k: int,
             document_ids: list[str] | None = None) -> RunResult:
    """Systems A/B/C: a single retrieval with the raw question, then answer."""
    t0 = time.perf_counter()
    chunks = label_evidence(retriever.search(question, k, document_ids))
    result, events = generate_answer(llm, question, chunks)
    result.retrieval_rounds = 1
    result.trace = [{"node": "retrieve", "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}] + events
    return result


@traceable(name="multi_query baseline", run_type="chain")
def multi_query(question: str, retriever: Retriever, llm: LLM, k: int, max_queries: int,
                max_context: int, document_ids: list[str] | None = None) -> RunResult:
    """System D: LLM query expansion, one retrieval round, no evaluator loop."""
    usages = []
    try:
        plan, usage = llm.complete_json(prompts.PLAN_SYSTEM.format(max_queries=max_queries),
                                        prompts.plan_user(question, []), QueryPlan, fast=True)
        usages.append(usage)
        queries = [q for q in plan.queries if q.strip()][:max_queries] or [question]
    except LLMOutputError:
        queries = [question]
    found = []
    for q in queries:
        found.extend(retriever.search(q, k, document_ids))
    chunks = label_evidence(dedupe(found)[:max_context])
    result, events = generate_answer(llm, question, chunks)
    result.retrieval_rounds = 1
    result.usages = usages + result.usages
    result.trace = [{"node": "plan", "queries": queries}] + events
    return result
