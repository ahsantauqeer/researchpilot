"""The bounded agent loop (LangGraph): plan -> retrieve -> evaluate -> (refine -> retrieve)* -> answer.

`attempts` is incremented in retrieve; the router checks it before every refine, so the
number of retrieval rounds can never exceed `max_rounds`.
"""

from __future__ import annotations

import operator
import time
from collections.abc import Iterator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app import prompts
from app.answering import RunResult, generate_answer
from app.config import Settings
from app.llm import LLM, LLMOutputError, LLMUsage
from app.retrieval import Retriever, dedupe, label_evidence
from app.schemas import EvidenceVerdict, QueryPlan, RetrievedChunk
from app.vector_store import VectorStore

MAX_COMPARISON_DOCS = 5


class AgentState(TypedDict, total=False):
    question: str
    document_ids: list[str] | None
    queries: list[str]
    all_queries: list[str]
    is_comparison: bool
    documents: list[RetrievedChunk]
    relevant_labels: list[str]
    attempts: int
    max_rounds: int
    sufficient: bool
    missing_information: list[str]
    confidence: str
    result: RunResult
    usages: Annotated[list[LLMUsage], operator.add]
    trace_events: Annotated[list[dict], operator.add]


def _clean_queries(queries: list[str], limit: int, exclude: list[str] | None = None) -> list[str]:
    seen = {q.strip().lower() for q in (exclude or [])}
    out = []
    for q in queries:
        q = " ".join(str(q).split())
        if q and q.lower() not in seen:
            seen.add(q.lower())
            out.append(q[:300])
    return out[:limit]


def _event(node: str, t0: float, usages: list[LLMUsage] | None = None, **details: Any) -> dict:
    tokens = sum(u.prompt_tokens + u.completion_tokens for u in usages or [])
    return {"node": node, "latency_ms": round((time.perf_counter() - t0) * 1000, 1), "tokens": tokens, **details}


class ResearchAgent:
    def __init__(self, llm: LLM, retriever: Retriever, store: VectorStore, settings: Settings):
        self.llm = llm
        self.retriever = retriever
        self.store = store
        self.s = settings
        self.graph = self._build()

    # ------------------------------------------------------------------ nodes
    def plan(self, state: AgentState) -> dict:
        t0 = time.perf_counter()
        docs = self.store.list_documents()
        if state.get("document_ids"):
            docs = [d for d in docs if d.document_id in state["document_ids"]]
        system = prompts.PLAN_SYSTEM.format(max_queries=self.s.max_queries)
        usages: list[LLMUsage] = []
        try:
            plan, usage = self.llm.complete_json(system, prompts.plan_user(state["question"], [d.filename for d in docs]),
                                                 QueryPlan, fast=True)
            usages.append(usage)
            queries = _clean_queries(plan.queries, self.s.max_queries)
            is_comparison = plan.is_comparison
            fallback = False
        except LLMOutputError:
            queries, is_comparison, fallback = [], False, True
        if not queries:
            queries = [state["question"]]
        return {
            "queries": queries, "all_queries": list(queries), "is_comparison": is_comparison,
            "documents": [], "relevant_labels": [], "attempts": 0, "usages": usages,
            "trace_events": [_event("plan", t0, usages, queries=queries, is_comparison=is_comparison,
                                    fallback=fallback)],
        }

    def _comparison_targets(self, state: AgentState) -> list[str]:
        if state.get("document_ids"):
            return state["document_ids"][:MAX_COMPARISON_DOCS]
        return [d.document_id for d in self.store.list_documents()][:MAX_COMPARISON_DOCS]

    def retrieve(self, state: AgentState) -> dict:
        t0 = time.perf_counter()
        doc_ids = state.get("document_ids")
        targets = self._comparison_targets(state) if state.get("is_comparison") else []
        new: list[RetrievedChunk] = []
        for q in state["queries"]:
            if len(targets) > 1:
                per_doc_k = max(2, self.s.top_k // 2)
                new.extend(self.retriever.search_per_document(q, targets, per_doc_k))
            else:
                new.extend(self.retriever.search(q, self.s.top_k, doc_ids))

        existing = state.get("documents", [])
        existing_ids = {c.chunk_id for c in existing}
        fresh = [c for c in dedupe(new) if c.chunk_id not in existing_ids]
        relevant = set(state.get("relevant_labels", []))
        # When over budget, keep evidence the evaluator marked relevant, then the new evidence.
        ordered = ([c for c in existing if c.label in relevant] + fresh
                   + [c for c in existing if c.label not in relevant])
        kept_ids = {c.chunk_id for c in ordered[: self.s.max_context_chunks]}
        merged = [c for c in existing + fresh if c.chunk_id in kept_ids]
        documents = label_evidence(merged)
        attempts = state.get("attempts", 0) + 1
        return {
            "documents": documents, "attempts": attempts,
            "trace_events": [_event("retrieve", t0, round=attempts, queries=state["queries"],
                                    new_chunks=len(fresh), total_chunks=len(documents),
                                    per_document=len(targets) > 1)],
        }

    def evaluate(self, state: AgentState) -> dict:
        t0 = time.perf_counter()
        docs = state.get("documents", [])
        if not docs:
            return {"sufficient": False, "missing_information": [state["question"]], "confidence": "low",
                    "relevant_labels": [], "trace_events": [_event("evaluate", t0, event="empty_retrieval")]}
        usages: list[LLMUsage] = []
        try:
            verdict, usage = self.llm.complete_json(prompts.EVALUATE_SYSTEM,
                                                    prompts.evaluate_user(state["question"], docs),
                                                    EvidenceVerdict, fast=True)
            usages.append(usage)
            malformed = False
        except LLMOutputError:
            # Invalid evaluator output is treated as "insufficient", never as success.
            verdict = EvidenceVerdict(sufficient=False, missing_information=[], confidence="low")
            malformed = True
        valid = {c.label for c in docs}
        relevant = [lbl for lbl in verdict.relevant_source_ids if lbl in valid]
        return {
            "sufficient": verdict.sufficient, "missing_information": verdict.missing_information,
            "confidence": verdict.confidence, "relevant_labels": relevant, "usages": usages,
            "trace_events": [_event("evaluate", t0, usages, sufficient=verdict.sufficient,
                                    confidence=verdict.confidence, missing=verdict.missing_information,
                                    malformed=malformed)],
        }

    def route_after_evaluate(self, state: AgentState) -> str:
        if state.get("sufficient"):
            return "answer"
        if state.get("attempts", 0) < state.get("max_rounds", self.s.max_rounds):
            return "refine"
        return "answer"

    def refine(self, state: AgentState) -> dict:
        t0 = time.perf_counter()
        previous = state.get("all_queries", [])
        missing = state.get("missing_information") or []
        system = prompts.REFINE_SYSTEM.format(max_queries=self.s.max_queries)
        usages: list[LLMUsage] = []
        try:
            plan, usage = self.llm.complete_json(system, prompts.refine_user(state["question"], missing, previous),
                                                 QueryPlan, fast=True)
            usages.append(usage)
            queries = _clean_queries(plan.queries, self.s.max_queries, exclude=previous)
        except LLMOutputError:
            queries = []
        if not queries:  # fall back to searching for the missing facts directly
            queries = _clean_queries(missing, self.s.max_queries, exclude=previous) or [
                f"{state['question']} details"
            ]
        return {"queries": queries, "all_queries": previous + queries, "usages": usages,
                "trace_events": [_event("refine", t0, usages, queries=queries)]}

    def answer(self, state: AgentState) -> dict:
        t0 = time.perf_counter()
        limitation = None if state.get("sufficient") else (state.get("missing_information") or [])
        result, events = generate_answer(self.llm, state["question"], state.get("documents", []),
                                         limitation=limitation)
        result.retrieval_rounds = state.get("attempts", 0)
        return {"result": result, "usages": result.usages,
                "trace_events": events + [_event("answer", t0, result.usages, sources=len(result.sources),
                                                 insufficient=result.insufficient_evidence)]}

    # ------------------------------------------------------------------ graph
    def _build(self):
        g = StateGraph(AgentState)
        g.add_node("plan", self.plan)
        g.add_node("retrieve", self.retrieve)
        g.add_node("evaluate", self.evaluate)
        g.add_node("refine", self.refine)
        g.add_node("answer", self.answer)
        g.add_edge(START, "plan")
        g.add_edge("plan", "retrieve")
        g.add_edge("retrieve", "evaluate")
        g.add_conditional_edges("evaluate", self.route_after_evaluate, {"answer": "answer", "refine": "refine"})
        g.add_edge("refine", "retrieve")
        g.add_edge("answer", END)
        return g.compile()

    def _initial(self, question: str, document_ids: list[str] | None, max_rounds: int | None) -> AgentState:
        return {"question": question, "document_ids": document_ids or None,
                "max_rounds": max(1, max_rounds or self.s.max_rounds), "usages": [], "trace_events": []}

    def _config(self, tags: list[str] | None = None, run_id: str | None = None) -> dict:
        # recursion_limit is a second safety net on top of max_rounds.
        # run_name / tags / metadata / run_id only label the run in LangSmith (ignored when tracing is off).
        extra = {"run_id": run_id} if run_id else {}
        return {
            **extra,
            "recursion_limit": 50,
            "run_name": "ResearchPilot agent",
            "tags": ["agent", *(tags or [])],
            "metadata": {"llm_model": self.s.active_model, "fast_model": self.s.fast_model,
                         "max_rounds": self.s.max_rounds, "top_k": self.s.top_k,
                         "hybrid": self.retriever.config.hybrid, "rerank": self.retriever.config.rerank},
        }

    def run(self, question: str, document_ids: list[str] | None = None, max_rounds: int | None = None,
            tags: list[str] | None = None, run_id: str | None = None) -> RunResult:
        final = self.graph.invoke(self._initial(question, document_ids, max_rounds), self._config(tags, run_id))
        result: RunResult = final["result"]
        result.usages = final.get("usages", [])
        result.trace = final.get("trace_events", [])
        return result

    def stream(self, question: str, document_ids: list[str] | None = None,
               run_id: str | None = None) -> Iterator[tuple[str, Any]]:
        """Yields ("progress", event) per node, then ("result", RunResult)."""
        usages: list[LLMUsage] = []
        trace: list[dict] = []
        result: RunResult | None = None
        for update in self.graph.stream(self._initial(question, document_ids, None),
                                        self._config(["stream"], run_id), stream_mode="updates"):
            for _node, delta in update.items():
                usages += delta.get("usages", [])
                for ev in delta.get("trace_events", []):
                    trace.append(ev)
                    yield "progress", ev
                if "result" in delta:
                    result = delta["result"]
        assert result is not None
        result.usages, result.trace = usages, trace
        yield "result", result
