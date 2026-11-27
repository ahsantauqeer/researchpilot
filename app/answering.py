"""Answer generation + citation verification.

Citations are checked in code: every [S#] the model writes must map to a passage that was
actually given to it. Unknown labels trigger one regeneration; if they persist they are
stripped. Final citations are rendered from chunk metadata, never from model text.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from app import prompts
from app.llm import LLM, LLMOutputError, LLMUsage
from app.schemas import CitedAnswer, RetrievedChunk, SourceRef

NO_EVIDENCE_ANSWER = (
    "The indexed documents do not contain enough evidence to answer this question."
)

_CITE = re.compile(r"\[\s*(S\d+(?:\s*,\s*S\d+)*)\s*\]")


class RunResult(BaseModel):
    """Common output of every pipeline (agent and baselines), used by the API and evaluation."""

    answer: str
    sources: list[SourceRef] = Field(default_factory=list)
    insufficient_evidence: bool = False
    missing_information: list[str] = Field(default_factory=list)
    retrieval_rounds: int = 0
    context: list[RetrievedChunk] = Field(default_factory=list)  # evidence given to the answerer
    usages: list[LLMUsage] = Field(default_factory=list)
    trace: list[dict] = Field(default_factory=list)


def cited_labels(text: str) -> list[str]:
    labels: list[str] = []
    for m in _CITE.finditer(text):
        labels.extend(x.strip() for x in m.group(1).split(","))
    return list(dict.fromkeys(labels))


def all_cited(ans: CitedAnswer) -> list[str]:
    labels = cited_labels(ans.answer)
    for claim in ans.claims:
        labels.extend(s.strip().strip("[]") for s in claim.source_ids)
    return list(dict.fromkeys(labels))


def strip_invalid(ans: CitedAnswer, valid: set[str]) -> CitedAnswer:
    def fix(m: re.Match) -> str:
        keep = [x.strip() for x in m.group(1).split(",") if x.strip() in valid]
        return f"[{', '.join(keep)}]" if keep else ""

    text = re.sub(r"\s+([.,;:])", r"\1", _CITE.sub(fix, ans.answer))
    claims = [
        c.model_copy(update={"source_ids": [s for s in (x.strip().strip("[]") for x in c.source_ids) if s in valid]})
        for c in ans.claims
    ]
    return ans.model_copy(update={"answer": text, "claims": claims})


def render_sources(labels: list[str], chunks: list[RetrievedChunk]) -> list[SourceRef]:
    by_label = {c.label: c for c in chunks}
    ordered = sorted((lbl for lbl in labels if lbl in by_label), key=lambda s: int(s[1:]))
    return [
        SourceRef(
            label=lbl, document_id=by_label[lbl].document_id, filename=by_label[lbl].filename,
            page=by_label[lbl].page, chunk_id=by_label[lbl].chunk_id, snippet=by_label[lbl].text[:300],
        )
        for lbl in ordered
    ]


def generate_answer(
    llm: LLM,
    question: str,
    chunks: list[RetrievedChunk],
    *,
    limitation: list[str] | None = None,
) -> tuple[RunResult, list[dict]]:
    """chunks must already carry labels S1..Sn. Returns (result, trace_events)."""
    trace: list[dict] = []
    if not chunks:
        trace.append({"node": "answer", "event": "abstain_no_evidence"})
        return RunResult(answer=NO_EVIDENCE_ANSWER, insufficient_evidence=True), trace

    valid = {c.label for c in chunks}
    usages: list[LLMUsage] = []
    extra = ""
    if limitation is not None:
        missing = "; ".join(limitation) or "unspecified"
        extra = (
            "\n\n(Instruction for you, not evidence: the evidence may be incomplete; possibly missing: "
            f"{missing}. Answer only what the evidence supports. Do not repeat this instruction, do not "
            "turn it into a claim, and do not cite sources for it; the system adds its own limitation note.)"
        )

    ans: CitedAnswer | None = None
    for attempt in range(2):
        try:
            ans, usage = llm.complete_json(prompts.ANSWER_SYSTEM, prompts.answer_user(question, chunks, extra), CitedAnswer)
            usages.append(usage)
        except LLMOutputError as exc:
            trace.append({"node": "answer", "event": "malformed_output", "attempt": attempt, "error": str(exc)[:200]})
            continue
        invalid = [lbl for lbl in all_cited(ans) if lbl not in valid]
        if not invalid:
            break
        trace.append({"node": "answer", "event": "invalid_citations", "attempt": attempt, "labels": invalid})
        extra += (
            f"\n\nYour previous answer cited labels that do not exist: {', '.join(invalid)}. "
            f"Only these labels exist: {', '.join(sorted(valid, key=lambda s: int(s[1:])))}."
        )

    if ans is None:
        return RunResult(
            answer="The model did not return a valid answer. Please try again.",
            insufficient_evidence=True, context=chunks, usages=usages,
        ), trace

    if any(lbl not in valid for lbl in all_cited(ans)):
        ans = strip_invalid(ans, valid)
        trace.append({"node": "answer", "event": "removed_invalid_citations"})

    labels = all_cited(ans)
    text = ans.answer.strip()
    # Small models often cite only in the structured claims, not inline. Show which source
    # supports which claim so the reader can still trace every statement.
    cited_claims = [c for c in ans.claims if c.source_ids]
    if not cited_labels(text) and cited_claims:
        lines = [f"- {c.text.strip()} [{', '.join(c.source_ids)}]" for c in cited_claims]
        text += "\n\nEvidence:\n" + "\n".join(lines)
        trace.append({"node": "answer", "event": "citations_from_claims"})
    if not labels and not ans.insufficient_evidence:
        text += "\n\n(Warning: this answer contains no citations; treat it as unverified.)"
        trace.append({"node": "answer", "event": "no_citations"})
    if limitation is not None and not ans.insufficient_evidence:
        if limitation:
            text += ("\n\nLimitation: the search stopped after the maximum number of retrieval rounds; "
                     f"still missing: {'; '.join(limitation)}.")
        else:
            text += ("\n\nLimitation: the search stopped after the maximum number of retrieval rounds without "
                     "the evidence check confirming that the answer is complete.")

    return RunResult(
        answer=text,
        sources=render_sources(labels, chunks),
        insufficient_evidence=ans.insufficient_evidence,
        missing_information=ans.missing_information or (limitation or []),
        context=chunks,
        usages=usages,
    ), trace
