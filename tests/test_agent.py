import re

from app.agent import ResearchAgent
from app.answering import NO_EVIDENCE_ANSWER, generate_answer
from app.llm import LLMOutputError
from app.retrieval import label_evidence
from app.schemas import CitedAnswer, Claim, EvidenceVerdict, QueryPlan, RetrievedChunk
from tests.conftest import ScriptedLLM


def first_label(user: str) -> str:
    return re.search(r"\[(S\d+)\]", user).group(1)


def plan(queries=("attention memory",), comparison=False):
    return lambda s, u: QueryPlan(queries=list(queries), is_comparison=comparison)


def cited_answer(s, u):
    lbl = first_label(u)
    return CitedAnswer(answer=f"TileAttn reduces memory [{lbl}].", claims=[Claim(text="x", source_ids=[lbl])])


def verdict(sufficient):
    return lambda s, u: EvidenceVerdict(sufficient=sufficient, missing_information=[] if sufficient else ["speed"],
                                        confidence="high")


def make_agent(llm, retriever, store, settings):
    return ResearchAgent(llm, retriever, store, settings)


def test_sufficient_evidence_answers_in_one_round(indexed, retriever, store, settings):
    llm = ScriptedLLM({"QueryPlan": plan(), "EvidenceVerdict": verdict(True), "CitedAnswer": cited_answer})
    result = make_agent(llm, retriever, store, settings).run("How does TileAttn save memory?")
    assert result.retrieval_rounds == 1
    assert result.sources and all(s.chunk_id for s in result.sources)
    assert llm.count("QueryPlan") == 1  # plan only, no refine
    nodes = [e["node"] for e in result.trace]
    assert nodes[:3] == ["plan", "retrieve", "evaluate"] and nodes[-1] == "answer"


def test_insufficient_retries_then_stops_at_max_rounds(indexed, retriever, store, settings):
    llm = ScriptedLLM({"QueryPlan": plan(), "EvidenceVerdict": verdict(False), "CitedAnswer": cited_answer})
    result = make_agent(llm, retriever, store, settings).run("Everything about TileAttn?")
    assert result.retrieval_rounds == settings.max_rounds == 3
    assert llm.count("EvidenceVerdict") == 3
    assert llm.count("QueryPlan") == 1 + 2  # plan + two refines
    assert "Limitation" in result.answer and "speed" in result.answer


def test_max_rounds_one_means_no_retry(indexed, retriever, store, settings):
    llm = ScriptedLLM({"QueryPlan": plan(), "EvidenceVerdict": verdict(False), "CitedAnswer": cited_answer})
    result = make_agent(llm, retriever, store, settings).run("q", max_rounds=1)
    assert result.retrieval_rounds == 1


def test_retry_then_sufficient(indexed, retriever, store, settings):
    verdicts = iter([False, True])
    llm = ScriptedLLM({"QueryPlan": plan(), "CitedAnswer": cited_answer,
                       "EvidenceVerdict": lambda s, u: verdict(next(verdicts))(s, u)})
    result = make_agent(llm, retriever, store, settings).run("TileAttn speed and memory?")
    assert result.retrieval_rounds == 2
    assert "Limitation" not in result.answer


def test_empty_index_abstains_without_answer_call(store, retriever, settings):
    llm = ScriptedLLM({"QueryPlan": plan(), "EvidenceVerdict": verdict(True), "CitedAnswer": cited_answer})
    result = make_agent(llm, retriever, store, settings).run("What is TileAttn?")
    assert result.answer == NO_EVIDENCE_ANSWER and result.insufficient_evidence
    assert result.sources == []
    assert llm.count("EvidenceVerdict") == 0 and llm.count("CitedAnswer") == 0
    assert result.retrieval_rounds == settings.max_rounds


def test_malformed_evaluator_counts_as_insufficient(indexed, retriever, store, settings):
    def broken(s, u):
        raise LLMOutputError("not json")

    llm = ScriptedLLM({"QueryPlan": plan(), "EvidenceVerdict": broken, "CitedAnswer": cited_answer})
    result = make_agent(llm, retriever, store, settings).run("q")
    assert result.retrieval_rounds == settings.max_rounds
    assert any(e.get("malformed") for e in result.trace if e["node"] == "evaluate")


def test_malformed_plan_falls_back_to_question(indexed, retriever, store, settings):
    def broken(s, u):
        raise LLMOutputError("bad")

    llm = ScriptedLLM({"QueryPlan": broken, "EvidenceVerdict": verdict(True), "CitedAnswer": cited_answer})
    result = make_agent(llm, retriever, store, settings).run("How does RankAdapt work?")
    assert result.trace[0]["queries"] == ["How does RankAdapt work?"]
    assert result.sources


def test_comparison_uses_per_document_retrieval(indexed, retriever, store, settings):
    llm = ScriptedLLM({"QueryPlan": plan(["memory usage"], comparison=True), "EvidenceVerdict": verdict(True),
                       "CitedAnswer": cited_answer})
    agent = make_agent(llm, retriever, store, settings)
    agent.run("Compare memory use of the two papers")
    evidence = [c for c in llm.calls if c["schema"] == "EvidenceVerdict"][0]["user"]
    assert "alpha.pdf" in evidence and "beta.pdf" in evidence


# --------------------------------------------------------------- citation verification
def _chunks(n=2):
    return label_evidence([RetrievedChunk(chunk_id=f"c{i}", document_id="d", filename="f.pdf", page=i + 1,
                                          chunk_index=0, text=f"text {i}") for i in range(n)])


def test_invalid_citation_triggers_regeneration():
    answers = iter([
        CitedAnswer(answer="Claim [S9].", claims=[Claim(text="c", source_ids=["S9"])]),
        CitedAnswer(answer="Claim [S2].", claims=[Claim(text="c", source_ids=["S2"])]),
    ])
    llm = ScriptedLLM({"CitedAnswer": lambda s, u: next(answers)})
    result, events = generate_answer(llm, "q", _chunks())
    assert llm.count("CitedAnswer") == 2
    assert [s.label for s in result.sources] == ["S2"] and result.sources[0].page == 2
    assert any(e["event"] == "invalid_citations" for e in events)


def test_persistent_invalid_citations_are_stripped():
    llm = ScriptedLLM({"CitedAnswer": lambda s, u: CitedAnswer(answer="A [S1, S7]. B [S8].",
                                                               claims=[Claim(text="a", source_ids=["S7"])])})
    result, _ = generate_answer(llm, "q", _chunks())
    assert "S7" not in result.answer and "S8" not in result.answer and "[S1]" in result.answer
    assert [s.label for s in result.sources] == ["S1"]


def test_claim_only_citations_are_shown_in_answer():
    llm = ScriptedLLM({"CitedAnswer": lambda s, u: CitedAnswer(
        answer="Prose without markers.", claims=[Claim(text="Fact one", source_ids=["S2"])])})
    result, events = generate_answer(llm, "q", _chunks())
    assert "- Fact one [S2]" in result.answer
    assert [s.label for s in result.sources] == ["S2"]
    assert any(e["event"] == "citations_from_claims" for e in events)


def test_sources_render_from_metadata_not_model_text():
    llm = ScriptedLLM({"CitedAnswer": lambda s, u: CitedAnswer(answer="See fake.pdf page 99 [S1].")})
    result, _ = generate_answer(llm, "q", _chunks())
    assert result.sources[0].filename == "f.pdf" and result.sources[0].page == 1
