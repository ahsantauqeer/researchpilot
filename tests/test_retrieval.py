from app.retrieval import RetrievalConfig, Retriever, dedupe, label_evidence, rrf_fuse, tokenize
from app.schemas import RetrievedChunk


def test_tokenize_keeps_identifiers():
    assert tokenize("GPT-4 beats bert_base on v1.2") == ["gpt-4", "beats", "bert_base", "on", "v1.2"]


def test_rrf_fuse_rewards_agreement():
    fused = dict(rrf_fuse([["a", "b", "c"], ["b", "a", "d"]]))
    assert fused["a"] == fused["b"] > fused["c"] > 0
    assert set(fused) == {"a", "b", "c", "d"}


def test_dense_search_finds_relevant_paper(indexed, store, embedder):
    r = Retriever(store, embedder, RetrievalConfig(top_k=2, hybrid=False))
    hits = r.search("low rank matrices trainable parameters")
    assert hits and hits[0].filename == "beta.pdf"


def test_document_filter(indexed, retriever):
    hits = retriever.search("memory", k=10, document_ids=[indexed["alpha.pdf"]])
    assert hits and all(h.document_id == indexed["alpha.pdf"] for h in hits)


def test_hybrid_matches_exact_term(indexed, retriever):
    hits = retriever.search("RankAdapt", k=2)
    assert hits[0].filename == "beta.pdf"


def test_per_document_retrieval_covers_each_doc(indexed, retriever):
    hits = retriever.search_per_document("memory", list(indexed.values()), k=1)
    assert {h.filename for h in hits} == {"alpha.pdf", "beta.pdf"}


def test_bm25_index_refreshes_after_delete(indexed, retriever, store):
    assert retriever.search("RankAdapt", k=2)
    store.delete_document(indexed["beta.pdf"])
    assert all(h.filename != "beta.pdf" for h in retriever.search("RankAdapt", k=5))


def test_rerank_reorders(indexed, store, embedder):
    class ReverseReranker:  # scores later candidates higher
        def score(self, query, texts):
            return list(range(len(texts)))

    base = Retriever(store, embedder, RetrievalConfig(top_k=3, hybrid=False))
    rr = Retriever(store, embedder, RetrievalConfig(top_k=3, hybrid=False, rerank=True, rerank_candidates=3),
                   reranker=ReverseReranker())
    q = "attention memory"
    assert [c.chunk_id for c in rr.search(q)] == [c.chunk_id for c in base.search(q)][::-1]


def _c(cid, score=0.0):
    return RetrievedChunk(chunk_id=cid, document_id="d", filename="f.pdf", page=1, chunk_index=0,
                          text="t", score=score)


def test_dedupe_and_labels():
    out = label_evidence(dedupe([_c("a", 0.1), _c("b"), _c("a", 0.9)]))
    assert [c.chunk_id for c in out] == ["a", "b"]
    assert out[0].score == 0.9
    assert [c.label for c in out] == ["S1", "S2"]
