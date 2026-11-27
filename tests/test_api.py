import json
import re

import pytest
from fastapi.testclient import TestClient

from app.agent import ResearchAgent
from app.main import Deps, create_app
from app.schemas import CitedAnswer, EvidenceVerdict, QueryPlan
from tests.conftest import PAPER_ALPHA, ScriptedLLM, make_pdf


@pytest.fixture
def client(settings, embedder, store, retriever):
    def answer(s, u):
        lbl = re.search(r"\[(S\d+)\]", u).group(1)
        return CitedAnswer(answer=f"Answer [{lbl}].")

    llm = ScriptedLLM({"QueryPlan": lambda s, u: QueryPlan(queries=["attention memory"]),
                       "EvidenceVerdict": lambda s, u: EvidenceVerdict(sufficient=True, confidence="high"),
                       "CitedAnswer": answer})
    deps = Deps(settings, embedder, store, retriever, llm, ResearchAgent(llm, retriever, store, settings))
    with TestClient(create_app(deps)) as c:
        yield c


def upload(client, name="alpha.pdf", data=None, ctype="application/pdf"):
    return client.post("/upload", files={"file": (name, data or make_pdf(PAPER_ALPHA), ctype)})


def test_root_and_health(client):
    assert client.get("/").json()["status"] == "ok"
    h = client.get("/health")
    assert h.status_code == 200 and h.json()["ready"]


def test_upload_list_ask_delete(client):
    r = upload(client)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["pages_processed"] == 2 and body["chunks_indexed"] > 0 and not body["duplicate"]
    assert upload(client).json()["duplicate"]

    docs = client.get("/documents").json()
    assert [d["document_id"] for d in docs] == [body["document_id"]]

    ans = client.post("/ask", json={"question": "How does TileAttn save memory?", "include_trace": True})
    assert ans.status_code == 200, ans.text
    a = ans.json()
    assert a["sources"][0]["filename"] == "alpha.pdf" and a["sources"][0]["page"] >= 1
    assert a["retrieval_rounds"] == 1 and a["trace"][-1]["node"] == "total"

    assert client.delete(f"/documents/{body['document_id']}").json()["deleted"]
    assert client.get("/documents").json() == []
    assert client.delete(f"/documents/{body['document_id']}").status_code == 404


def test_pdf_is_served_for_viewer_and_removed_on_delete(client):
    data = make_pdf(PAPER_ALPHA)
    doc_id = upload(client, data=data).json()["document_id"]
    r = client.get(f"/documents/{doc_id}/pdf")
    assert r.status_code == 200 and r.content == data
    assert r.headers["content-type"] == "application/pdf"
    assert r.headers["content-disposition"].startswith("inline")
    client.delete(f"/documents/{doc_id}")
    assert client.get(f"/documents/{doc_id}/pdf").status_code == 404


def test_pdf_endpoint_rejects_bad_ids(client):
    assert client.get("/documents/..%2F..%2Fsecret/pdf").status_code == 404
    assert client.get("/documents/not-a-real-id/pdf").status_code == 404


def test_ask_without_tracing_has_no_trace_url(client):
    upload(client)
    a = client.post("/ask", json={"question": "TileAttn memory?"}).json()
    assert a["run_id"] and a["trace_url"] is None


def test_upload_errors(client):
    assert upload(client, name="x.txt").status_code == 415
    assert upload(client, data=b"not a pdf at all").status_code == 415
    assert upload(client, data=make_pdf([""])).status_code == 422


def test_ask_errors(client):
    assert client.post("/ask", json={"question": "   "}).status_code == 422
    assert client.post("/ask", json={}).status_code == 422
    assert client.post("/ask", json={"question": "q", "document_ids": ["nope"]}).status_code == 404


def test_stream(client):
    upload(client)
    with client.stream("POST", "/ask/stream", json={"question": "TileAttn memory?"}) as r:
        text = "".join(r.iter_text())
    events = re.findall(r"event: (\w+)\ndata: (.*)\n", text)
    kinds = [k for k, _ in events]
    assert kinds[0] == "progress" and kinds[-1] == "result"
    assert json.loads(events[-1][1])["sources"]
