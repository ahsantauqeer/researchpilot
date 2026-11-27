"""ResearchPilot UI (Streamlit). Talks to the FastAPI backend over HTTP.

Run (with the API already running):  streamlit run ui/streamlit_app.py

API_URL         where this UI server reaches the API (default http://127.0.0.1:8000)
PUBLIC_API_URL  where the *browser* reaches the API, for the PDF viewer (default: API_URL)
"""

from __future__ import annotations

import html
import json
import os
import re
import time
import uuid

import requests
import streamlit as st
import streamlit.components.v1 as components

API = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
PUBLIC_API = os.getenv("PUBLIC_API_URL", API).rstrip("/")

st.set_page_config(page_title="ResearchPilot", layout="wide", initial_sidebar_state="expanded")

# ---------------------------------------------------------------------------------------- style
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
:root {
  --bg:#07090D; --panel:#0B0F15; --panel2:#0F141C; --line:#1A2330; --line2:#243042;
  --text:#C9D1D9; --hi:#E6EDF3; --muted:#6B7785; --cyan:#00E5FF; --mag:#FF2E88;
  --mono:'JetBrains Mono', ui-monospace, Consolas, monospace; --sans:'Inter', system-ui, sans-serif;
}
html, body, .stApp, .stMarkdown, button, input, textarea { font-family: var(--sans); }
.stApp {
  background:
    radial-gradient(900px 500px at 85% -10%, rgba(0,229,255,0.07), transparent 60%),
    linear-gradient(rgba(0,229,255,0.025) 1px, transparent 1px) 0 0 / 44px 44px,
    linear-gradient(90deg, rgba(0,229,255,0.025) 1px, transparent 1px) 0 0 / 44px 44px,
    var(--bg);
}
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.2rem; max-width: 1500px; }
section[data-testid="stSidebar"] { background: #080B10; border-right: 1px solid var(--line); }

/* buttons */
.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {
  border-radius: 2px; border: 1px solid var(--line2); background: transparent; color: var(--text);
  font-family: var(--mono); font-size: 11.5px; letter-spacing: .1em; text-transform: uppercase;
  transition: all .15s ease;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
  border-color: var(--cyan); color: var(--cyan);
  box-shadow: 0 0 0 1px rgba(0,229,255,.18), 0 0 22px rgba(0,229,255,.10);
}
button[kind="primary"], button[data-testid="stBaseButton-primary"],
button[data-testid="stBaseButton-primaryFormSubmit"] {
  background: var(--cyan) !important; color: #00161B !important; border-color: var(--cyan) !important; font-weight: 600;
}
button[kind="primary"]:hover, button[data-testid="stBaseButton-primaryFormSubmit"]:hover {
  box-shadow: 0 0 0 1px rgba(0,229,255,.5), 0 0 26px rgba(0,229,255,.35) !important; color: #00161B !important;
}
button[data-testid="stBaseButton-tertiary"] { border: none !important; padding: 0 !important; color: var(--muted) !important; }
button[data-testid="stBaseButton-tertiary"]:hover { color: var(--cyan) !important; box-shadow: none !important; }

/* inputs */
textarea, input, div[data-baseweb="select"] > div, div[data-baseweb="textarea"], div[data-baseweb="input"] {
  border-radius: 2px !important; background: var(--panel) !important;
}
div[data-baseweb="textarea"]:focus-within, div[data-baseweb="input"]:focus-within { border-color: var(--cyan) !important; }
[data-testid="stFileUploaderDropzone"] { background: var(--panel); border: 1px dashed var(--line2); border-radius: 2px; }
[data-testid="stForm"] { border: 1px solid var(--line); border-radius: 2px; background: rgba(11,15,21,.75); }
[data-testid="stExpander"] details { border: 1px solid var(--line); border-radius: 2px; background: rgba(11,15,21,.6); }
[data-testid="stStatusWidget"], [data-testid="stExpander"] summary { font-family: var(--mono); font-size: 12px; }
iframe { border: 1px solid var(--line) !important; border-radius: 2px; background: #0B0F15; }

/* custom elements */
.rp-eyebrow { font-family: var(--mono); font-size: 11px; letter-spacing: .22em; text-transform: uppercase; color: var(--cyan); }
.rp-title { font-size: 30px; font-weight: 600; color: var(--hi); letter-spacing: -.015em; margin: 2px 0 6px; }
.rp-sub { color: var(--muted); font-size: 13.5px; }
.rp-rule { height: 1px; margin: 14px 0 18px; background: linear-gradient(90deg, var(--cyan), rgba(0,229,255,.15) 35%, transparent 70%); }
.rp-chips { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }
.rp-chip { font-family: var(--mono); font-size: 10.5px; letter-spacing: .08em; text-transform: uppercase;
  color: var(--muted); border: 1px solid var(--line); padding: 3px 8px; border-radius: 2px; }
.rp-chip b { color: var(--text); font-weight: 500; }
.rp-chip.on { border-color: rgba(0,229,255,.35); }
.rp-chip.on b { color: var(--cyan); }
.rp-chip.off b { color: var(--mag); }
.rp-label { font-family: var(--mono); font-size: 10.5px; letter-spacing: .2em; text-transform: uppercase;
  color: var(--muted); margin: 18px 0 8px; display: flex; align-items: center; gap: 10px; }
.rp-label::after { content: ""; flex: 1; height: 1px; background: var(--line); }
.rp-label span { color: var(--cyan); }
.rp-q { font-size: 17px; color: var(--hi); font-weight: 500; line-height: 1.45; }
.rp-meta { font-family: var(--mono); font-size: 11px; color: var(--muted); letter-spacing: .04em; margin-top: 6px; }
.rp-meta a { color: var(--cyan); text-decoration: none; border-bottom: 1px solid rgba(0,229,255,.35); }
.rp-meta .sep { color: var(--line2); margin: 0 8px; }
.rp-answer { font-size: 15px; line-height: 1.75; color: var(--text); }
.rp-answer p { margin: 0 0 12px; }
.rp-answer ul { margin: 0 0 12px 0; padding-left: 18px; }
.rp-answer li { margin-bottom: 6px; }
.rp-answer li::marker { color: var(--cyan); }
.rp-answer .ev { font-family: var(--mono); font-size: 10.5px; letter-spacing: .2em; color: var(--muted); text-transform: uppercase; margin: 16px 0 6px; }
.rp-answer .limit { border-left: 2px solid var(--mag); padding: 6px 12px; color: #E8B4CB; background: rgba(255,46,136,.05); font-size: 13.5px; }
a.cite { font-family: var(--mono); font-size: 10.5px; color: var(--cyan); text-decoration: none; border: 1px solid rgba(0,229,255,.35);
  padding: 0 5px; margin: 0 1px; border-radius: 2px; vertical-align: 1px; background: rgba(0,229,255,.04); }
a.cite:hover { background: rgba(0,229,255,.14); box-shadow: 0 0 12px rgba(0,229,255,.25); }
.rp-warn { border: 1px solid rgba(255,46,136,.35); background: rgba(255,46,136,.05); color: #F2B6CF;
  padding: 8px 12px; font-size: 13px; border-radius: 2px; margin: 4px 0 12px; }
.rp-src { border: 1px solid var(--line); background: var(--panel); padding: 10px 12px; border-radius: 2px; scroll-margin-top: 90px; }
.rp-src.sel { border-color: rgba(0,229,255,.55); box-shadow: 0 0 0 1px rgba(0,229,255,.15), 0 0 20px rgba(0,229,255,.06); }
.rp-src .hd { font-family: var(--mono); font-size: 11px; color: var(--muted); display: flex; gap: 10px; align-items: baseline; }
.rp-src .hd .l { color: var(--cyan); }
.rp-src .hd .f { color: var(--text); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 70%; }
.rp-src .sn { color: var(--muted); font-size: 12.5px; line-height: 1.55; margin-top: 6px;
  display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
.rp-step { font-family: var(--mono); font-size: 12px; line-height: 1.6; color: var(--text); }
.rp-step .n { color: var(--line2); margin-right: 10px; }
.rp-step .k { color: var(--cyan); display: inline-block; width: 78px; letter-spacing: .08em; }
.rp-step .t { color: var(--muted); margin-right: 10px; }
.rp-step .d { color: var(--muted); display: block; padding-left: 128px; font-size: 11.5px; }
.rp-empty { border: 1px dashed var(--line2); padding: 28px 22px; color: var(--muted); font-size: 13.5px; line-height: 1.8; border-radius: 2px; }
.rp-empty b { color: var(--text); font-weight: 500; font-family: var(--mono); font-size: 11px; letter-spacing: .14em; margin-right: 10px; }
.rp-doc { font-size: 13px; color: var(--text); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.rp-doc-meta { font-family: var(--mono); font-size: 10.5px; color: var(--muted); letter-spacing: .06em; }
.rp-viewer-hd { font-family: var(--mono); font-size: 11px; color: var(--muted); margin-bottom: 8px; display: flex; justify-content: space-between; }
.rp-viewer-hd .f { color: var(--text); }
.rp-viewer-hd a { color: var(--cyan); text-decoration: none; }
"""
st.html(f"<style>{CSS}</style>")


# ---------------------------------------------------------------------------------------- state
ss = st.session_state
ss.setdefault("history", [])  # newest first: {id, question, result, elapsed, scope}
ss.setdefault("viewer", None)  # {document_id, page, filename, label, qid}
ss.setdefault("upload_key", 0)
ss.setdefault("confirm_delete", None)


def api(method: str, path: str, quiet: bool = False, **kw):
    try:
        r = requests.request(method, f"{API}{path}", timeout=kw.pop("timeout", 300), **kw)
    except requests.ConnectionError:
        if not quiet:
            st.error(f"Cannot reach the API at {API}. Start it with: uvicorn app.main:app --host 127.0.0.1 --port 8000")
        return None
    if r.status_code >= 400:
        if not quiet:
            try:
                detail = r.json().get("detail")
            except ValueError:
                detail = r.text
            st.error(f"{r.status_code}: {detail}")
        return None
    return r


def esc(s) -> str:
    return html.escape(str(s))


def label_html(text: str, accent: str = "") -> str:
    return f'<div class="rp-label">{esc(text)}{f" <span>{esc(accent)}</span>" if accent else ""}</div>'


# ---------------------------------------------------------------------------------------- answer rendering
_CITE = re.compile(r"\[(S\d+(?:\s*,\s*S\d+)*)\]")
_BULLET = re.compile(r"^\s*[-*] ")


def answer_html(text: str, qid: str, sources: dict) -> str:
    """Escape the model's text, then turn [S1, S2] into chips linking to the source cards."""
    def chips(m: re.Match) -> str:
        out = []
        for lbl in (x.strip() for x in m.group(1).split(",")):
            s = sources.get(lbl)
            tip = f"{s['filename']} - page {s['page']}" if s else lbl
            out.append(f'<a class="cite" href="#src-{qid}-{lbl}" title="{esc(tip)}">{lbl}</a>')
        return "".join(out)

    body = esc(text)
    body = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", body)
    body = _CITE.sub(chips, body)

    parts: list[str] = []
    for block in re.split(r"\n\s*\n", body.strip()):
        lines = [ln for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue
        if lines[0].strip() == "Evidence:":
            parts.append('<div class="ev">Evidence</div>')
            lines = lines[1:]
        if lines and lines[0].startswith("Limitation:"):
            parts.append(f'<p class="limit">{"<br>".join(lines)}</p>')
            continue
        bullets = [_BULLET.sub("", ln) for ln in lines if _BULLET.match(ln)]
        if bullets and len(bullets) == len(lines):
            items = "".join(f"<li>{b}</li>" for b in bullets)
            parts.append(f"<ul>{items}</ul>")
        elif lines:
            parts.append(f"<p>{'<br>'.join(lines)}</p>")
    return f'<div class="rp-answer">{"".join(parts)}</div>'


def set_viewer(src: dict, qid: str) -> None:
    ss.viewer = {**src, "qid": qid}


def render_result(item: dict, full: bool) -> None:
    r, qid = item["result"], item["id"]
    sources = {s["label"]: s for s in r["sources"]}
    meta = [f"{r['retrieval_rounds']} ROUND{'S' if r['retrieval_rounds'] != 1 else ''}",
            f"{len(r['sources'])} SOURCE{'S' if len(r['sources']) != 1 else ''}",
            f"{item['elapsed']:.0f}S", esc(item["scope"]).upper()]
    meta_html = '<span class="sep">/</span>'.join(meta)
    if r.get("trace_url"):
        meta_html += f'<span class="sep">/</span><a href="{esc(r["trace_url"])}" target="_blank">TRACE</a>'

    st.markdown(f'<div class="rp-q">{esc(item["question"])}</div><div class="rp-meta">{meta_html}</div>',
                unsafe_allow_html=True)
    if r["insufficient_evidence"]:
        st.markdown('<div class="rp-warn">The indexed documents did not contain enough evidence for a complete answer.</div>',
                    unsafe_allow_html=True)
    st.markdown(answer_html(r["answer"], qid, sources), unsafe_allow_html=True)

    if r["sources"]:
        st.markdown(label_html("Sources", f"{len(r['sources'])}"), unsafe_allow_html=True)
        v = ss.viewer
        for s in r["sources"]:
            selected = v and v.get("qid") == qid and v.get("label") == s["label"]
            c1, c2 = st.columns([6, 1], vertical_alignment="center")
            c1.markdown(
                f'<div id="src-{qid}-{s["label"]}" class="rp-src{" sel" if selected else ""}">'
                f'<div class="hd"><span class="l">{s["label"]}</span><span class="f">{esc(s["filename"])}</span>'
                f'<span>P.{s["page"]}</span></div><div class="sn">{esc(s["snippet"])}</div></div>',
                unsafe_allow_html=True)
            c2.button("View", key=f"view-{qid}-{s['label']}", on_click=set_viewer, args=(s, qid),
                      type="primary" if selected else "secondary", use_container_width=True)

    if full and r.get("trace"):
        with st.expander("Agent trace"):
            st.json(r["trace"])


# ---------------------------------------------------------------------------------------- data
health_r = api("GET", "/health", quiet=True)
health = health_r.json() if health_r is not None else None
docs_r = api("GET", "/documents", quiet=True)
docs = docs_r.json() if docs_r is not None else []
names = {d["document_id"]: d["filename"] for d in docs}

# ---------------------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown('<div class="rp-eyebrow">Index</div>', unsafe_allow_html=True)
    files = st.file_uploader("Upload text-based PDFs", type=["pdf"], accept_multiple_files=True,
                             key=f"uploader-{ss.upload_key}", label_visibility="collapsed")
    if files and st.button("Index files", type="primary", use_container_width=True):
        ok = True
        for f in files:
            with st.spinner(f"Indexing {f.name}"):
                r = api("POST", "/upload", files={"file": (f.name, f.getvalue(), "application/pdf")})
            ok = ok and r is not None
        if ok:
            ss.upload_key += 1
            st.rerun()

    st.markdown(label_html("Documents", str(len(docs))), unsafe_allow_html=True)
    if not docs:
        st.markdown('<div class="rp-doc-meta">No documents indexed.</div>', unsafe_allow_html=True)
    for d in docs:
        c1, c2 = st.columns([5, 2], vertical_alignment="center")
        c1.markdown(f'<div class="rp-doc" title="{esc(d["filename"])}">{esc(d["filename"])}</div>'
                    f'<div class="rp-doc-meta">{d["pages"]} PP / {d["chunks"]} CHUNKS</div>', unsafe_allow_html=True)
        if ss.confirm_delete == d["document_id"]:
            if c2.button("Confirm", key=f"cdel-{d['document_id']}", type="primary"):
                api("DELETE", f"/documents/{d['document_id']}")
                ss.confirm_delete = None
                if ss.viewer and ss.viewer.get("document_id") == d["document_id"]:
                    ss.viewer = None
                st.rerun()
        elif c2.button("Remove", key=f"del-{d['document_id']}", type="tertiary"):
            ss.confirm_delete = d["document_id"]
            st.rerun()

    if ss.history:
        st.markdown(label_html("Session"), unsafe_allow_html=True)
        if st.button("Clear history", type="tertiary"):
            ss.history, ss.viewer = [], None
            st.rerun()

# ---------------------------------------------------------------------------------------- header
if health:
    chips = [
        ("on", "Model", health.get("llm_model", "?")),
        ("on" if health.get("ready") else "off", "API", "online" if health.get("ready") else "not ready"),
        ("on", "Docs", str(health.get("documents", 0))),
        ("on" if health.get("tracing") else "", "Trace", "langsmith" if health.get("tracing") else "off"),
    ]
else:
    chips = [("off", "API", "offline")]
chips_html = "".join(f'<span class="rp-chip {c}">{esc(k)} <b>{esc(v)}</b></span>' for c, k, v in chips)
st.markdown(
    '<div class="rp-eyebrow">ResearchPilot // Agentic RAG</div>'
    '<div class="rp-title">Research across your documents.</div>'
    '<div class="rp-sub">Plans searches, checks whether the evidence is sufficient, and cites a file and page for every claim. '
    'Each question is answered independently.</div>'
    f'<div class="rp-chips">{chips_html}</div><div class="rp-rule"></div>',
    unsafe_allow_html=True)
if health is None:
    st.error(f"Cannot reach the API at {API}. Start it with: uvicorn app.main:app --host 127.0.0.1 --port 8000")

left, right = st.columns([3, 2], gap="large")

# ---------------------------------------------------------------------------------------- query
with left:
    st.markdown(label_html("Query"), unsafe_allow_html=True)
    with st.form("ask", border=True):
        question = st.text_area("Question", height=96, label_visibility="collapsed",
                                placeholder="Compare the approaches in these papers and recommend one for a low-memory deployment.")
        c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
        selected = c1.multiselect("Scope", options=list(names), format_func=names.get,
                                  placeholder="All documents", label_visibility="collapsed")
        show_trace = c1.checkbox("Include agent trace", value=False)
        submitted = c2.form_submit_button("Run query", type="primary", use_container_width=True,
                                          disabled=health is None)

    if submitted and question.strip():
        STEP_NAMES = {"plan": "PLAN", "retrieve": "RETRIEVE", "evaluate": "EVALUATE", "refine": "REFINE",
                      "answer": "ANSWER"}
        t0 = time.time()
        result, error, step = None, None, 0
        with st.status("Researching", expanded=True) as status:
            body = {"question": question, "document_ids": selected or None, "include_trace": show_trace}
            try:
                with requests.post(f"{API}/ask/stream", json=body, stream=True, timeout=1800) as resp:
                    if resp.status_code >= 400:
                        error = f"{resp.status_code}: {resp.text}"
                    event = None
                    for line in resp.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        if line.startswith("event: "):
                            event = line[7:]
                            continue
                        if not line.startswith("data: "):
                            continue
                        data = json.loads(line[6:])
                        if event == "result":
                            result = data
                        elif event == "error":
                            error = data.get("detail")
                        elif event == "progress" and data.get("node") in STEP_NAMES and "latency_ms" in data:
                            step += 1
                            node = data["node"]
                            detail = ""
                            if node in ("plan", "refine"):
                                detail = " | ".join(data.get("queries", []))
                            elif node == "retrieve":
                                detail = (f"round {data.get('round')}  +{data.get('new_chunks', 0)} passages  "
                                          f"({data.get('total_chunks', 0)} in context)")
                            elif node == "evaluate":
                                if data.get("sufficient"):
                                    detail = "evidence sufficient"
                                else:
                                    missing = "; ".join(data.get("missing") or [])[:220]
                                    detail = "insufficient" + (f": {missing}" if missing else "")
                            elif node == "answer":
                                detail = f"{data.get('sources', 0)} sources cited"
                            status.markdown(
                                f'<div class="rp-step"><span class="n">{step:02d}</span>'
                                f'<span class="k">{STEP_NAMES[node]}</span>'
                                f'<span class="t">{time.time() - t0:5.0f}s</span>'
                                f'<span class="d">{esc(detail)}</span></div>', unsafe_allow_html=True)
                            status.update(label=f"Researching  /  {STEP_NAMES[node].lower()} complete")
            except requests.ConnectionError:
                error = f"Cannot reach the API at {API}."
            if error:
                status.update(label="Failed", state="error")
            else:
                status.update(label=f"Complete in {time.time() - t0:.0f}s", state="complete", expanded=False)
        if error:
            st.error(error)
        elif result:
            scope = "all documents" if not selected else f"{len(selected)} document{'s' if len(selected) > 1 else ''}"
            ss.history.insert(0, {"id": uuid.uuid4().hex[:8], "question": question.strip(), "result": result,
                                  "elapsed": time.time() - t0, "scope": scope})
            first = result["sources"][0] if result["sources"] else None
            ss.viewer = {**first, "qid": ss.history[0]["id"]} if first else ss.viewer
            st.rerun()

    # ------------------------------------------------------------------------------------ results
    if not ss.history:
        st.markdown(
            '<div class="rp-empty">'
            '<div><b>01</b>Upload text-based PDFs in the index panel.</div>'
            '<div><b>02</b>Ask a question, optionally scoped to selected documents.</div>'
            '<div><b>03</b>Review the answer; every claim links to a source and page.</div>'
            '<div><b>04</b>Open any source in the viewer to check it against the original.</div>'
            '</div>', unsafe_allow_html=True)
    else:
        st.markdown(label_html("Response"), unsafe_allow_html=True)
        render_result(ss.history[0], full=True)
        if len(ss.history) > 1:
            st.markdown(label_html("History", str(len(ss.history) - 1)), unsafe_allow_html=True)
            for item in ss.history[1:]:
                with st.expander(item["question"][:110] + ("..." if len(item["question"]) > 110 else "")):
                    render_result(item, full=False)

# ---------------------------------------------------------------------------------------- viewer
with right:
    st.markdown(label_html("Source viewer"), unsafe_allow_html=True)
    v = ss.viewer
    if not v:
        st.markdown('<div class="rp-empty">Select <b>VIEW</b>on any source to open the original PDF at the cited page.</div>',
                    unsafe_allow_html=True)
    else:
        url = f"{PUBLIC_API}/documents/{v['document_id']}/pdf"
        stored = api("GET", f"/documents/{v['document_id']}/pdf", quiet=True, stream=True, timeout=30)
        st.markdown(
            f'<div class="rp-viewer-hd"><span><span style="color:var(--cyan)">{esc(v["label"])}</span>'
            f'&nbsp;&nbsp;<span class="f">{esc(v["filename"])}</span>&nbsp;&nbsp;P.{v["page"]}</span>'
            f'<a href="{esc(url)}#page={v["page"]}" target="_blank">OPEN IN TAB</a></div>', unsafe_allow_html=True)
        if stored is None:
            st.markdown('<div class="rp-warn">The original PDF is not stored for this document. '
                        'Upload the file again to enable the viewer (it will not be re-indexed).</div>',
                        unsafe_allow_html=True)
        else:
            stored.close()
            # the query string forces a reload so the viewer jumps to the new page
            components.iframe(f"{url}?v={v['qid']}-{v['label']}#page={v['page']}&view=FitH", height=860)
