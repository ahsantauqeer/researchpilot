"""End-to-end demo against a running API: upload PDFs from a folder, then ask a question.

  python scripts/demo_ask.py "Your question" [--papers data/papers] [--api http://127.0.0.1:8000]

Streams each agent step as it happens, then prints the answer and its sources.
"""

import argparse
import json
import time
from pathlib import Path

import requests


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--papers", default="data/papers")
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    args = ap.parse_args()

    for _ in range(120):  # wait up to ~2 min for models to load
        try:
            if requests.get(f"{args.api}/health", timeout=5).ok:
                break
        except requests.ConnectionError:
            pass
        time.sleep(1)
    print("health:", requests.get(f"{args.api}/health", timeout=10).json())

    for pdf in sorted(Path(args.papers).glob("*.pdf")):
        r = requests.post(f"{args.api}/upload", files={"file": (pdf.name, pdf.read_bytes(), "application/pdf")},
                          timeout=300)
        print("upload:", r.status_code, r.json())

    t0 = time.time()
    body = {"question": args.question, "include_trace": True}
    with requests.post(f"{args.api}/ask/stream", json=body, stream=True, timeout=1800) as resp:
        event = None
        for line in resp.iter_lines(decode_unicode=True):
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
                if event == "progress":
                    extra = {k: v for k, v in data.items() if k in ("queries", "sufficient", "missing", "new_chunks",
                                                                    "total_chunks", "event", "per_document")}
                    print(f"[{time.time() - t0:6.0f}s] {data['node']}: {extra}", flush=True)
                elif event == "result":
                    print(f"\n=== ANSWER ({data['retrieval_rounds']} rounds, {time.time() - t0:.0f}s, "
                          f"insufficient={data['insufficient_evidence']}) ===\n{data['answer']}\n")
                    for s in data["sources"]:
                        print(f"  [{s['label']}] {s['filename']} p.{s['page']}")
                    print("\ntotals:", data["trace"][-1])
                elif event == "error":
                    print("ERROR:", data)


if __name__ == "__main__":
    main()
