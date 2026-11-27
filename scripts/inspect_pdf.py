"""Inspect how a PDF is extracted, cleaned and chunked, before trusting it in the index.

  python scripts/inspect_pdf.py data/papers/paper.pdf                 # per-page summary + chunk stats
  python scripts/inspect_pdf.py data/papers/paper.pdf --page 3        # full cleaned text of page 3
  python scripts/inspect_pdf.py data/papers/paper.pdf --grep "LoRA"   # which pages mention a phrase
  python scripts/inspect_pdf.py data/papers/*.pdf --grep "rank"        # search several files

Use --grep to find page numbers when writing evaluation questions.
"""

import argparse
import glob
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.ingestion import clean_pages, extract_pages, make_chunks  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdfs", nargs="+")
    ap.add_argument("--page", type=int)
    ap.add_argument("--grep")
    ap.add_argument("--raw", action="store_true", help="show text before cleaning")
    args = ap.parse_args()
    s = get_settings()
    files = [f for p in args.pdfs for f in (glob.glob(p) or [p])]

    for path in files:
        data = Path(path).read_bytes()
        raw = extract_pages(data)
        pages = raw if args.raw else clean_pages(raw, s.drop_references)
        name = Path(path).name
        if args.grep:
            needle = args.grep.lower()
            for i, text in enumerate(pages, 1):
                low = text.lower()
                if needle in low:
                    pos = low.find(needle)
                    snippet = text[max(0, pos - 80): pos + 120].replace("\n", " ")
                    print(f"{name} page {i}: ...{snippet}...")
            continue
        if args.page:
            print(f"===== {name} page {args.page} =====")
            print(pages[args.page - 1])
            continue
        chunks = make_chunks(pages, "inspect", name, "", s.chunk_size, s.chunk_overlap)
        print(f"===== {name}: {len(raw)} pages, {len(chunks)} chunks "
              f"(size={s.chunk_size}, overlap={s.chunk_overlap}) =====")
        for i, (r, c) in enumerate(zip(raw, pages), 1):
            preview = c[:90].replace("\n", " ")
            print(f"  p{i:>3}: raw {len(r):>5} chars, clean {len(c):>5} | {preview}")
        if chunks:
            lens = [len(c.text) for c in chunks]
            print(f"  chunk length: min {min(lens)}, median {statistics.median(lens):.0f}, max {max(lens)}")
        empty = [i for i, p in enumerate(raw, 1) if not p.strip()]
        if empty:
            print(f"  WARNING: pages with no text (scanned/images?): {empty}")


if __name__ == "__main__":
    main()
