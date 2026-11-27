"""Send one request to the configured LLM through the app's own client (app/llm.py).

  python scripts/test_ollama.py

Uses whatever .env selects (LLM_PROVIDER=ollama -> Ollama cloud). Prints the provider, model
and the model's reply. Never prints the API key.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import openai  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.llm import OpenAICompatibleLLM  # noqa: E402

HINTS = {
    401: "key is wrong or missing (check OLLAMA_API_KEY in .env)",
    402: "the account needs usage credits",
    404: "model name is wrong (see https://ollama.com/api/tags)",
    429: "usage limit reached",
}


class Reply(BaseModel):
    reply: str


def main() -> None:
    s = get_settings()
    print(f"provider={s.llm_provider} base_url={s.active_base_url} model={s.active_model}")
    llm = OpenAICompatibleLLM(s)
    try:
        out, usage = llm.complete_json(
            "You are a helpful assistant.", "In one sentence, what is retrieval-augmented generation?", Reply
        )
    except openai.APIStatusError as exc:
        print(f"HTTP {exc.status_code}: {HINTS.get(exc.status_code, '')}\n{exc.message}")
        raise SystemExit(1) from exc
    print(f"reply: {out.reply}")
    print(f"tokens: {usage.prompt_tokens} in / {usage.completion_tokens} out, {usage.latency_ms:.0f} ms")


if __name__ == "__main__":
    main()
