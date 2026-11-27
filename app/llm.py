"""Thin LLM interface.

Everything else in the app talks to `LLM.complete_json`, so the provider can be swapped
(OpenAI, Ollama, vLLM, ...) and tests can inject a fake. Outputs are validated against a
Pydantic schema; malformed output raises `LLMOutputError` and callers decide what to do.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from app.config import Settings, get_settings

T = TypeVar("T", bound=BaseModel)


class LLMOutputError(Exception):
    """The model returned something that is not valid JSON for the requested schema."""


class LLMUsage(BaseModel):
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cached: bool = False


class LLM(Protocol):
    def complete_json(
        self, system: str, user: str, schema: type[T], *, fast: bool = False
    ) -> tuple[T, LLMUsage]: ...


def _extract_json(text: str) -> str:
    """Accept raw JSON, or JSON wrapped in ```json fences / surrounding prose."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            text = text[start : end + 1]
    return text


def parse_json_as(raw: str, schema: type[T]) -> T:
    try:
        return schema.model_validate(json.loads(_extract_json(raw)))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise LLMOutputError(f"invalid {schema.__name__} output: {exc}") from exc


class _Cache:
    """Tiny sqlite cache so evaluation runs are reproducible and cheap to re-run."""

    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")
        self._lock = threading.Lock()

    def get(self, key: str) -> str | None:
        with self._lock:
            row = self._db.execute("SELECT v FROM c WHERE k=?", (key,)).fetchone()
        return row[0] if row else None

    def put(self, key: str, value: str) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO c VALUES (?, ?)", (key, value))
            self._db.commit()


class OpenAICompatibleLLM:
    def __init__(self, settings: Settings | None = None, cache_path: str | None = None):
        from openai import OpenAI

        self.s = settings or get_settings()
        if self.s.llm_provider == "ollama":
            if not self.s.ollama_api_key:
                raise RuntimeError(
                    "LLM_PROVIDER=ollama but OLLAMA_API_KEY is not set. Add it to .env "
                    "(create a key at https://ollama.com/settings/keys)."
                )
        elif not self.s.llm_api_key and not self.s.llm_base_url:
            raise RuntimeError(
                "LLM_API_KEY is not set. Copy .env.example to .env and fill it in "
                "(or set LLM_PROVIDER=ollama with OLLAMA_API_KEY, or LLM_BASE_URL for a local server)."
            )
        # The key is always passed explicitly, so the SDK never falls back to OPENAI_API_KEY
        # and sends it to a different provider.
        self.client = OpenAI(
            api_key=self.s.active_api_key or "not-needed",
            base_url=self.s.active_base_url,
            timeout=self.s.llm_timeout_s,
        )
        from app.tracing import configure_tracing, wrap_llm_client

        self.tracing = configure_tracing(self.s)
        if self.tracing:
            self.client = wrap_llm_client(self.client, self.s.llm_provider)
        self.cache = _Cache(cache_path) if cache_path else None

    def complete_json(self, system, user, schema, *, fast=False):
        model = self.s.fast_model if fast else self.s.active_model
        system = (
            system
            + "\n\nRespond with a single JSON object matching this JSON schema, and nothing else:\n"
            + json.dumps(schema.model_json_schema())
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        key = hashlib.sha256(
            json.dumps([model, self.s.llm_temperature, self.s.llm_reasoning_effort, messages], sort_keys=True).encode()
        ).hexdigest()

        if self.cache and (hit := self.cache.get(key)):
            data = json.loads(hit)
            usage = LLMUsage(**{**data["usage"], "cached": True})
            return parse_json_as(data["content"], schema), usage

        kwargs = {"model": model, "messages": messages, "temperature": self.s.llm_temperature}
        if self.s.llm_json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self.s.llm_reasoning_effort:
            kwargs["reasoning_effort"] = self.s.llm_reasoning_effort
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(**kwargs)
        latency = (time.perf_counter() - t0) * 1000
        content = resp.choices[0].message.content or ""
        usage = LLMUsage(
            model=model,
            prompt_tokens=getattr(resp.usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(resp.usage, "completion_tokens", 0) or 0,
            latency_ms=latency,
        )
        parsed = parse_json_as(content, schema)  # raises before caching bad output
        if self.cache:
            self.cache.put(key, json.dumps({"content": content, "usage": usage.model_dump()}))
        return parsed, usage
