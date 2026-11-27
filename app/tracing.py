"""Optional LangSmith tracing. Off unless LANGSMITH_TRACING=true and LANGSMITH_API_KEY are set.

When on:
- LangGraph runs (plan / retrieve / evaluate / refine / answer) are traced automatically.
- Every LLM call is traced with its prompt, output, tokens and latency (OpenAI client wrapper),
  nested under the step that made it.

Privacy: traces include the document passages sent to the LLM. Only enable this for documents
you are comfortable sending to LangSmith's servers.
"""

from __future__ import annotations

import logging
import os

from app.config import Settings

log = logging.getLogger("researchpilot")


def configure_tracing(s: Settings) -> bool:
    """Export LangSmith settings from .env to the environment (where the SDK reads them).

    Returns True if tracing is active. Safe to call more than once.
    """
    if not s.langsmith_tracing:
        return False
    if not s.langsmith_api_key:
        log.warning("LANGSMITH_TRACING=true but LANGSMITH_API_KEY is empty; tracing stays off.")
        return False
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = s.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = s.langsmith_project
    if s.langsmith_endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = s.langsmith_endpoint
    return True


_project_cache: dict[str, tuple[str, str]] = {}  # project name -> (tenant_id, project_id)


def trace_url(s: Settings, run_id: str) -> str | None:
    """Link to a run in the LangSmith UI, or None if tracing is off / the project is not known yet.

    The project is created by LangSmith on the first trace, so the very first run may get no link.
    """
    if not (s.langsmith_tracing and s.langsmith_api_key and run_id):
        return None
    try:
        if s.langsmith_project not in _project_cache:
            from langsmith import Client

            p = Client(api_key=s.langsmith_api_key, api_url=s.langsmith_endpoint or None).read_project(
                project_name=s.langsmith_project)
            _project_cache[s.langsmith_project] = (str(p.tenant_id), str(p.id))
        tenant, project = _project_cache[s.langsmith_project]
        host = "https://eu.smith.langchain.com" if "eu." in (s.langsmith_endpoint or "") else "https://smith.langchain.com"
        return f"{host}/o/{tenant}/projects/p/{project}/r/{run_id}?trace_id={run_id}"
    except Exception as exc:  # noqa: BLE001 - a missing link must never fail a request
        log.info("No LangSmith trace URL yet: %s", exc)
        return None


def wrap_llm_client(client, provider: str):
    """Wrap an OpenAI SDK client so each call is traced. Falls back to the plain client on error."""
    try:
        from langsmith.wrappers import wrap_openai

        return wrap_openai(client, chat_name=f"LLM ({provider})")
    except Exception as exc:  # noqa: BLE001 - tracing must never break the app
        log.warning("Could not enable LangSmith LLM tracing: %s", exc)
        return client
