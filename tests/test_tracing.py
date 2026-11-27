"""LangSmith tracing must be opt-in and must never break the app."""

import os

import pytest

from app.config import Settings
from app.tracing import configure_tracing, wrap_llm_client

LS_VARS = ("LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT", "LANGSMITH_ENDPOINT")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in LS_VARS:
        monkeypatch.delenv(var, raising=False)


def test_off_by_default():
    assert configure_tracing(Settings(_env_file=None)) is False
    assert "LANGSMITH_TRACING" not in os.environ


def test_enabled_without_key_stays_off():
    assert configure_tracing(Settings(_env_file=None, langsmith_tracing=True)) is False
    assert "LANGSMITH_TRACING" not in os.environ


def test_enabled_with_key_exports_env(monkeypatch):
    s = Settings(_env_file=None, langsmith_tracing=True, langsmith_api_key="lsv2-test", langsmith_project="p1")
    assert configure_tracing(s) is True
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_PROJECT"] == "p1"
    assert "LANGSMITH_ENDPOINT" not in os.environ


def test_wrap_keeps_client_usable():
    from openai import OpenAI

    wrapped = wrap_llm_client(OpenAI(api_key="x", base_url="http://127.0.0.1:9"), "openai")
    assert hasattr(wrapped.chat.completions, "create")


def test_wrap_falls_back_on_error():
    # Not an OpenAI client: whatever langsmith does with it, our wrapper must not raise.
    assert wrap_llm_client(object(), "openai") is not None
