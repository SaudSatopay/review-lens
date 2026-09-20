"""Tests for the LLM executive summary — everything except a live model.

The digest builder is pure; provider resolution reads a dict; the HTTP layer is
injected. No test here touches the network.
"""

from __future__ import annotations

import pandas as pd
import pytest

from reviewlens.aggregate.llm_summary import (
    ProviderConfig,
    _load_dotenv,
    build_digest,
    generate_summary,
    resolve_provider,
)


@pytest.fixture
def aspect_rows() -> pd.DataFrame:
    """Six mentions over two themes: screen net +1.00, battery net -1.00."""
    return pd.DataFrame(
        {
            "review_id": [1, 1, 2, 2, 3, 3],
            "theme": ["screen", "battery", "screen", "battery", "screen", "battery"],
            "aspect": ["screen", "battery", "display", "battery", "screen", "charging"],
            "aspect_sentiment": [
                "positive", "negative", "positive", "negative", "positive", "negative",
            ],
            "aspect_compound": [0.9, -0.8, 0.8, -0.9, 0.7, -0.7],
            "sentence": [
                "Screen is great.", "Battery is awful.", "Love the display.",
                "Battery died fast.", "Crisp screen.", "Charging takes ages.",
            ],
        }
    )


def test_build_digest_contains_stats_and_quotes(aspect_rows):
    digest = build_digest(aspect_rows, group_col="theme", min_mentions=2)

    assert "Reviews analyzed: 3" in digest
    assert "Aspect mentions: 6" in digest
    assert "Reviews with mixed per-aspect sentiment: 3" in digest
    assert "- screen: net +1.00" in digest
    assert "- battery: net -1.00" in digest
    assert '"Screen is great."' in digest  # a representative quote made it in


def test_build_digest_respects_min_mentions(aspect_rows):
    digest = build_digest(aspect_rows, group_col="theme", min_mentions=99)
    assert "(none above the mention threshold)" in digest


def test_build_digest_empty_frame():
    assert "No aspect mentions" in build_digest(pd.DataFrame())


def test_resolve_provider_prefers_openai_compatible():
    provider = resolve_provider(
        {"LLM_API_BASE": "https://api.example.com/v1/", "LLM_API_KEY": "sk-x", "LLM_MODEL": "m1"}
    )
    assert provider.kind == "openai"
    assert provider.url == "https://api.example.com/v1/chat/completions"
    assert provider.model == "m1"
    assert provider.headers["Authorization"] == "Bearer sk-x"


def test_resolve_provider_defaults_to_local_ollama():
    provider = resolve_provider({})
    assert provider.kind == "ollama"
    assert provider.url == "http://localhost:11434/api/chat"
    assert provider.model == "llama3.1"
    assert provider.headers == {}


def test_resolve_provider_requires_model_for_hosted_api():
    with pytest.raises(RuntimeError, match="LLM_MODEL"):
        resolve_provider({"LLM_API_BASE": "https://api.example.com/v1"})


def test_payload_and_extract_shapes():
    ollama = ProviderConfig("ollama", "http://x/api/chat", "llama3.1")
    payload = ollama.payload("sys", "user")
    assert payload["stream"] is False
    assert payload["messages"][0] == {"role": "system", "content": "sys"}
    assert ollama.extract({"message": {"content": " hi "}}) == "hi"

    openai = ProviderConfig("openai", "http://x/chat/completions", "m")
    payload = openai.payload("sys", "user")
    assert "stream" not in payload
    assert openai.extract({"choices": [{"message": {"content": "ok"}}]}) == "ok"

    with pytest.raises(RuntimeError, match="Unexpected response"):
        openai.extract({"choices": []})


def test_generate_summary_uses_transport(aspect_rows, monkeypatch):
    monkeypatch.delenv("LLM_API_BASE", raising=False)
    monkeypatch.setenv("OLLAMA_HOST", "http://fake-host:1")
    calls = {}

    def fake_transport(url, payload, headers):
        calls["url"] = url
        calls["payload"] = payload
        return {"message": {"content": "## Overview\nAll good."}}

    out = generate_summary(aspect_rows, group_col="theme", transport=fake_transport)

    assert out.startswith("## Overview")
    assert calls["url"] == "http://fake-host:1/api/chat"
    user_msg = calls["payload"]["messages"][1]["content"]
    assert "- battery: net -1.00" in user_msg  # digest reached the prompt


def test_load_dotenv_parses_without_overriding(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\nLLM_MODEL='quoted-model'\nOLLAMA_HOST=http://from-file:1\n\nBADLINE\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OLLAMA_HOST", "http://already-set:2")
    monkeypatch.delenv("LLM_MODEL", raising=False)

    _load_dotenv(env_file)

    import os

    assert os.environ["LLM_MODEL"] == "quoted-model"       # quotes stripped
    assert os.environ["OLLAMA_HOST"] == "http://already-set:2"  # not overridden
