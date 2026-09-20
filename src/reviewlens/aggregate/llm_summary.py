"""Optional LLM executive summary — turn the aspect table into prose.

The aggregation stage already knows *what* customers love and hate; this module
asks an LLM to write the "so what": a short executive summary with
recommendations, grounded in a compact stats digest built from the pipeline
output (never the raw reviews — the prompt stays small and the numbers stay
authoritative).

Providers, resolved from the environment (see ``.env.example``; a project-root
``.env`` file is read if present, without overriding real env vars):

* **Ollama** (default) — ``OLLAMA_HOST`` (default ``http://localhost:11434``),
  native ``/api/chat`` endpoint, model from ``LLM_MODEL`` (default ``llama3.1``).
* **OpenAI-compatible API** — set ``LLM_API_BASE`` (e.g. ``https://.../v1``),
  ``LLM_API_KEY`` and ``LLM_MODEL``; uses ``/chat/completions``.

Everything network-facing goes through an injectable ``transport`` callable so
tests exercise prompt building and response parsing without a live server.
Stdlib HTTP only — no new dependency for an optional step.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from reviewlens.aggregate.summary import representative_quotes, top_hated, top_loved
from reviewlens.config import PROJECT_ROOT, load_config

Transport = Callable[[str, dict, dict[str, str]], dict]

_SETUP_HINT = (
    "No LLM provider configured. Either run Ollama locally (https://ollama.com, "
    "`ollama pull llama3.1`) or set LLM_API_BASE / LLM_API_KEY / LLM_MODEL for an "
    "OpenAI-compatible endpoint — see .env.example."
)

SYSTEM_PROMPT = (
    "You are a product analyst. You receive aggregated aspect-based sentiment "
    "statistics computed from customer reviews. Write an executive summary in "
    "markdown with exactly these sections: '## Overview' (2-3 sentences), "
    "'## What customers love', '## What customers hate' (bullet points citing "
    "the net scores and mention counts you were given), and "
    "'## Recommendations' (3 concrete, prioritized actions). Ground every "
    "claim in the provided numbers and quotes; do not invent statistics."
)


def _load_dotenv(path=None) -> None:
    """Load ``KEY=VALUE`` lines from ``.env`` into ``os.environ`` (no override)."""
    path = path or (PROJECT_ROOT / ".env")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, OSError):
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def build_digest(
    aspects: pd.DataFrame,
    group_col: str = "theme",
    k: int = 8,
    min_mentions: int = 2,
    quotes_per_group: int = 1,
) -> str:
    """Compact plain-text stats block the LLM is asked to summarize.

    Pure and deterministic — this is the tested part; the model only ever sees
    numbers that the aggregation stage actually computed.
    """
    if aspects.empty:
        return "No aspect mentions were extracted."

    n_reviews = aspects["review_id"].nunique()
    mixed = int(aspects.groupby("review_id")["aspect_sentiment"].nunique().gt(1).sum())
    loved = top_loved(aspects, group_col=group_col, k=k, min_mentions=min_mentions)
    hated = top_hated(aspects, group_col=group_col, k=k, min_mentions=min_mentions)
    quotes = representative_quotes(aspects, group_col=group_col, per_side=quotes_per_group)

    def _table(df: pd.DataFrame) -> list[str]:
        return [
            f"- {r[group_col]}: net {r['net_score']:+.2f} "
            f"({int(r['positive'])} pos / {int(r['neutral'])} neu / "
            f"{int(r['negative'])} neg, {int(r['mentions'])} mentions)"
            for _, r in df.iterrows()
        ]

    lines = [
        f"Reviews analyzed: {n_reviews}",
        f"Aspect mentions: {len(aspects)}",
        f"Reviews with mixed per-aspect sentiment: {mixed}",
        "",
        f"Top loved (by {group_col}, net score in [-1, +1]):",
        *(_table(loved) or ["- (none above the mention threshold)"]),
        "",
        f"Top hated (by {group_col}):",
        *(_table(hated) or ["- (none above the mention threshold)"]),
    ]

    featured = set(loved[group_col]) | set(hated[group_col])
    quotes = quotes[quotes[group_col].isin(featured)]
    if not quotes.empty:
        lines += ["", "Representative quotes:"]
        for _, q in quotes.iterrows():
            lines.append(f'- [{q[group_col]} / {q["side"]}] "{q["sentence"]}"')

    return "\n".join(lines)


@dataclass
class ProviderConfig:
    """A resolved chat endpoint: where to POST and how to read the reply."""

    kind: str  # "ollama" | "openai"
    url: str
    model: str
    headers: dict[str, str] = field(default_factory=dict)

    def payload(self, system: str, user: str) -> dict:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if self.kind == "ollama":
            return {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.2},
            }
        return {"model": self.model, "messages": messages, "temperature": 0.2}

    def extract(self, response: dict) -> str:
        try:
            if self.kind == "ollama":
                return str(response["message"]["content"]).strip()
            return str(response["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected response from {self.kind} endpoint: {exc}") from exc


def resolve_provider(env: dict[str, str] | None = None) -> ProviderConfig:
    """Pick the provider from the environment (loading ``.env`` first).

    ``LLM_API_BASE`` wins (hosted OpenAI-compatible endpoint); otherwise Ollama
    at ``OLLAMA_HOST`` / localhost.
    """
    if env is None:
        _load_dotenv()
        env = dict(os.environ)

    api_base = env.get("LLM_API_BASE", "").rstrip("/")
    if api_base:
        model = env.get("LLM_MODEL", "")
        if not model:
            raise RuntimeError("LLM_API_BASE is set but LLM_MODEL is not — see .env.example.")
        headers = {}
        if env.get("LLM_API_KEY"):
            headers["Authorization"] = f"Bearer {env['LLM_API_KEY']}"
        return ProviderConfig("openai", f"{api_base}/chat/completions", model, headers)

    host = env.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    return ProviderConfig("ollama", f"{host}/api/chat", env.get("LLM_MODEL", "llama3.1"))


def _http_transport(url: str, payload: dict, headers: dict[str, str]) -> dict:
    """POST JSON, return parsed JSON. Friendly error when nothing is listening."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"LLM endpoint returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach LLM endpoint at {url} ({exc.reason}). {_SETUP_HINT}"
        ) from exc


def generate_summary(
    aspects: pd.DataFrame,
    group_col: str = "theme",
    config: dict[str, Any] | None = None,
    transport: Transport | None = None,
) -> str:
    """Digest -> prompt -> LLM -> markdown executive summary."""
    cfg = (config or load_config())["summary"]
    digest = build_digest(
        aspects,
        group_col=group_col,
        k=cfg.get("top_k_aspects", 8),
        min_mentions=cfg.get("min_mentions", 2),
        quotes_per_group=cfg.get("quotes_per_aspect", 2),
    )
    provider = resolve_provider()
    user_prompt = (
        "Aspect-based sentiment statistics for a set of product reviews:\n\n"
        f"{digest}\n\nWrite the executive summary."
    )
    send = transport or _http_transport
    response = send(provider.url, provider.payload(SYSTEM_PROMPT, user_prompt), provider.headers)
    return provider.extract(response)
