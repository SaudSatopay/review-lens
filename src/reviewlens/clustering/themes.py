"""Group raw aspect terms into higher-level themes.

Baseline ("normalized") strategy: normalize each term (lowercase + naive
singularization) and map it to a theme via keyword matching. This turns
``display``, ``screens`` and ``screen`` into one **screen** theme without any
model download, giving the dashboard clean buckets on day one.

Upgrade (``method: kmeans`` / ``hdbscan`` in config): embed the unique aspect
terms with a sentence-transformers model (default ``all-MiniLM-L6-v2``) and
cluster the embeddings — :func:`embed_and_cluster`. Themes are then
*discovered* from the data instead of enumerated in :data:`THEME_KEYWORDS`,
which is what lets the pipeline survive real review corpora whose vocabulary
nobody hand-listed. Each cluster is named after its most frequent member term,
so labels stay human-readable ("battery", not "cluster_3").
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from reviewlens.config import load_config

# Ordered theme -> keyword substrings. First matching theme wins, so put more
# specific themes before broader ones if their keywords could overlap.
THEME_KEYWORDS: dict[str, list[str]] = {
    "battery": ["battery", "charge", "charging", "power"],
    "screen": ["screen", "display", "touchscreen"],
    "camera": ["camera", "photo", "photos", "picture", "lens", "video"],
    "price": ["price", "cost", "pricing", "expensive", "cheap", "value for"],
    "sound": ["sound", "audio", "speaker", "bass", "noise cancellation", "mic", "microphone"],
    "performance": ["performance", "speed", "fast", "lag", "software", "interface", "app"],
    "delivery": ["delivery", "shipping", "packaging", "arrived", "shipment"],
    "service": ["service", "support", "warranty", "customer"],
    "build": ["build", "quality", "strap", "material", "design", "premium"],
    "comfort": ["fit", "comfort", "comfortable", "wear"],
}


def normalize_term(term: str) -> str:
    """Lowercase and naively singularize an aspect term.

    Crude on purpose (baseline): ``batteries`` -> ``battery``, ``screens`` ->
    ``screen``. Real morphology is handled by the embedding-based clustering,
    which puts inflected variants next to each other in vector space anyway.
    """
    t = str(term).lower().strip()
    if not t:
        return t
    words = t.split()
    last = words[-1]
    if last.endswith("ies") and len(last) > 3:
        last = last[:-3] + "y"
    elif last.endswith(("ses", "xes", "zes", "ches", "shes")):
        last = last[:-2]
    elif last.endswith("s") and not last.endswith("ss") and len(last) > 3:
        last = last[:-1]
    words[-1] = last
    return " ".join(words)


def assign_theme(term: str, fallback_to_term: bool = True) -> str:
    """Map an aspect term to a theme label via keyword matching.

    If no theme keyword matches, returns the normalized term itself (so unknown
    aspects still form their own single-term theme) or ``"other"``.
    """
    norm = normalize_term(term)
    for theme, keywords in THEME_KEYWORDS.items():
        if any(kw in norm for kw in keywords):
            return theme
    return norm if fallback_to_term else "other"


def assign_themes(terms: Iterable[str]) -> dict[str, str]:
    """Return a ``{term: theme}`` mapping for a collection of terms."""
    return {term: assign_theme(term) for term in set(terms)}


# --------------------------------------------------------------------------- #
# Embedding-based clustering (kmeans / hdbscan)
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=2)
def _get_embedder(model_name: str):
    """Load (and cache) a sentence-transformers model. Lazy: baseline never pays."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise ImportError(
            "Embedding-based theme clustering needs sentence-transformers. "
            'Install the extras:\n    pip install -e ".[ml]"'
        ) from exc
    return SentenceTransformer(model_name)


def _embed_terms(terms: list[str], model_name: str) -> np.ndarray:
    """Embed terms -> (n_terms, dim) array. Tests monkeypatch this function."""
    embedder = _get_embedder(model_name)
    return np.asarray(embedder.encode(terms, show_progress_bar=False))


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms == 0, 1.0, norms)


def _label_for(members: list[str], counts: Counter) -> str:
    """Human-readable cluster name: most frequent member term.

    Ties break to the shorter term, then alphabetically — deterministic across
    runs, and short frequent terms ("battery") beat long rare ones
    ("battery life expectancy").
    """
    return min(members, key=lambda t: (-counts[t], len(t), t))


def embed_and_cluster(
    terms: Iterable[str],
    method: str = "kmeans",
    n_clusters: int = 8,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    min_cluster_size: int = 2,
    random_state: int = 42,
    embed_fn: Callable[[list[str]], np.ndarray] | None = None,
) -> dict[str, str]:
    """Cluster aspect terms by embedding similarity -> ``{term: theme}``.

    ``terms`` may (and should) contain duplicates: mention frequency decides
    which member term names each cluster. Unique *normalized* terms are
    embedded, L2-normalized (so euclidean k-means/HDBSCAN behaves like cosine),
    and clustered:

    * ``kmeans``  — exactly ``min(n_clusters, n_unique)`` themes.
    * ``hdbscan`` — density-based; the cluster count is discovered, and noise
      terms keep their own normalized name (mirroring the baseline fallback).

    ``embed_fn`` overrides the sentence-transformers encoder (used in tests).
    """
    terms = [str(t) for t in terms]
    if not terms:
        return {}

    norm_of = {t: normalize_term(t) for t in set(terms)}
    counts = Counter(norm_of[t] for t in terms)  # mention-weighted
    unique = sorted(counts)  # deterministic order

    if len(unique) == 1:
        return {t: norm_of[t] for t in norm_of}

    embed = embed_fn or (lambda batch: _embed_terms(batch, embedding_model))
    vectors = _l2_normalize(np.asarray(embed(unique), dtype=np.float64))

    if method == "kmeans":
        from sklearn.cluster import KMeans

        k = max(2, min(n_clusters, len(unique)))
        labels = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit_predict(vectors)
    elif method == "hdbscan":
        from sklearn.cluster import HDBSCAN  # sklearn>=1.3 — no C-toolchain package

        size = max(2, min(min_cluster_size, len(unique)))
        labels = HDBSCAN(min_cluster_size=size).fit_predict(vectors)
    else:
        raise ValueError(f"Unknown clustering method: {method!r} (kmeans | hdbscan)")

    theme_of_norm: dict[str, str] = {}
    for cluster_id in set(labels):
        members = [u for u, lbl in zip(unique, labels, strict=True) if lbl == cluster_id]
        if cluster_id == -1:  # HDBSCAN noise: every term is its own theme
            for m in members:
                theme_of_norm[m] = m
        else:
            label = _label_for(members, counts)
            for m in members:
                theme_of_norm[m] = label

    return {t: theme_of_norm[norm_of[t]] for t in norm_of}


def add_theme_column(
    df: pd.DataFrame,
    aspect_col: str = "aspect",
    theme_col: str = "theme",
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Add a ``theme`` column derived from the aspect column.

    Dispatches on ``clustering.method``: ``normalized`` (default) maps terms
    through :func:`assign_theme`; ``kmeans`` / ``hdbscan`` run
    :func:`embed_and_cluster` over the column's terms.
    """
    out = df.copy()
    if out.empty:
        out[theme_col] = pd.Series(dtype="object")
        return out

    cfg = (config or load_config()).get("clustering", {})
    method = cfg.get("method", "normalized")

    if method in ("kmeans", "hdbscan"):
        mapping = embed_and_cluster(
            out[aspect_col].tolist(),
            method=method,
            n_clusters=cfg.get("n_clusters", 8),
            embedding_model=cfg.get(
                "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            min_cluster_size=cfg.get("min_cluster_size", 2),
        )
        out[theme_col] = out[aspect_col].map(mapping)
    else:
        out[theme_col] = out[aspect_col].map(assign_theme)
    return out
