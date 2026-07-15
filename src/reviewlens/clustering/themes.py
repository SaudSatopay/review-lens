"""Group raw aspect terms into higher-level themes.

Baseline ("normalized") strategy: normalize each term (lowercase + naive
singularization) and map it to a theme via keyword matching. This turns
``display``, ``screens`` and ``screen`` into one **screen** theme without any
model download, giving the dashboard clean buckets on day one.

The upgrade path (``method: kmeans`` / ``hdbscan`` in config) embeds aspect
terms with all-MiniLM-L6-v2 and clusters them; :func:`embed_and_cluster` is the
lazy-imported stub for that slice.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

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
    "battery_life": ["battery life"],  # kept distinct example; matched via 'battery' anyway
    "comfort": ["fit", "comfort", "comfortable", "wear"],
}


def normalize_term(term: str) -> str:
    """Lowercase and naively singularize an aspect term.

    Crude on purpose (baseline): ``batteries`` -> ``battery``, ``screens`` ->
    ``screen``. Real morphology arrives with the embedding-based clustering slice.
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
            # 'battery_life' is illustrative; fold it into 'battery'.
            return "battery" if theme == "battery_life" else theme
    return norm if fallback_to_term else "other"


def assign_themes(terms: Iterable[str]) -> dict[str, str]:
    """Return a ``{term: theme}`` mapping for a collection of terms."""
    return {term: assign_theme(term) for term in set(terms)}


def add_theme_column(
    df: pd.DataFrame,
    aspect_col: str = "aspect",
    theme_col: str = "theme",
    config: dict[str, Any] | None = None,  # noqa: ARG001 - reserved for kmeans slice
) -> pd.DataFrame:
    """Add a ``theme`` column derived from the aspect column."""
    out = df.copy()
    if out.empty:
        out[theme_col] = pd.Series(dtype="object")
        return out
    out[theme_col] = out[aspect_col].map(assign_theme)
    return out


def embed_and_cluster(*args, **kwargs):  # noqa: D401 - stub
    """Embedding-based clustering (all-MiniLM-L6-v2 + KMeans/HDBSCAN) -- NEXT SLICE."""
    raise NotImplementedError(
        "Embedding-based clustering is the next slice. Baseline grouping is in "
        "assign_theme / add_theme_column."
    )
