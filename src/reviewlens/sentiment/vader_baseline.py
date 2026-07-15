"""Document-level sentiment baseline (VADER).

This is the point of comparison that motivates the whole project: a single
compound score per review collapses mixed opinions ("love the screen, hate the
battery") into one label. The per-aspect model exists precisely to recover what
this baseline throws away, so we keep it around to benchmark against.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from reviewlens.config import load_config


@lru_cache(maxsize=1)
def get_analyzer() -> SentimentIntensityAnalyzer:
    """Return a shared VADER analyzer (lexicon load is cached)."""
    return SentimentIntensityAnalyzer()


def score_compound(text: str) -> float:
    """VADER compound polarity in [-1, 1] for a piece of text."""
    if not text or not str(text).strip():
        return 0.0
    return get_analyzer().polarity_scores(str(text))["compound"]


def label_sentiment(
    compound: float,
    pos_threshold: float = 0.05,
    neg_threshold: float = -0.05,
) -> str:
    """Map a compound score to positive / negative / neutral."""
    if compound >= pos_threshold:
        return "positive"
    if compound <= neg_threshold:
        return "negative"
    return "neutral"


def document_sentiment(
    df: pd.DataFrame,
    text_col: str = "text",
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Add ``doc_compound`` and ``doc_sentiment`` columns (one label per review)."""
    cfg = (config or load_config())["sentiment"]
    out = df.copy()
    out["doc_compound"] = out[text_col].map(score_compound)
    out["doc_sentiment"] = out["doc_compound"].map(
        lambda c: label_sentiment(c, cfg["pos_threshold"], cfg["neg_threshold"])
    )
    return out
