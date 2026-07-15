"""Sentiment scoring: VADER doc-level baseline + per-aspect sentiment."""

from reviewlens.sentiment.aspect_sentiment import score_aspect_sentiment
from reviewlens.sentiment.vader_baseline import (
    document_sentiment,
    label_sentiment,
    score_compound,
)

__all__ = [
    "document_sentiment",
    "label_sentiment",
    "score_compound",
    "score_aspect_sentiment",
]
