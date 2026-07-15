"""Aggregation: turn per-aspect rows into business-ready summaries."""

from reviewlens.aggregate.summary import (
    aspect_distribution,
    representative_quotes,
    sentiment_by_rating,
    sentiment_over_time,
    top_hated,
    top_loved,
)

__all__ = [
    "aspect_distribution",
    "top_loved",
    "top_hated",
    "representative_quotes",
    "sentiment_by_rating",
    "sentiment_over_time",
]
