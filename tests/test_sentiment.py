"""Tests for VADER baseline and per-aspect sentiment."""

from __future__ import annotations

from reviewlens.sentiment.aspect_sentiment import score_aspect_sentiment
from reviewlens.sentiment.vader_baseline import label_sentiment, score_compound


def test_label_sentiment_thresholds():
    assert label_sentiment(0.5) == "positive"
    assert label_sentiment(-0.5) == "negative"
    assert label_sentiment(0.0) == "neutral"
    # custom thresholds
    assert label_sentiment(0.1, pos_threshold=0.2) == "neutral"


def test_score_compound_direction():
    assert score_compound("This is absolutely wonderful and I love it") > 0.05
    assert score_compound("This is terrible and I hate it") < -0.05
    assert score_compound("") == 0.0


def test_score_aspect_sentiment_returns_label_and_compound():
    label, compound = score_aspect_sentiment("The battery is amazing", aspect="battery")
    assert label == "positive"
    assert isinstance(compound, float)
