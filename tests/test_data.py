"""Tests for ingest / clean / split."""

from __future__ import annotations

import pandas as pd

from reviewlens.config import load_config
from reviewlens.data.clean import clean_reviews, clean_text
from reviewlens.data.ingest import load_reviews
from reviewlens.data.split import explode_sentences, split_into_sentences


def test_clean_text_strips_html_urls_and_whitespace():
    dirty = "  Great   phone!! <br> Visit https://x.com now &amp; save  "
    cleaned = clean_text(dirty)
    assert "<br>" not in cleaned
    assert "https://x.com" not in cleaned
    assert "&amp;" not in cleaned and "&" in cleaned  # entity unescaped
    assert "  " not in cleaned  # whitespace collapsed
    assert cleaned.startswith("Great phone")


def test_clean_text_handles_missing():
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""


def test_clean_reviews_drops_empty_rows():
    df = pd.DataFrame({"text": ["real review", "   ", "<br>"]})
    out = clean_reviews(df)
    assert len(out) == 1
    assert out.iloc[0]["text"] == "real review"


def test_load_reviews_normalizes_schema():
    df = pd.DataFrame({"text": ["a review"], "rating": ["5"]})
    out = load_reviews(df, load_config())
    # canonical columns present, review_id auto-created, rating numeric
    assert list(out.columns) == ["review_id", "text", "rating", "date", "product"]
    assert out.iloc[0]["review_id"] == 1
    assert out.iloc[0]["rating"] == 5


def test_split_into_sentences_counts():
    assert split_into_sentences("First. Second! Third?") == ["First.", "Second!", "Third?"]
    assert split_into_sentences("   ") == []


def test_explode_sentences_expands_and_keeps_metadata(sample_reviews):
    reviews = load_reviews(sample_reviews, load_config())
    exploded = explode_sentences(reviews)
    # first review has 2 sentences, so total > number of reviews
    assert len(exploded) > len(reviews)
    assert {"review_id", "sentence_id", "sentence"}.issubset(exploded.columns)
    assert (exploded["sentence"].str.len() > 0).all()
