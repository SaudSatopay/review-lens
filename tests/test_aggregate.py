"""Tests for aggregation / summary tables."""

from __future__ import annotations

import pandas as pd
import pytest

from reviewlens.aggregate.summary import (
    aspect_distribution,
    representative_quotes,
    sentiment_by_rating,
    sentiment_over_time,
    top_hated,
    top_loved,
)


@pytest.fixture
def aspect_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "aspect": ["battery", "battery", "battery", "screen", "screen"],
            "theme": ["battery", "battery", "battery", "screen", "screen"],
            "aspect_sentiment": ["positive", "positive", "negative", "positive", "negative"],
            "aspect_compound": [0.6, 0.5, -0.7, 0.4, -0.3],
            "sentence": ["s1", "s2", "s3", "s4", "s5"],
            "rating": [5, 5, 2, 4, 3],
            "date": pd.to_datetime(
                ["2026-01-01", "2026-01-15", "2026-02-01", "2026-01-10", "2026-02-05"]
            ),
        }
    )


def test_aspect_distribution_counts_and_scores(aspect_rows):
    dist = aspect_distribution(aspect_rows).set_index("aspect")
    assert dist.loc["battery", "mentions"] == 3
    assert dist.loc["battery", "positive"] == 2
    assert dist.loc["battery", "negative"] == 1
    assert dist.loc["battery", "net_score"] == pytest.approx((2 - 1) / 3)
    assert dist.loc["screen", "net_score"] == pytest.approx(0.0)


def test_top_loved_and_hated_ordering(aspect_rows):
    loved = top_loved(aspect_rows, min_mentions=2)
    hated = top_hated(aspect_rows, min_mentions=2)
    assert loved.iloc[0]["aspect"] == "battery"   # net +0.33 beats screen 0.0
    assert hated.iloc[0]["aspect"] == "screen"    # net 0.0 is the lowest


def test_representative_quotes_picks_extremes(aspect_rows):
    quotes = representative_quotes(aspect_rows, per_side=1)
    battery = quotes[quotes["aspect"] == "battery"]
    loved = battery[battery["side"] == "loved"].iloc[0]
    hated = battery[battery["side"] == "hated"].iloc[0]
    assert loved["sentence"] == "s1"   # highest compound (0.6)
    assert hated["sentence"] == "s3"   # lowest compound (-0.7)


def test_sentiment_by_rating(aspect_rows):
    out = sentiment_by_rating(aspect_rows)
    row = out[(out["rating"] == 5) & (out["aspect_sentiment"] == "positive")]
    assert int(row["count"].iloc[0]) == 2


def test_sentiment_over_time_buckets_by_month(aspect_rows):
    out = sentiment_over_time(aspect_rows, freq="ME")
    assert len(out) == 2  # January and February 2026
    assert set(["period", "positive", "neutral", "negative"]).issubset(out.columns)


def test_empty_inputs_return_empty_frames():
    empty = pd.DataFrame(
        columns=["aspect", "aspect_sentiment", "aspect_compound", "sentence", "rating", "date"]
    )
    assert aspect_distribution(empty).empty
    assert representative_quotes(empty).empty
    assert sentiment_over_time(empty).empty
