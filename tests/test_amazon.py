"""Tests for the Amazon demo-data converter (no network — tiny inline fixtures)."""

from __future__ import annotations

import gzip
import json

import pandas as pd

from reviewlens.data.amazon import parse_amazon_jsonl, to_canonical


def _raw(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_to_canonical_maps_schema_and_dates():
    raw = _raw(
        [
            {
                "reviewerID": "A1", "asin": "B001", "overall": 5.0,
                "unixReviewTime": 1577836800,  # 2020-01-01
                "reviewText": "Great blender, powerful motor.",
            },
            {
                "reviewerID": "A2", "asin": "B002", "overall": 2.0,
                "unixReviewTime": 1580515200,  # 2020-02-01
                "reviewText": "The lid cracked after a week.",
            },
        ]
    )
    out = to_canonical(raw)

    assert list(out.columns) == ["review_id", "text", "rating", "date", "product"]
    assert list(out["review_id"]) == [1, 2]
    assert out.loc[0, "rating"] == 5.0
    assert str(out.loc[0, "date"]) == "2020-01-01"
    assert out.loc[1, "product"] == "B002"


def test_to_canonical_drops_empty_text_and_duplicates():
    row = {
        "reviewerID": "A1", "asin": "B001", "overall": 4.0,
        "unixReviewTime": 1577836800, "reviewText": "Solid product.",
    }
    raw = _raw([row, dict(row), {**row, "reviewText": "   "}, {**row, "reviewText": None}])

    out = to_canonical(raw)
    assert len(out) == 1


def test_to_canonical_sampling_is_seeded():
    raw = _raw(
        [
            {
                "reviewerID": f"A{i}", "asin": f"B{i}", "overall": 3.0,
                "unixReviewTime": 1577836800 + i, "reviewText": f"Review number {i}.",
            }
            for i in range(50)
        ]
    )
    a = to_canonical(raw, limit=10, seed=42)
    b = to_canonical(raw, limit=10, seed=42)

    assert len(a) == 10
    assert list(a["text"]) == list(b["text"])          # reproducible
    assert list(a["review_id"]) == list(range(1, 11))  # ids re-assigned after sampling


def test_parse_amazon_jsonl_reads_gzip(tmp_path):
    records = [
        {"reviewText": "Nice.", "overall": 5, "asin": "B1", "unixReviewTime": 1},
        {"reviewText": "Bad.", "overall": 1, "asin": "B2", "unixReviewTime": 2},
    ]
    path = tmp_path / "sample.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    df = parse_amazon_jsonl(path)
    assert len(df) == 2
    assert set(df.columns) >= {"reviewText", "overall", "asin"}
