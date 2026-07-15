"""End-to-end test of the baseline pipeline on an in-memory frame."""

from __future__ import annotations

from reviewlens.config import load_config
from reviewlens.pipeline import ASPECT_COLUMNS, run_pipeline


def test_run_pipeline_end_to_end(sample_reviews):
    result = run_pipeline(source=sample_reviews, config=load_config())

    # doc-level baseline attached to reviews
    assert "doc_sentiment" in result.reviews.columns
    assert len(result.reviews) == 3

    # aspect frame has the agreed schema and is non-empty
    assert list(result.aspects.columns) == ASPECT_COLUMNS
    assert not result.aspects.empty

    aspects = set(result.aspects["aspect"])
    assert any("battery" in a for a in aspects)
    assert any("screen" in a for a in aspects)


def test_pipeline_captures_mixed_sentiment(sample_reviews):
    result = run_pipeline(source=sample_reviews, config=load_config())
    # "Great screen. Terrible battery life." -> one review, two opposite aspects
    per_review = result.aspects.groupby("review_id")["aspect_sentiment"].nunique()
    assert (per_review > 1).any(), "expected at least one mixed-sentiment review"


def test_aspect_summary_helper(sample_reviews):
    result = run_pipeline(source=sample_reviews, config=load_config())
    summary = result.aspect_summary(group_col="theme")
    assert "net_score" in summary.columns
    assert (summary["mentions"] > 0).all()
