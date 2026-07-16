"""Tests for the evaluation metrics (hand-computed expectations)."""

from __future__ import annotations

import pandas as pd
import pytest

from reviewlens.evaluation.metrics import extraction_scores, sentiment_scores


def _terms(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["sentence_id", "term"])


def test_extraction_scores_hand_case():
    gold = _terms([("s1", "battery"), ("s1", "screen"), ("s2", "price")])
    pred = _terms([("s1", "battery"), ("s1", "keyboard"), ("s3", "shipping")])
    # tp=1 (battery); fp=2 (keyboard, shipping); fn=2 (screen, price)
    scores = extraction_scores(gold, pred)
    assert scores["true_positives"] == 1
    assert scores["false_positives"] == 2
    assert scores["false_negatives"] == 2
    assert scores["precision"] == pytest.approx(1 / 3, abs=1e-4)
    assert scores["recall"] == pytest.approx(1 / 3, abs=1e-4)
    assert scores["f1"] == pytest.approx(1 / 3, abs=1e-4)


def test_extraction_scores_case_insensitive():
    gold = _terms([("s1", "Battery Life")])
    pred = _terms([("s1", "battery life")])
    assert extraction_scores(gold, pred)["f1"] == 1.0


def test_extraction_scores_empty_pred():
    gold = _terms([("s1", "battery")])
    scores = extraction_scores(gold, _terms([]))
    assert scores["precision"] == 0.0
    assert scores["recall"] == 0.0
    assert scores["f1"] == 0.0


def test_sentiment_scores_hand_case():
    y_true = ["positive", "negative", "neutral", "positive"]
    y_pred = ["positive", "negative", "positive", "positive"]
    # positive: P=2/3 R=1 F1=0.8 | negative: F1=1.0 | neutral: F1=0.0
    scores = sentiment_scores(y_true, y_pred)
    assert scores["accuracy"] == 0.75
    assert scores["macro_f1"] == pytest.approx((0.8 + 1.0 + 0.0) / 3, abs=1e-4)
    assert scores["per_class"]["positive"]["f1"] == pytest.approx(0.8, abs=1e-4)
    assert scores["per_class"]["neutral"]["support"] == 1


def test_sentiment_scores_validates_input():
    with pytest.raises(ValueError, match="length mismatch"):
        sentiment_scores(["positive"], [])
    with pytest.raises(ValueError, match="no examples"):
        sentiment_scores([], [])
