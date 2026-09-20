"""Tests for the classical models: CRF extractor + Naive Bayes sentiment.

Both train in-test on tiny synthetic data (milliseconds, CPU) — no SemEval
download, no model artifacts required.
"""

from __future__ import annotations

import joblib
import pytest

from reviewlens.aspects.crf import (
    CRFAspectExtractor,
    bio_tags_for_sentence,
    featurize,
    tokenize_with_offsets,
    train_crf,
)
from reviewlens.sentiment.naive_bayes import NaiveBayesAspectSentiment, train_naive_bayes


def test_tokenize_with_offsets_roundtrips():
    text = "The battery life isn't great."
    triples = tokenize_with_offsets(text)
    assert [t for t, _, _ in triples] == ["The", "battery", "life", "isn't", "great", "."]
    for token, start, end in triples:
        assert text[start:end] == token


def test_bio_tags_for_gold_spans():
    text = "The battery life is bad"
    # gold span covers "battery life"
    tags = bio_tags_for_sentence(text, [(4, 16)])
    assert tags == ["O", "B-ASP", "I-ASP", "O", "O"]


def test_featurize_produces_context_features():
    feats, offsets = featurize("Great screen quality")
    assert len(feats) == len(offsets) == 3
    assert feats[0]["BOS"] is True and feats[-1]["EOS"] is True
    assert feats[1]["-1:word.lower"] == "great"
    assert "pos" in feats[1]


def test_crf_learns_a_tiny_pattern(tmp_path):
    # "battery"/"screen" after "the ... is" are aspects; teach + verify.
    sentences = [
        "the battery is bad", "the screen is great", "the battery is huge",
        "the screen is dim", "I love it", "delivery was fast",
    ] * 4
    spans = ([[(4, 11)], [(4, 10)], [(4, 11)], [(4, 10)], [], []]) * 4
    crf = train_crf(sentences, spans, max_iterations=50)

    path = tmp_path / "crf.joblib"
    joblib.dump(crf, path)
    extractor = CRFAspectExtractor(path)

    out = extractor.extract_batch(["the battery is awful", "", "the screen is nice"])
    assert out[0] == ["battery"]
    assert out[1] == []
    assert out[2] == ["screen"]


def test_crf_missing_model_raises_with_instructions(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_classical"):
        CRFAspectExtractor(tmp_path / "nope.joblib")


def test_naive_bayes_contract(tmp_path):
    pairs = [
        ("the battery is terrible and awful", "battery"),
        ("terrible awful product broke", "product"),
        ("great excellent screen love it", "screen"),
        ("excellent great love this camera", "camera"),
        ("it is a phone nothing special", "phone"),
        ("average ordinary standard box", "box"),
    ] * 5
    labels = ["negative", "negative", "positive", "positive", "neutral", "neutral"] * 5
    model = train_naive_bayes(pairs, labels)

    path = tmp_path / "nb.joblib"
    joblib.dump(model, path)
    nb = NaiveBayesAspectSentiment(path)

    preds = nb.predict_batch(
        [("terrible awful battery", "battery"), ("great excellent screen", "screen")]
    )
    (neg_label, neg_score), (pos_label, pos_score) = preds
    assert neg_label == "negative" and neg_score < 0
    assert pos_label == "positive" and pos_score > 0
    assert nb.predict_batch([]) == []


def test_naive_bayes_missing_model_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_classical"):
        NaiveBayesAspectSentiment(tmp_path / "nope.joblib")
