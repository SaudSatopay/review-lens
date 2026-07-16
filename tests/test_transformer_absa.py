"""Tests for the transformer ABSA classifier.

Label-mapping tests always run (pure functions, no weights). The tests that need
the actual checkpoint are opt-in, because downloading ~370MB has no business
running in a default `pytest` or in CI:

    REVIEWLENS_RUN_MODEL_TESTS=1 pytest tests/test_transformer_absa.py
"""

from __future__ import annotations

import os

import pytest

from reviewlens.sentiment.transformer_absa import _normalize_label

run_model_tests = pytest.mark.skipif(
    os.environ.get("REVIEWLENS_RUN_MODEL_TESTS") != "1",
    reason="set REVIEWLENS_RUN_MODEL_TESTS=1 to run tests that download the ABSA model",
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Positive", "positive"),
        ("NEGATIVE", "negative"),
        ("Neutral", "neutral"),
        ("POS", "positive"),
        ("neg", "negative"),
        ("LABEL_2", "positive"),
        ("LABEL_0", "negative"),
        ("LABEL_1", "neutral"),
    ],
)
def test_normalize_label_handles_checkpoint_variants(raw, expected):
    assert _normalize_label(raw) == expected


def test_normalize_label_rejects_unknown():
    with pytest.raises(ValueError, match="Unrecognized sentiment label"):
        _normalize_label("mildly_annoyed")


def test_module_imports_without_torch_installed():
    # Importing must never pull torch in; that's the whole point of the lazy init.
    import reviewlens.sentiment.transformer_absa as mod

    assert hasattr(mod, "TransformerAspectSentiment")


@run_model_tests
def test_absa_separates_contrasting_aspects_in_one_sentence():
    """The reason this model exists: VADER labels both aspects identically here."""
    from reviewlens.sentiment.transformer_absa import get_absa_model

    model = get_absa_model()
    sentence = "The display is stunning but the battery is a dealbreaker."
    display_label, display_score = model.predict(sentence, "display")
    battery_label, battery_score = model.predict(sentence, "battery")

    assert display_label == "positive"
    assert battery_label == "negative"
    assert display_score > 0 > battery_score


@run_model_tests
def test_absa_batch_matches_single_predictions():
    from reviewlens.sentiment.transformer_absa import get_absa_model

    model = get_absa_model()
    pairs = [
        ("Great camera but the battery drains way too fast.", "camera"),
        ("Great camera but the battery drains way too fast.", "battery"),
    ]
    batched = model.predict_batch(pairs)
    singles = [model.predict(s, a) for s, a in pairs]

    assert [x[0] for x in batched] == [x[0] for x in singles]
    assert batched[0][0] == "positive"
    assert batched[1][0] == "negative"
