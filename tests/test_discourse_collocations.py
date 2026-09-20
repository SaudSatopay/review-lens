"""Tests for the anaphora resolver and PMI collocation mining."""

from __future__ import annotations

from reviewlens.aspects.anaphora import resolve_pronoun_aspects
from reviewlens.aspects.collocations import pmi_collocations


def test_anaphora_fills_pronoun_initial_empty_sentence():
    review_ids = [1, 1, 1, 2]
    sentences = [
        "The battery is huge.",
        "It drains in an hour.",         # empty + pronoun -> inherits "battery"
        "Great screen though.",
        "It broke on day one.",          # review 2 has no antecedent -> untouched
    ]
    terms = [["battery"], [], ["screen"], []]

    out = resolve_pronoun_aspects(review_ids, sentences, terms)

    assert out[1] == ["battery"]
    assert out[3] == []                   # no antecedent in review 2
    assert out[0] == ["battery"] and out[2] == ["screen"]
    assert terms[1] == []                 # input untouched


def test_anaphora_never_touches_non_empty_or_non_pronoun_rows():
    out = resolve_pronoun_aspects(
        [1, 1, 1],
        ["The battery is huge.", "Amazing value here.", "It also has a screen."],
        [["battery"], [], ["screen"]],
    )
    assert out[1] == []                   # empty but not pronoun-initial
    assert out[2] == ["screen"]           # pronoun-initial but already has aspects


def test_anaphora_uses_most_recent_aspect():
    out = resolve_pronoun_aspects(
        [7, 7, 7],
        ["Screen is fine.", "Battery is big.", "It lasts forever."],
        [["screen"], ["battery"], []],
    )
    assert out[2] == ["battery"]          # recency, not first mention


def test_pmi_surfaces_collocations_and_excludes_stopword_pairs():
    corpus = (
        ["the battery life is bad"] * 5
        + ["battery life could improve"] * 3
        + ["the screen is good", "good screen overall", "battery is fine"] * 4
    )
    out = pmi_collocations(corpus, min_count=3, top_k=5)

    terms = dict(zip(out["term"], out["count"], strict=True))
    assert terms.get("battery life") == 8
    # "screen overall" only ever co-occurs -> legitimately outranks the looser
    # "battery life" pair (battery also appears alone): PMI doing its job.
    assert out.iloc[0]["term"] == "screen overall"
    # stopword-headed bigrams are excluded even when frequent
    assert not any(term.startswith(("the ", "is ")) for term in out["term"])


def test_pmi_empty_and_min_count():
    assert pmi_collocations([]).empty
    assert pmi_collocations(["battery life"], min_count=2).empty
