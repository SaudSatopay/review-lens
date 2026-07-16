"""Tests for the noun-phrase aspect-extraction baseline."""

from __future__ import annotations

from reviewlens.aspects.baseline import extract_aspects


def test_extracts_noun_aspects():
    aspects = extract_aspects("The battery life is great but the screen is dim.")
    assert any("battery" in a for a in aspects)
    assert any("screen" in a for a in aspects)


def test_excludes_pure_opinion_words():
    # Adjectives carry opinion, not aspect — they should not be returned.
    aspects = extract_aspects("It is great and very fast.")
    assert "great" not in aspects
    assert "fast" not in aspects


def test_sentence_initial_adjective_is_not_part_of_the_aspect():
    # Regression: the tagger reads a capitalized leading "Great" as a proper
    # noun, which used to yield the aspect "great camera" instead of "camera".
    aspects = extract_aspects("Great camera and gorgeous display.")
    assert "camera" in aspects
    assert "great camera" not in aspects


def test_leading_acronym_is_preserved():
    # The de-capitalization fix must not mangle acronyms.
    aspects = extract_aspects("GPS locks on quickly during runs.")
    assert any("gps" in a for a in aspects)


def test_empty_input_returns_empty():
    assert extract_aspects("") == []
    assert extract_aspects("   ") == []


def test_respects_max_words():
    aspects = extract_aspects("The customer service response time was slow.")
    assert all(len(a.split()) <= 4 for a in aspects)
