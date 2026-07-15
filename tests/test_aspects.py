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


def test_empty_input_returns_empty():
    assert extract_aspects("") == []
    assert extract_aspects("   ") == []


def test_respects_max_words():
    aspects = extract_aspects("The customer service response time was slow.")
    assert all(len(a.split()) <= 4 for a in aspects)
