"""Tests for baseline theme grouping."""

from __future__ import annotations

import pandas as pd

from reviewlens.clustering.themes import add_theme_column, assign_theme, normalize_term


def test_normalize_term_singularizes():
    assert normalize_term("Screens") == "screen"
    assert normalize_term("batteries") == "battery"
    assert normalize_term("cameras") == "camera"


def test_assign_theme_keyword_mapping():
    assert assign_theme("display") == "screen"
    assert assign_theme("shipping") == "delivery"
    assert assign_theme("customer service") == "service"
    assert assign_theme("battery life") == "battery"


def test_assign_theme_falls_back_to_normalized_term():
    # Unknown aspect keeps its own (normalized) identity.
    assert assign_theme("keyboards") == "keyboard"


def test_add_theme_column():
    df = pd.DataFrame({"aspect": ["display", "shipping", "widget"]})
    out = add_theme_column(df)
    assert list(out["theme"]) == ["screen", "delivery", "widget"]
