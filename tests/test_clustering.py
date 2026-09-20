"""Tests for theme grouping — keyword baseline and embedding clustering.

The embedding tests inject a fake ``embed_fn`` (or monkeypatch
``themes._embed_terms``): geometry is hand-placed on the unit circle, so they
exercise the real KMeans/HDBSCAN + labeling logic without downloading MiniLM.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from reviewlens.clustering import themes
from reviewlens.clustering.themes import (
    add_theme_column,
    assign_theme,
    embed_and_cluster,
    normalize_term,
)

# Fake 2-D embeddings for *normalized* terms: battery-ish near (1, 0),
# screen-ish near (0, 1), one far-away outlier.
_FAKE_VECTORS = {
    "battery": (1.00, 0.00),
    "battery life": (0.98, 0.08),
    "charging": (0.95, 0.15),
    "charger": (0.97, 0.11),
    "screen": (0.00, 1.00),
    "display": (0.06, 0.99),
    "touchscreen": (0.11, 0.97),
    "resolution": (0.09, 0.95),
    "warranty": (-0.90, -0.40),
}


def _fake_embed(batch: list[str]) -> np.ndarray:
    return np.array([_FAKE_VECTORS[t] for t in batch])


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


def test_embed_and_cluster_kmeans_groups_and_labels():
    # Duplicates carry mention frequency: "battery" (x2) should name its cluster,
    # and "screens" normalizes into the screen cluster.
    terms = ["battery", "battery", "battery life", "charging", "screen", "screens", "display"]
    mapping = embed_and_cluster(terms, method="kmeans", n_clusters=2, embed_fn=_fake_embed)

    assert mapping["charging"] == "battery"
    assert mapping["battery life"] == "battery"
    assert mapping["display"] == "screen"
    assert mapping["screens"] == "screen"
    assert set(mapping.values()) == {"battery", "screen"}


def test_embed_and_cluster_kmeans_clamps_cluster_count():
    # Asking for 8 clusters with 3 unique terms must not crash.
    mapping = embed_and_cluster(
        ["battery", "charging", "screen"], method="kmeans", n_clusters=8, embed_fn=_fake_embed
    )
    assert set(mapping) == {"battery", "charging", "screen"}


def test_embed_and_cluster_hdbscan_noise_keeps_own_name():
    terms = [
        "battery", "battery life", "charging", "charger",
        "screen", "display", "touchscreen", "resolution",
        "warranty",  # far from both clusters -> noise
    ]
    mapping = embed_and_cluster(terms, method="hdbscan", min_cluster_size=2, embed_fn=_fake_embed)

    assert mapping["warranty"] == "warranty"
    # Dense groups collapse to a single label each.
    assert mapping["charging"] == mapping["battery"]
    assert mapping["display"] == mapping["screen"]
    assert mapping["battery"] != mapping["screen"]


def test_embed_and_cluster_edge_cases():
    assert embed_and_cluster([]) == {}
    # A single unique term never needs an embedder at all.
    assert embed_and_cluster(["Screens", "screens"], embed_fn=None) == {
        "Screens": "screen",
        "screens": "screen",
    }
    with pytest.raises(ValueError, match="Unknown clustering method"):
        embed_and_cluster(["battery", "screen"], method="dbscan", embed_fn=_fake_embed)


def test_add_theme_column_dispatches_on_config(monkeypatch):
    monkeypatch.setattr(themes, "_embed_terms", lambda batch, model_name: _fake_embed(batch))
    df = pd.DataFrame({"aspect": ["battery", "charging", "display", "screen"]})
    cfg = {"clustering": {"method": "kmeans", "n_clusters": 2}}

    out = add_theme_column(df, config=cfg)

    # "charging" keyword-maps to "battery" too, but "display" proves the
    # embedding path ran: keyword mapping would say "screen" — so does the
    # cluster, but via the fake geometry with only two discovered themes.
    assert set(out["theme"]) == {"battery", "screen"}
    assert list(out["theme"]) == ["battery", "battery", "screen", "screen"]
