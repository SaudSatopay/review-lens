"""Shared pytest fixtures."""

from __future__ import annotations

import pandas as pd
import pytest

from reviewlens.nltk_setup import ensure_nltk_data


@pytest.fixture(scope="session", autouse=True)
def _nltk_data() -> None:
    """Make sure NLTK corpora are present before any test that tokenizes."""
    ensure_nltk_data()


@pytest.fixture
def sample_reviews() -> pd.DataFrame:
    """A tiny in-memory review set with mixed per-aspect sentiment."""
    return pd.DataFrame(
        {
            "text": [
                "Great screen. Terrible battery life.",
                "The price is too high but the camera is excellent.",
                "Delivery was fast and packaging was great.",
            ],
            "rating": [4, 3, 5],
            "date": ["2026-01-01", "2026-01-15", "2026-02-01"],
            "product": ["PhoneX", "PhoneX", "PhoneX"],
        }
    )
