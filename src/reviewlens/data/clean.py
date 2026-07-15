"""Light text cleaning for reviews.

Deliberately conservative: we strip markup, URLs, and stray whitespace but keep
punctuation and casing, because both the sentence splitter and VADER rely on
them (e.g. "!" intensifies sentiment, capital letters signal emphasis).
"""

from __future__ import annotations

import html
import re

import pandas as pd

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normalize a single review string.

    Steps: unescape HTML entities -> drop HTML tags -> drop URLs ->
    collapse whitespace -> strip.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = str(text)
    text = html.unescape(text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def clean_reviews(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Apply :func:`clean_text` to a column and drop rows left empty."""
    out = df.copy()
    out[text_col] = out[text_col].map(clean_text)
    out = out[out[text_col].str.strip() != ""].reset_index(drop=True)
    return out
