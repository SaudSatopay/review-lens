"""Sentence splitting.

ABSA operates at sentence granularity: a single review often praises one aspect
while criticizing another ("great camera, terrible battery"), so we explode each
review into sentences before extracting aspects and scoring sentiment.
"""

from __future__ import annotations

import pandas as pd
from nltk.tokenize import sent_tokenize

from reviewlens.nltk_setup import ensure_nltk_data


def split_into_sentences(text: str) -> list[str]:
    """Split a text into a list of non-empty, stripped sentences."""
    if not text or not str(text).strip():
        return []
    ensure_nltk_data()
    return [s.strip() for s in sent_tokenize(str(text)) if s.strip()]


def explode_sentences(
    df: pd.DataFrame,
    text_col: str = "text",
    id_col: str = "review_id",
) -> pd.DataFrame:
    """Explode a review-level frame into a sentence-level frame.

    Returns one row per sentence, carrying review metadata plus:
        sentence_id  -- 0-based index of the sentence within its review
        sentence     -- the sentence text
    """
    ensure_nltk_data()
    records: list[dict] = []
    meta_cols = [c for c in df.columns if c != text_col]

    for _, row in df.iterrows():
        sentences = split_into_sentences(row[text_col])
        for i, sentence in enumerate(sentences):
            record = {c: row[c] for c in meta_cols}
            record["sentence_id"] = i
            record["sentence"] = sentence
            records.append(record)

    out = pd.DataFrame.from_records(records)
    if out.empty:
        # Preserve a stable schema even when there is nothing to split.
        return pd.DataFrame(columns=[*meta_cols, "sentence_id", "sentence"])

    # Stable, readable column order: ids/meta first, then sentence fields.
    ordered = [id_col] + [c for c in meta_cols if c != id_col] + ["sentence_id", "sentence"]
    return out[ordered]
