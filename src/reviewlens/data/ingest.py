"""Load reviews from CSV (or a DataFrame) into ReviewLens' canonical schema.

Canonical columns (see ``config.yaml`` -> ``schema``):
    review_id, text, rating, date, product

Only ``text`` is required in the source. Missing optional columns are created so
downstream code can rely on their presence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from reviewlens.config import load_config

CANONICAL_COLUMNS = ["review_id", "text", "rating", "date", "product"]


def load_reviews(
    source: str | Path | pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Read reviews and normalize to the canonical schema.

    Parameters
    ----------
    source:
        Path to a CSV file, or an already-loaded DataFrame.
    config:
        Optional pre-loaded config dict. Falls back to ``config.yaml``.
    """
    cfg = config or load_config()
    schema: dict[str, str] = cfg["schema"]

    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        df = pd.read_csv(source)

    # Map source column names -> canonical names where the source column exists.
    rename_map = {src: canon for canon, src in schema.items() if src in df.columns}
    df = df.rename(columns=rename_map)

    if "text" not in df.columns:
        raise ValueError(
            f"Input is missing the required text column. "
            f"Expected source column '{schema['text']}'. Found: {list(df.columns)}"
        )

    # Guarantee optional columns exist.
    if "review_id" not in df.columns:
        df.insert(0, "review_id", range(1, len(df) + 1))
    for col in ("rating", "date", "product"):
        if col not in df.columns:
            df[col] = pd.NA

    # Coerce dtypes that downstream steps assume.
    df["text"] = df["text"].astype("string")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Drop rows without usable text.
    df = df[df["text"].notna() & (df["text"].str.strip() != "")].reset_index(drop=True)

    return df[CANONICAL_COLUMNS]
