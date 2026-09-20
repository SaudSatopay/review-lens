"""Real-world demo data: Amazon product reviews -> canonical ReviewLens CSV.

Uses the 5-core category files of the Amazon Review Data (2018) corpus
(Ni, Li & McAuley, EMNLP 2019), served from the public UCSD research mirror —
the same arrangement as the SemEval downloader: fetched on demand into the
git-ignored ``data/raw/``, never redistributed from this repository.

Only small categories are offered so the demo stays a seconds-long download;
``Software`` is the sweet spot — 12.8k reviews dense with opinions about
interfaces, installation, pricing and support.

Record fields used: ``reviewText`` (text), ``overall`` (rating),
``unixReviewTime`` (date), ``asin`` (product).
"""

from __future__ import annotations

import gzip
import json
import urllib.request
from pathlib import Path

import pandas as pd

_MIRROR = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall"

# category name -> (filename stem, approx. review count) — small 5-core files only.
AMAZON_CATEGORIES: dict[str, tuple[str, str]] = {
    "appliances": ("Appliances_5", "~2.3k reviews, 73 KB"),
    "software": ("Software_5", "~12.8k reviews, 5.1 MB"),
    "magazines": ("Magazine_Subscriptions_5", "~2.4k reviews, 392 KB"),
    "beauty": ("All_Beauty_5", "~5.3k reviews, 620 KB"),
}


def download_amazon(category: str, dest_dir: str | Path, force: bool = False) -> Path:
    """Fetch one category's 5-core ``.json.gz`` into ``dest_dir`` (cached)."""
    if category not in AMAZON_CATEGORIES:
        raise ValueError(f"Unknown category {category!r}. Pick from: {sorted(AMAZON_CATEGORIES)}")
    stem, _ = AMAZON_CATEGORIES[category]
    dest = Path(dest_dir) / f"{stem}.json.gz"
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{_MIRROR}/{stem}.json.gz"
    with urllib.request.urlopen(url, timeout=120) as resp, open(dest, "wb") as out:
        out.write(resp.read())
    return dest


def parse_amazon_jsonl(path: str | Path) -> pd.DataFrame:
    """Read a (possibly gzipped) JSON-lines review file into a raw DataFrame."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    records = []
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)


def to_canonical(raw: pd.DataFrame, limit: int = 0, seed: int = 42) -> pd.DataFrame:
    """Map raw Amazon records to the canonical review schema.

    Drops records without usable ``reviewText``, de-duplicates exact repeats
    (the 5-core files contain some), optionally downsamples to ``limit`` rows
    (seeded — the demo is reproducible), and assigns sequential review ids.
    """
    df = raw.copy()
    for col in ("reviewText", "overall", "unixReviewTime", "asin"):
        if col not in df.columns:
            df[col] = pd.NA

    df["text"] = df["reviewText"].astype("string").str.strip()
    df = df[df["text"].notna() & (df["text"] != "")]
    dedupe_keys = [c for c in ("reviewerID", "asin", "unixReviewTime", "text") if c in df.columns]
    df = df.drop_duplicates(subset=dedupe_keys)

    if limit and len(df) > limit:
        df = df.sample(n=limit, random_state=seed)

    dates = pd.to_datetime(df["unixReviewTime"], unit="s", errors="coerce")
    out = pd.DataFrame(
        {
            "review_id": range(1, len(df) + 1),
            "text": df["text"].values,
            "rating": pd.to_numeric(df["overall"], errors="coerce").values,
            "date": dates.dt.date.values,
            "product": df["asin"].values,
        }
    )
    return out.reset_index(drop=True)
