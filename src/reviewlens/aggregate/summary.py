"""Aggregate per-aspect sentiment rows into dashboard-ready tables.

Every function takes the long "aspect rows" frame produced by the pipeline with
(at least) the columns::

    aspect, theme, aspect_sentiment, aspect_compound, sentence, rating, date

and returns a tidy DataFrame ready to hand to Plotly / Streamlit.
"""

from __future__ import annotations

import pandas as pd

_SENTIMENTS = ["positive", "neutral", "negative"]


def aspect_distribution(df: pd.DataFrame, group_col: str = "aspect") -> pd.DataFrame:
    """Per-group sentiment counts + shares.

    Columns: ``<group_col>, positive, neutral, negative, mentions,
    pct_positive, pct_negative, net_score`` (net_score = (pos - neg) / mentions,
    in [-1, 1]). Sorted by mentions descending.
    """
    cols = [group_col, *_SENTIMENTS, "mentions", "pct_positive", "pct_negative", "net_score"]
    if df.empty:
        return pd.DataFrame(columns=cols)

    counts = (
        df.pivot_table(
            index=group_col,
            columns="aspect_sentiment",
            values="sentence",
            aggfunc="count",
            fill_value=0,
        )
        .reindex(columns=_SENTIMENTS, fill_value=0)
    )
    counts["mentions"] = counts[_SENTIMENTS].sum(axis=1)
    counts["pct_positive"] = counts["positive"] / counts["mentions"]
    counts["pct_negative"] = counts["negative"] / counts["mentions"]
    counts["net_score"] = (counts["positive"] - counts["negative"]) / counts["mentions"]
    return (
        counts.reset_index()
        .sort_values("mentions", ascending=False)
        .reset_index(drop=True)[cols]
    )


def _rank(
    df: pd.DataFrame,
    group_col: str,
    k: int,
    min_mentions: int,
    ascending: bool,
) -> pd.DataFrame:
    dist = aspect_distribution(df, group_col=group_col)
    dist = dist[dist["mentions"] >= min_mentions]
    return dist.sort_values("net_score", ascending=ascending).head(k).reset_index(drop=True)


def top_loved(
    df: pd.DataFrame,
    group_col: str = "aspect",
    k: int = 8,
    min_mentions: int = 2,
) -> pd.DataFrame:
    """Highest net-sentiment aspects (what customers love)."""
    return _rank(df, group_col, k, min_mentions, ascending=False)


def top_hated(
    df: pd.DataFrame,
    group_col: str = "aspect",
    k: int = 8,
    min_mentions: int = 2,
) -> pd.DataFrame:
    """Lowest net-sentiment aspects (what customers hate)."""
    return _rank(df, group_col, k, min_mentions, ascending=True)


def representative_quotes(
    df: pd.DataFrame,
    group_col: str = "aspect",
    per_side: int = 1,
) -> pd.DataFrame:
    """Most positive and most negative example sentence(s) per group.

    Columns: ``<group_col>, side, aspect_sentiment, aspect_compound, sentence``.
    """
    cols = [group_col, "side", "aspect_sentiment", "aspect_compound", "sentence"]
    if df.empty:
        return pd.DataFrame(columns=cols)

    rows: list[dict] = []
    for name, grp in df.groupby(group_col):
        positives = grp.sort_values("aspect_compound", ascending=False).head(per_side)
        negatives = grp.sort_values("aspect_compound", ascending=True).head(per_side)
        for side, subset in (("loved", positives), ("hated", negatives)):
            for _, r in subset.iterrows():
                rows.append(
                    {
                        group_col: name,
                        "side": side,
                        "aspect_sentiment": r["aspect_sentiment"],
                        "aspect_compound": r["aspect_compound"],
                        "sentence": r["sentence"],
                    }
                )
    return pd.DataFrame(rows, columns=cols)


def sentiment_by_rating(df: pd.DataFrame) -> pd.DataFrame:
    """Aspect-sentiment counts broken down by star rating (long format).

    Columns: ``rating, aspect_sentiment, count``. Rows with no rating are dropped.
    """
    cols = ["rating", "aspect_sentiment", "count"]
    if df.empty or "rating" not in df.columns:
        return pd.DataFrame(columns=cols)
    d = df.dropna(subset=["rating"])
    if d.empty:
        return pd.DataFrame(columns=cols)
    out = (
        d.groupby(["rating", "aspect_sentiment"]).size().reset_index(name="count")
    )
    return out


def sentiment_over_time(df: pd.DataFrame, freq: str = "ME") -> pd.DataFrame:
    """Aspect-sentiment counts over time, resampled at ``freq`` (default month-end).

    Columns: ``period, positive, neutral, negative``. Rows with no date dropped.
    """
    cols = ["period", *_SENTIMENTS]
    if df.empty or "date" not in df.columns:
        return pd.DataFrame(columns=cols)
    d = df.dropna(subset=["date"]).copy()
    if d.empty:
        return pd.DataFrame(columns=cols)
    d["period"] = pd.to_datetime(d["date"]).dt.to_period(freq[0]).dt.to_timestamp()
    out = (
        d.groupby(["period", "aspect_sentiment"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=_SENTIMENTS, fill_value=0)
        .reset_index()
    )
    return out
