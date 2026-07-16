"""ReviewLens dashboard (Streamlit + Plotly).

Run from the repo root::

    streamlit run app/streamlit_app.py

Two data modes (sidebar):

* **Processed outputs** — reads ``data/processed/*.parquet`` written by the
  pipeline CLI (whatever models that run used).
* **Live sample** — runs the pipeline on the bundled sample right in the app,
  with selectable extractor / sentiment models, so the baseline-vs-transformer
  difference can be flipped live.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from copy import deepcopy

import pandas as pd
import plotly.express as px
import streamlit as st

# Run without needing `pip install -e .`
SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reviewlens.aggregate.summary import (  # noqa: E402
    aspect_distribution,
    representative_quotes,
    sentiment_over_time,
)
from reviewlens.config import load_config, resolve_path  # noqa: E402
from reviewlens.pipeline import run_pipeline  # noqa: E402

SENTIMENT_COLORS = {"positive": "#2ca02c", "neutral": "#9e9e9e", "negative": "#d62728"}
FINETUNED_ABSA = "absa — our fine-tune"
PRETRAINED_ABSA = "absa — pretrained checkpoint"

st.set_page_config(page_title="ReviewLens", page_icon="🔍", layout="wide")


def _model_options() -> tuple[list[str], list[str]]:
    """Model choices that are actually available in this environment."""
    cfg = load_config()
    extractors = ["baseline"]
    if resolve_path(cfg["aspects"]["transformer_model_dir"]).exists():
        extractors.append("transformer")

    sentiments = ["baseline"]
    if importlib.util.find_spec("torch") is not None:
        if resolve_path(cfg["sentiment"]["absa_finetuned_dir"]).exists():
            sentiments.append(FINETUNED_ABSA)
        sentiments.append(PRETRAINED_ABSA)
    return extractors, sentiments


@st.cache_data(show_spinner="Running the pipeline on the sample…")
def build_live(extractor: str, sentiment: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the pipeline on the bundled sample with the selected models."""
    cfg = deepcopy(load_config())
    cfg["aspects"]["extractor"] = extractor
    if sentiment == "baseline":
        cfg["sentiment"]["aspect_model"] = "baseline"
    else:
        cfg["sentiment"]["aspect_model"] = "absa"
        if sentiment == FINETUNED_ABSA:
            cfg["sentiment"]["absa_model_name"] = str(
                resolve_path(cfg["sentiment"]["absa_finetuned_dir"])
            )
    result = run_pipeline(config=cfg)
    return result.reviews, result.aspects


@st.cache_data(show_spinner="Loading processed outputs…")
def load_processed() -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_config()
    processed = resolve_path(cfg["paths"]["data_processed"])
    aspects = pd.read_parquet(processed / "aspects.parquet")
    reviews_path = processed / "reviews.parquet"
    reviews = pd.read_parquet(reviews_path) if reviews_path.exists() else pd.DataFrame()
    return reviews, aspects


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    st.sidebar.header("Pipeline")
    processed_exists = (
        resolve_path(load_config()["paths"]["data_processed"]) / "aspects.parquet"
    ).exists()

    options = ["live sample"] + (["processed outputs"] if processed_exists else [])
    source = st.sidebar.radio("Data", options, index=0)

    if source == "processed outputs":
        return load_processed()

    extractors, sentiments = _model_options()
    extractor = st.sidebar.selectbox("Aspect extractor", extractors, index=0)
    sentiment = st.sidebar.selectbox("Aspect sentiment", sentiments, index=0)
    if st.sidebar.button("↻ Re-run"):
        build_live.clear()
    return build_live(extractor, sentiment)


def sidebar_filters(aspects: pd.DataFrame) -> dict:
    st.sidebar.header("Filters")
    group_by = st.sidebar.radio("Group aspects by", ["theme", "aspect"], index=0)

    products = sorted(p for p in aspects["product"].dropna().unique())
    selected_products = st.sidebar.multiselect("Product", products, default=products)

    ratings = aspects["rating"].dropna()
    if not ratings.empty:
        lo, hi = int(ratings.min()), int(ratings.max())
        rating_range = st.sidebar.slider("Rating", lo, hi, (lo, hi)) if lo < hi else (lo, hi)
    else:
        rating_range = None

    min_mentions = st.sidebar.slider("Min mentions per group", 1, 10, 2)
    return {
        "group_by": group_by,
        "products": selected_products,
        "rating_range": rating_range,
        "min_mentions": min_mentions,
    }


def apply_filters(aspects: pd.DataFrame, f: dict) -> pd.DataFrame:
    df = aspects
    if f["products"]:
        df = df[df["product"].isin(f["products"]) | df["product"].isna()]
    if f["rating_range"] is not None:
        lo, hi = f["rating_range"]
        df = df[df["rating"].between(lo, hi) | df["rating"].isna()]
    return df


def render_overview(reviews: pd.DataFrame, aspects: pd.DataFrame) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reviews", int(reviews["review_id"].nunique()) if not reviews.empty else 0)
    c2.metric("Aspect mentions", len(aspects))
    c3.metric("Distinct aspects", aspects["aspect"].nunique())
    mixed = aspects.groupby("review_id")["aspect_sentiment"].nunique().gt(1).sum()
    c4.metric("Mixed-sentiment reviews", int(mixed))


def render_distribution(aspects: pd.DataFrame, group_by: str, min_mentions: int) -> None:
    dist = aspect_distribution(aspects, group_col=group_by)
    dist = dist[dist["mentions"] >= min_mentions]
    if dist.empty:
        st.info("No groups meet the minimum-mentions threshold.")
        return

    st.subheader(f"Sentiment by {group_by}")
    long = dist.melt(
        id_vars=[group_by, "mentions", "net_score"],
        value_vars=["positive", "neutral", "negative"],
        var_name="sentiment",
        value_name="count",
    )
    order = dist.sort_values("net_score")[group_by].tolist()
    fig = px.bar(
        long, x="count", y=group_by, color="sentiment", orientation="h",
        color_discrete_map=SENTIMENT_COLORS, category_orders={group_by: order},
        height=max(300, 40 * len(order)),
    )
    fig.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Net sentiment (loved vs hated)")
    net = dist.sort_values("net_score")
    fig2 = px.bar(
        net, x="net_score", y=group_by, orientation="h", range_x=[-1, 1],
        color="net_score", color_continuous_scale=["#d62728", "#9e9e9e", "#2ca02c"],
        color_continuous_midpoint=0, height=max(300, 40 * len(net)),
    )
    fig2.update_layout(coloraxis_showscale=False, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig2, use_container_width=True)


def render_trend(aspects: pd.DataFrame) -> None:
    trend = sentiment_over_time(aspects, freq="ME")
    if trend.empty:
        return
    st.subheader("Sentiment over time")
    long = trend.melt(id_vars="period", var_name="sentiment", value_name="count")
    fig = px.area(
        long, x="period", y="count", color="sentiment",
        color_discrete_map=SENTIMENT_COLORS,
    )
    fig.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)


def render_quotes(aspects: pd.DataFrame, group_by: str) -> None:
    st.subheader("Representative quotes")
    quotes = representative_quotes(aspects, group_col=group_by, per_side=1)
    if quotes.empty:
        return
    for side, emoji in (("loved", "👍"), ("hated", "👎")):
        subset = quotes[quotes["side"] == side]
        if subset.empty:
            continue
        st.markdown(f"**{emoji} {side.title()}**")
        st.dataframe(
            subset[[group_by, "sentence", "aspect_compound"]].reset_index(drop=True),
            use_container_width=True, hide_index=True,
        )


def main() -> None:
    st.title("🔍 ReviewLens — Aspect-Based Review Intelligence")
    st.caption(
        "Per-aspect sentiment beats a single review score. Switch the models in "
        "the sidebar to watch the baseline and the transformer disagree."
    )

    reviews, aspects = load_data()
    if aspects.empty:
        st.warning("No aspect data available. Run the pipeline first.")
        return

    f = sidebar_filters(aspects)
    filtered = apply_filters(aspects, f)

    render_overview(reviews, filtered)
    st.divider()
    render_distribution(filtered, f["group_by"], f["min_mentions"])
    render_trend(filtered)
    render_quotes(filtered, f["group_by"])

    with st.expander("🧠 LLM executive summary (coming in a later slice)"):
        st.write(
            "This panel will host an LLM-generated summary of the top loved and "
            "hated aspects with recommendations (optional Ollama / API step)."
        )


if __name__ == "__main__":
    main()
else:
    # `streamlit run` executes the module top-to-bottom without __main__.
    main()
