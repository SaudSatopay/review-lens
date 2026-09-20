"""ReviewLens dashboard (Streamlit + Plotly) — the editorial instrument.

Run from the repo root::

    streamlit run app/streamlit_app.py

Two data modes (sidebar):

* **Processed outputs** — reads ``data/processed/*.parquet`` written by the
  pipeline CLI (whatever models that run used).
* **Live** — runs the pipeline right in the app (bundled sample or any demo CSV
  from ``scripts/download_reviews.py``), with selectable extractor / sentiment /
  theme models, so the baseline-vs-transformer difference can be flipped live.

Design notes — "print noir": a review is an *opinion column*, so the UI reads
like a well-set editorial page. Fraunces (display serif) carries headlines and
pull-quotes, Archivo carries UI, Spline Sans Mono carries every numeral. The
loved/hated axis is a diverging pair validated for the dark surface:
jade ``#2DA671`` / warm-gray midpoint ``#6E6557`` / vermilion ``#E4593B``
(poles pass lightness band, chroma floor, CVD + normal-vision separation and
contrast; the deliberately-gray midpoint leans on 2px segment gaps, the legend
and hover labels as secondary encoding).
"""

from __future__ import annotations

import html as html_lib
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

# ---------------------------------------------------------------------------
# Palette — single source of truth; the CSS block below mirrors these values.
# Validated with the dataviz palette validator against SURFACE (dark mode).
# ---------------------------------------------------------------------------
PALETTE = {
    "surface": "#171412",
    "surface2": "#1E1A15",
    "surface3": "#241F19",
    "ink": "#EDE6DA",
    "ink_muted": "#A89F90",
    "ink_faint": "#7A7264",
    "hairline": "rgba(237,230,218,0.12)",
    "grid": "rgba(237,230,218,0.07)",
    "positive": "#2DA671",   # jade
    "neutral": "#6E6557",    # warm gray — diverging midpoint, gray on purpose
    "negative": "#E4593B",   # vermilion
    "gold": "#C89B3C",       # hairline flourishes only
}

SENTIMENT_COLORS = {
    "positive": PALETTE["positive"],
    "neutral": PALETTE["neutral"],
    "negative": PALETTE["negative"],
}

FONT_UI = "Archivo, sans-serif"
FONT_MONO = "'Spline Sans Mono', monospace"

MAX_GROUPS_SHOWN = 24  # real corpora surface hundreds of themes; show the top slice

FINETUNED_ABSA = "absa — our fine-tune"
PRETRAINED_ABSA = "absa — pretrained checkpoint"

st.set_page_config(page_title="ReviewLens", page_icon="🔍", layout="wide")

# ---------------------------------------------------------------------------
# Stylesheet — typography, motion, chrome. Hex values mirror PALETTE above.
# ---------------------------------------------------------------------------


def _inject_css() -> None:
    css = (pathlib.Path(__file__).parent / "style.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Plotly styling
# ---------------------------------------------------------------------------
def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _style_fig(fig, height: int, legend: bool = False):
    """Shared chart chrome: transparent paper, recessive grid, mono numerals."""
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_UI, color=PALETTE["ink_muted"], size=12.5),
        margin=dict(l=8, r=14, t=30 if legend else 10, b=10),
        hoverlabel=dict(
            bgcolor=PALETTE["surface3"],
            bordercolor="rgba(237,230,218,0.25)",
            font=dict(family=FONT_MONO, color=PALETTE["ink"], size=12),
        ),
        showlegend=legend,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            title_text="", font=dict(family=FONT_MONO, size=11, color=PALETTE["ink_muted"]),
        ),
    )
    fig.update_xaxes(
        gridcolor=PALETTE["grid"], zerolinecolor=PALETTE["hairline"], zerolinewidth=1,
        tickfont=dict(family=FONT_MONO, size=11, color=PALETTE["ink_faint"]),
        title_font=dict(family=FONT_MONO, size=11), title_text="",
    )
    fig.update_yaxes(
        gridcolor="rgba(0,0,0,0)",
        tickfont=dict(family=FONT_UI, size=12.5, color=PALETTE["ink"]),
        title_text="",
    )
    return fig


# ---------------------------------------------------------------------------
# Markup helpers
# ---------------------------------------------------------------------------
def render_hero() -> None:
    st.markdown(
        """
        <div class="rl-hero">
          <div class="rl-kicker">ReviewLens · aspect-based review intelligence</div>
          <h1>Stop scoring reviews.<br>Start <em>reading</em> them<span class="rl-dot">.</span></h1>
          <p class="rl-dek">One review holds many verdicts — the display can be stunning
          while the battery is a dealbreaker. Each aspect gets its own.</p>
          <div class="rl-rules">
            <div class="rl-rule"></div><div class="rl-rule rl-rule--thin"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section(num: str, title: str, note: str = "") -> None:
    note_html = f'<div class="rl-section__note">{html_lib.escape(note)}</div>' if note else ""
    st.markdown(
        f"""
        <div class="rl-section">
          <div class="rl-section__row">
            <span class="rl-section__num">{num}</span>
            <span class="rl-section__title">{html_lib.escape(title)}</span>
            {note_html}
          </div>
          <div class="rl-section__rule"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview(reviews: pd.DataFrame, aspects: pd.DataFrame) -> None:
    n_reviews = int(reviews["review_id"].nunique()) if not reviews.empty else 0
    mixed = int(aspects.groupby("review_id")["aspect_sentiment"].nunique().gt(1).sum())
    mixed_pct = f"{mixed / n_reviews:.0%} of all reviews" if n_reviews else "—"
    tiles = [
        ("Reviews", f"{n_reviews:,}", "unique review ids", ""),
        ("Aspect mentions", f"{len(aspects):,}", "sentence-level opinions", ""),
        ("Distinct aspects", f"{aspects['aspect'].nunique():,}", "terms extracted", ""),
        ("Mixed-sentiment reviews", f"{mixed:,}", mixed_pct, " rl-tile--hot"),
    ]
    cells = "".join(
        f"""<div class="rl-tile{hot}">
              <div class="rl-tile__label">{label}</div>
              <div class="rl-tile__num">{num}</div>
              <div class="rl-tile__sub">{sub}</div>
            </div>"""
        for label, num, sub, hot in tiles
    )
    st.markdown(f'<div class="rl-tiles">{cells}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data plumbing (unchanged logic)
# ---------------------------------------------------------------------------
def _model_options() -> tuple[list[str], list[str], list[str]]:
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

    themes = ["normalized"]
    if importlib.util.find_spec("sentence_transformers") is not None:
        themes += ["kmeans", "hdbscan"]
    return extractors, sentiments, themes


def _demo_csvs() -> dict[str, str]:
    """Real-data demo CSVs written by scripts/download_reviews.py, if any."""
    demo_dir = resolve_path(load_config()["paths"]["data_raw"]) / "amazon"
    return {f"live: {p.stem}": str(p) for p in sorted(demo_dir.glob("*_reviews.csv"))}


@st.cache_data(show_spinner="Running the pipeline…")
def build_live(
    extractor: str, sentiment: str, clustering: str, source: str | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the pipeline on the sample (or a demo CSV) with the selected models."""
    cfg = deepcopy(load_config())
    cfg["aspects"]["extractor"] = extractor
    cfg["clustering"]["method"] = clustering
    if sentiment == "baseline":
        cfg["sentiment"]["aspect_model"] = "baseline"
    else:
        cfg["sentiment"]["aspect_model"] = "absa"
        cfg["sentiment"]["absa_checkpoint"] = (
            "finetuned" if sentiment == FINETUNED_ABSA else "pretrained"
        )
    result = run_pipeline(source=source, config=cfg)
    return result.reviews, result.aspects


@st.cache_data(show_spinner="Loading processed outputs…")
def load_processed() -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_config()
    processed = resolve_path(cfg["paths"]["data_processed"])
    aspects = pd.read_parquet(processed / "aspects.parquet")
    reviews_path = processed / "reviews.parquet"
    reviews = pd.read_parquet(reviews_path) if reviews_path.exists() else pd.DataFrame()
    return reviews, aspects


def _side_head(text: str) -> None:
    st.sidebar.markdown(f'<div class="rl-side-head">{text}</div>', unsafe_allow_html=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    st.sidebar.markdown(
        """
        <div class="rl-wordmark">Review<span class="rl-dot">·</span>Lens</div>
        <div class="rl-wordmark-sub">every aspect, its own verdict</div>
        """,
        unsafe_allow_html=True,
    )
    _side_head("Pipeline")
    processed_exists = (
        resolve_path(load_config()["paths"]["data_processed"]) / "aspects.parquet"
    ).exists()

    demos = _demo_csvs()  # real Amazon data from scripts/download_reviews.py
    options = ["live sample", *demos] + (["processed outputs"] if processed_exists else [])
    source = st.sidebar.radio("Data", options, index=0)

    if source == "processed outputs":
        return load_processed()

    extractors, sentiments, themes = _model_options()
    extractor = st.sidebar.selectbox("Aspect extractor", extractors, index=0)
    sentiment = st.sidebar.selectbox("Aspect sentiment", sentiments, index=0)
    clustering = st.sidebar.selectbox("Theme grouping", themes, index=0)
    if st.sidebar.button("↻ Re-run"):
        build_live.clear()
    return build_live(extractor, sentiment, clustering, demos.get(source))


def sidebar_filters(aspects: pd.DataFrame) -> dict:
    _side_head("Filters")
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


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def render_distribution(aspects: pd.DataFrame, group_by: str, min_mentions: int) -> None:
    dist = aspect_distribution(aspects, group_col=group_by)
    dist = dist[dist["mentions"] >= min_mentions]
    if dist.empty:
        st.info("No groups meet the minimum-mentions threshold.")
        return

    total_groups = len(dist)
    dist = dist.head(MAX_GROUPS_SHOWN)  # already sorted by mentions desc
    note = (
        f"top {len(dist)} of {total_groups} by mentions"
        if total_groups > len(dist)
        else f"{total_groups} groups"
    )

    render_section("01", f"Sentiment by {group_by}", note)
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
        custom_data=["sentiment"],
    )
    # 2px surface gaps between stacked segments + rounded, thin marks.
    fig.update_traces(
        marker_line_color=PALETTE["surface"], marker_line_width=2,
        hovertemplate="%{y} · %{customdata[0]}: <b>%{x}</b><extra></extra>",
    )
    fig.update_layout(bargap=0.45, barcornerradius=3)
    _style_fig(fig, height=max(320, 34 * len(order)), legend=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    render_section("02", "The verdict", "net = (pos − neg) ÷ mentions")
    net = dist.sort_values("net_score")
    fig2 = px.bar(
        net, x="net_score", y=group_by, orientation="h", range_x=[-1.05, 1.05],
        color="net_score",
        color_continuous_scale=[PALETTE["negative"], PALETTE["neutral"], PALETTE["positive"]],
        color_continuous_midpoint=0,
        text=net["net_score"].map(lambda v: f"{v:+.2f}"),
    )
    fig2.update_traces(
        marker_line_color=PALETTE["surface"], marker_line_width=1.5,
        textposition="outside",
        textfont=dict(family=FONT_MONO, size=11, color=PALETTE["ink_faint"]),
        cliponaxis=False,
        hovertemplate="%{y}: <b>%{x:+.2f}</b><extra></extra>",
    )
    fig2.update_layout(coloraxis_showscale=False, bargap=0.5, barcornerradius=3)
    _style_fig(fig2, height=max(320, 34 * len(net)))
    fig2.add_vline(x=0, line_color="rgba(237,230,218,0.28)", line_width=1)
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    with st.expander("View as table"):
        st.dataframe(dist.reset_index(drop=True), use_container_width=True, hide_index=True)


def render_trend(aspects: pd.DataFrame) -> None:
    trend = sentiment_over_time(aspects, freq="ME")
    if trend.empty:
        return
    render_section("03", "Over time", "mentions per month")
    long = trend.melt(id_vars="period", var_name="sentiment", value_name="count")
    fig = px.area(
        long, x="period", y="count", color="sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        category_orders={"sentiment": ["positive", "neutral", "negative"]},
    )
    for tr in fig.data:
        base = SENTIMENT_COLORS.get(tr.name, PALETTE["ink_muted"])
        tr.update(line=dict(width=2, color=base), fillcolor=_rgba(base, 0.22))
    fig.update_traces(hovertemplate="%{x|%b %Y} · <b>%{y}</b><extra></extra>")
    _style_fig(fig, height=340, legend=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_quotes(aspects: pd.DataFrame, group_by: str) -> None:
    quotes = representative_quotes(aspects, group_col=group_by, per_side=1)
    if quotes.empty:
        return
    render_section("04", "In their own words", "most polarized quote per group")

    def _cards(subset: pd.DataFrame, side: str) -> str:
        cards = []
        for _, r in subset.head(6).iterrows():
            cards.append(
                f"""<div class="rl-quote rl-quote--{side}">
                     <p class="rl-quote__text">{html_lib.escape(str(r['sentence']))}</p>
                     <div class="rl-quote__meta">
                       <span class="rl-tag">{side}</span>
                       <span>{html_lib.escape(str(r[group_by]))}</span>
                       <span class="rl-score">{r['aspect_compound']:+.2f}</span>
                     </div>
                   </div>"""
            )
        return "".join(cards)

    loved = quotes[(quotes["side"] == "loved") & (quotes["aspect_compound"] > 0)]
    hated = quotes[(quotes["side"] == "hated") & (quotes["aspect_compound"] < 0)]
    # One sentence can carry several themes — show it once, under its strongest.
    loved = loved.sort_values("aspect_compound", ascending=False).drop_duplicates("sentence")
    hated = hated.sort_values("aspect_compound").drop_duplicates("sentence")

    col_l, col_r = st.columns(2, gap="large")
    with col_l:
        st.markdown(_cards(loved, "loved"), unsafe_allow_html=True)
    with col_r:
        st.markdown(_cards(hated, "hated"), unsafe_allow_html=True)


def render_llm_summary(aspects: pd.DataFrame, group_by: str) -> None:
    """Optional LLM executive summary — Ollama or an OpenAI-compatible endpoint."""
    render_section("05", "The executive summary", "optional · LLM")
    with st.expander("Ask a model to read the numbers"):
        st.caption(
            "A local Ollama model — or any OpenAI-compatible endpoint from your "
            ".env — receives the aggregated stats digest (never the raw reviews) "
            "and writes the “so what”. See .env.example for setup."
        )
        if st.button("Generate summary"):
            from reviewlens.aggregate.llm_summary import generate_summary

            try:
                with st.spinner("Asking the LLM…"):
                    st.session_state["llm_summary"] = generate_summary(
                        aspects, group_col=group_by
                    )
            except RuntimeError as exc:
                st.warning(str(exc))
        if st.session_state.get("llm_summary"):
            st.markdown(st.session_state["llm_summary"])


def render_footer() -> None:
    st.markdown(
        """
        <div class="rl-footer">
          <span>ReviewLens</span><span>·</span><span>MIT © 2026 Saud Satopay</span>
          <span class="rl-right">fine-tuned RoBERTa ABSA · measured on SemEval-2014</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
def main() -> None:
    _inject_css()
    render_hero()

    reviews, aspects = load_data()
    if aspects.empty:
        st.warning("No aspect data available. Run the pipeline first.")
        return

    f = sidebar_filters(aspects)
    filtered = apply_filters(aspects, f)

    render_overview(reviews, filtered)
    render_distribution(filtered, f["group_by"], f["min_mentions"])
    render_trend(filtered)
    render_quotes(filtered, f["group_by"])
    render_llm_summary(filtered, f["group_by"])
    render_footer()


if __name__ == "__main__":
    main()
else:
    # `streamlit run` executes the module top-to-bottom without __main__.
    main()
