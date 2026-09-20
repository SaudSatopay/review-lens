"""ReviewLens dashboard (Streamlit + Plotly) — the data journal.

Run from the repo root::

    streamlit run app/streamlit_app.py

Two data modes (sidebar):

* **Processed outputs** — reads ``data/processed/*.parquet`` written by the
  pipeline CLI (whatever models that run used).
* **Live** — runs the pipeline right in the app (bundled sample or any demo CSV
  from ``scripts/download_reviews.py``), with selectable extractor / sentiment /
  theme models, so the baseline-vs-transformer difference can be flipped live.

Design notes — warm paper, near-black ink, editorial composition: Gloock (a
display didone) sets the masthead, headlines and figures; Instrument Serif
italic carries deks and quotes; Instrument Sans is the working sans; Spline
Sans Mono sets every numeral, kicker and axis. Semantic color is a diverging
pair validated for the paper surface (pine-teal ``#0F8265`` / warm-gray
midpoint ``#A2988A`` / vermilion ``#C34A24`` — the poles pass lightness band,
chroma floor, CVD and normal-vision separation and contrast; the
deliberately-gray midpoint leans on 2px segment gaps, the legend, hover labels
and each chart's table view). Burgundy ``#6E2231`` is the brand accent and
never appears inside a chart.
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
# Palette — single source of truth; app/style.css mirrors these values.
# Validated with the dataviz palette validator against SURFACE (light mode).
# ---------------------------------------------------------------------------
PALETTE = {
    "surface": "#F5F0E6",     # warm paper
    "surface2": "#EFE8DA",    # sidebar paper
    "card": "#FCF9F1",        # clipping / hover card
    "ink": "#191511",
    "ink_muted": "#6B6154",
    "ink_faint": "#99907F",
    "hairline": "rgba(25,21,17,0.16)",
    "grid": "rgba(25,21,17,0.08)",
    "positive": "#0F8265",    # pine teal
    "neutral": "#A2988A",     # warm gray — diverging midpoint, gray on purpose
    "negative": "#C34A24",    # burnt vermilion
    "brand": "#6E2231",       # burgundy — UI accent only, never in charts
}

SENTIMENT_COLORS = {
    "positive": PALETTE["positive"],
    "neutral": PALETTE["neutral"],
    "negative": PALETTE["negative"],
}

FONT_UI = "'Instrument Sans', sans-serif"
FONT_MONO = "'Spline Sans Mono', monospace"

MAX_GROUPS_SHOWN = 24  # real corpora surface hundreds of themes; show the top slice

FINETUNED_ABSA = "absa — our fine-tune"
PRETRAINED_ABSA = "absa — pretrained checkpoint"
NB_SENTIMENT = "nb — naïve bayes"

st.set_page_config(page_title="ReviewLens", page_icon="🔍", layout="wide")


def _inject_css() -> None:
    css = (pathlib.Path(__file__).parent / "style.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Plotly styling — publication chrome shared by every chart.
# ---------------------------------------------------------------------------
def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _style_fig(fig, height: int, legend: bool = False):
    """Transparent paper, recessive warm grid, mono numerals, ink labels."""
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_UI, color=PALETTE["ink_muted"], size=12.5),
        margin=dict(l=8, r=16, t=30 if legend else 12, b=12),
        hoverlabel=dict(
            bgcolor=PALETTE["card"],
            bordercolor="rgba(25,21,17,0.35)",
            font=dict(family=FONT_MONO, color=PALETTE["ink"], size=12),
        ),
        showlegend=legend,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            title_text="", font=dict(family=FONT_MONO, size=10.5, color=PALETTE["ink_muted"]),
        ),
    )
    fig.update_xaxes(
        gridcolor=PALETTE["grid"], zerolinecolor="rgba(25,21,17,0.25)", zerolinewidth=1,
        tickfont=dict(family=FONT_MONO, size=10.5, color=PALETTE["ink_faint"]),
        title_text="", showline=False,
    )
    fig.update_yaxes(
        gridcolor="rgba(0,0,0,0)",
        tickfont=dict(family=FONT_UI, size=12.5, color=PALETTE["ink"]),
        title_text="",
    )
    return fig


# ---------------------------------------------------------------------------
# Editorial furniture
# ---------------------------------------------------------------------------
def render_masthead() -> None:
    st.markdown(
        """
        <div class="rl-masthead">
          <div class="rl-masthead__rule"></div>
          <div class="rl-masthead__rule rl-masthead__rule--thin"></div>
          <div class="rl-masthead__row">
            <span class="rl-masthead__brand">ReviewLens<span class="rl-dot">.</span></span>
            <span class="rl-masthead__tag">Aspect-based review intelligence</span>
            <span class="rl-masthead__edition">measured on SemEval-2014 · MIT</span>
          </div>
          <div class="rl-masthead__rule rl-masthead__rule--thin"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    """Headline left; right, the canonical two-star review as an annotated exhibit."""
    st.markdown(
        """
        <div class="rl-hero">
          <div class="rl-hero__left">
            <div class="rl-kicker">The case for reading properly</div>
            <h1>Stop scoring reviews.<br>Start <em>reading</em> them<span
              class="rl-dot">.</span></h1>
            <p class="rl-dek">One review holds many verdicts. A single sentiment
            score averages a stunning display against a dead battery and calls
            the result &ldquo;positive&rdquo; — per-aspect reading keeps every
            opinion intact.</p>
            <div class="rl-rules">
              <div class="rl-rule"></div><div class="rl-rule rl-rule--thin"></div>
            </div>
          </div>
          <div class="rl-specimen">
            <div class="rl-specimen__head">
              <span>Exhibit A — a two-star review</span>
              <span class="rl-specimen__stars">★★☆☆☆</span>
            </div>
            <p class="rl-specimen__text">&ldquo;The
            <span class="rl-m rl-m--pos">display</span> is stunning but the
            <span class="rl-m rl-m--neg">battery</span> is a dealbreaker. Also the
            <span class="rl-m rl-m--neg">price</span> keeps going up while quality
            stays the same.&rdquo;</p>
            <div class="rl-specimen__verdicts">
              <div><span>display</span><span class="rl-v--pos">positive&ensp;+0.997</span></div>
              <div><span>battery</span><span class="rl-v--neg">negative&ensp;−0.970</span></div>
              <div><span>price</span><span class="rl-v--neg">negative&ensp;−0.603</span></div>
              <div><span>one document-level score</span>
                   <span class="rl-v--flat">positive&ensp;+0.20</span></div>
            </div>
            <div class="rl-specimen__moral">— the flattening, exhibit closed</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section(num: str, title: str, note: str = "", stand: str = "") -> None:
    note_html = f'<div class="rl-section__note">{html_lib.escape(note)}</div>' if note else ""
    stand_html = (
        f'<div class="rl-section__stand">{html_lib.escape(stand)}</div>' if stand else ""
    )
    st.markdown(
        f"""
        <div class="rl-section">
          <div class="rl-section__row">
            <span class="rl-section__num">№ {num}</span>
            <span class="rl-section__title">{html_lib.escape(title)}</span>
            {note_html}
          </div>
          {stand_html}
          <div class="rl-section__rule"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _histogram_rows(meta: dict) -> str:
    histogram = {int(k): v for k, v in (meta.get("histogram") or {}).items()}
    return "".join(
        f"""<div class="rl-product__row"><span class="rl-product__stars">{stars}★</span>
              <div class="rl-product__bar"><div style="width:{histogram[stars]}%"></div></div>
              <span class="rl-product__pct">{histogram[stars]}%</span></div>"""
        for stars in sorted(histogram, reverse=True)
    )


def _stars(rating: float | None) -> str:
    if rating is None:
        return ""
    full = int(round(rating))
    return "★" * full + "☆" * (5 - full)


def _photo_html(meta: dict, css_class: str) -> str:
    url = meta.get("image_url")
    if not url:
        return ""
    return (
        f'<div class="{css_class}"><img src="{html_lib.escape(str(url))}" '
        f'alt="{html_lib.escape(str(meta.get("title", "product photo")))}"></div>'
    )


def render_product_strip(meta: dict) -> None:
    """Rating stats of a live-fetched Amazon product: photo, title, histogram."""
    count = meta.get("ratings_count")
    count_text = f"{count:,} ratings on Amazon" if count else "ratings on Amazon"
    average = meta.get("average_rating")
    st.markdown(
        f"""
        <div class="rl-product">
          {_photo_html(meta, "rl-product__photo")}
          <div class="rl-product__left">
            <div class="rl-kicker">Live from Amazon ·
              {html_lib.escape(str(meta.get('asin', '')))}</div>
            <div class="rl-product__title">{html_lib.escape(str(meta.get('title', '')))}</div>
            <div class="rl-product__sub">{count_text} · the page's
            {meta.get('fetched_reviews', '—')} public top reviews analyzed below</div>
          </div>
          <div class="rl-product__right">
            <div class="rl-product__avg">{average if average is not None else '—'}<span>/5
            </span></div>
            <div class="rl-product__bars">{_histogram_rows(meta)}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview(reviews: pd.DataFrame, aspects: pd.DataFrame) -> None:
    """The figures row — rules and type, no boxes."""
    n_reviews = int(reviews["review_id"].nunique()) if not reviews.empty else 0
    mixed = int(aspects.groupby("review_id")["aspect_sentiment"].nunique().gt(1).sum())
    mixed_pct = f"{mixed / n_reviews:.0%} of all reviews" if n_reviews else "—"
    figures = [
        ("Reviews", f"{n_reviews:,}", "unique review ids", ""),
        ("Aspect mentions", f"{len(aspects):,}", "sentence-level opinions", ""),
        ("Distinct aspects", f"{aspects['aspect'].nunique():,}", "terms extracted", ""),
        ("Mixed-sentiment reviews", f"{mixed:,}", mixed_pct, " rl-figure--hot"),
    ]
    cells = "".join(
        f"""<div class="rl-figure{hot}">
              <div class="rl-figure__label">{label}</div>
              <div class="rl-figure__num">{num}</div>
              <div class="rl-figure__sub">{sub}</div>
            </div>"""
        for label, num, sub, hot in figures
    )
    st.markdown(f'<div class="rl-figures">{cells}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data plumbing (unchanged logic)
# ---------------------------------------------------------------------------
def _model_options() -> tuple[list[str], list[str], list[str]]:
    """Model choices that are actually available in this environment."""
    cfg = load_config()
    extractors = ["baseline"]
    if resolve_path(cfg["aspects"]["crf_model_path"]).exists():
        extractors.append("crf")
    if resolve_path(cfg["aspects"]["transformer_model_dir"]).exists():
        extractors.append("transformer")

    sentiments = ["baseline"]
    if resolve_path(cfg["sentiment"]["nb_model_path"]).exists():
        sentiments.append(NB_SENTIMENT)
    if importlib.util.find_spec("torch") is not None:
        if resolve_path(cfg["sentiment"]["absa_finetuned_dir"]).exists():
            sentiments.append(FINETUNED_ABSA)
        sentiments.append(PRETRAINED_ABSA)

    themes = ["normalized", "wordnet"]
    if importlib.util.find_spec("sentence_transformers") is not None:
        themes += ["kmeans", "hdbscan"]
    return extractors, sentiments, themes


def _demo_csvs() -> dict[str, str]:
    """Demo/fetched CSVs from download_reviews.py and fetch_amazon.py, if any."""
    demo_dir = resolve_path(load_config()["paths"]["data_raw"]) / "amazon"
    return {f"live: {p.stem}": str(p) for p in sorted(demo_dir.glob("*_reviews.csv"))}


def _product_meta(source: str | None) -> dict | None:
    """Stats saved next to an Amazon-fetched CSV (title, rating, histogram)."""
    if not source:
        return None
    meta_path = pathlib.Path(str(source).replace("_reviews.csv", "_meta.json"))
    if not meta_path.exists():
        return None
    import json

    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


@st.cache_data(show_spinner="Running the pipeline…")
def build_live(
    extractor: str, sentiment: str, clustering: str, anaphora: bool, source: str | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the pipeline on the sample (or a demo CSV) with the selected models."""
    cfg = deepcopy(load_config())
    cfg["aspects"]["extractor"] = extractor
    cfg["aspects"]["anaphora"] = anaphora
    cfg["clustering"]["method"] = clustering
    if sentiment == NB_SENTIMENT:
        cfg["sentiment"]["aspect_model"] = "nb"
    elif sentiment == "baseline":
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


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, dict | None]:
    _side_head("Pipeline")
    processed_exists = (
        resolve_path(load_config()["paths"]["data_processed"]) / "aspects.parquet"
    ).exists()

    demos = _demo_csvs()  # downloaded corpora + live Amazon fetches
    options = ["live sample", *demos] + (["processed outputs"] if processed_exists else [])
    source = st.sidebar.radio("Data", options, index=0, key="data_source")
    meta = _product_meta(demos.get(source))

    if source == "processed outputs":
        reviews, aspects = load_processed()
        return reviews, aspects, None

    extractors, sentiments, themes = _model_options()
    extractor = st.sidebar.selectbox("Aspect extractor", extractors, index=0)
    sentiment = st.sidebar.selectbox("Aspect sentiment", sentiments, index=0)
    clustering = st.sidebar.selectbox("Theme grouping", themes, index=0)
    anaphora = st.sidebar.toggle(
        "Resolve pronouns", value=False,
        help="Aspect-less sentences starting with it/they/this inherit the "
        "review's most recent aspect (heuristic anaphora resolution).",
    )
    if st.sidebar.button("↻ Re-run"):
        build_live.clear()
    reviews, aspects = build_live(extractor, sentiment, clustering, anaphora, demos.get(source))
    return reviews, aspects, meta


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

    render_section(
        "01", f"Sentiment by {group_by}", note,
        "Every bar is one theme's mentions, split by verdict — mixed bars are "
        "the reviews a single score would flatten.",
    )
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
    # 2px paper gaps between stacked segments + rounded, thin marks.
    fig.update_traces(
        marker_line_color=PALETTE["surface"], marker_line_width=2,
        hovertemplate="%{y} · %{customdata[0]}: <b>%{x}</b><extra></extra>",
    )
    fig.update_layout(bargap=0.45, barcornerradius=3)
    _style_fig(fig, height=max(320, 34 * len(order)), legend=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    render_section(
        "02", "The verdict", "net = (pos − neg) ÷ mentions",
        "One number per theme, from −1 (unanimously hated) to +1 (unanimously loved).",
    )
    net = dist.sort_values("net_score")
    fig2 = px.bar(
        net, x="net_score", y=group_by, orientation="h", range_x=[-1.12, 1.12],
        color="net_score",
        color_continuous_scale=[PALETTE["negative"], PALETTE["neutral"], PALETTE["positive"]],
        color_continuous_midpoint=0,
        text=net["net_score"].map(lambda v: f"{v:+.2f}"),
    )
    fig2.update_traces(
        marker_line_color=PALETTE["surface"], marker_line_width=1.5,
        textposition="outside",
        textfont=dict(family=FONT_MONO, size=10.5, color=PALETTE["ink_faint"]),
        cliponaxis=False,
        hovertemplate="%{y}: <b>%{x:+.2f}</b><extra></extra>",
    )
    fig2.update_layout(coloraxis_showscale=False, bargap=0.5, barcornerradius=3)
    _style_fig(fig2, height=max(320, 34 * len(net)))
    fig2.add_vline(x=0, line_color="rgba(25,21,17,0.3)", line_width=1)
    if len(net) >= 2:  # annotate the poles of the ranking
        top, bottom = net.iloc[-1], net.iloc[0]
        for row, label, color, shift in (
            (top, "most loved", PALETTE["positive"], 14),
            (bottom, "most hated", PALETTE["negative"], -14),
        ):
            fig2.add_annotation(
                x=row["net_score"], y=row[group_by], text=label, showarrow=False,
                yshift=shift, xanchor="left" if row["net_score"] >= 0 else "right",
                xshift=8 if row["net_score"] >= 0 else -8,
                font=dict(family=FONT_MONO, size=9.5, color=color),
            )
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    with st.expander("View as table"):
        st.dataframe(dist.reset_index(drop=True), use_container_width=True, hide_index=True)


def _clip(text: str, n: int = 130) -> str:
    text = str(text).strip()
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def render_contradictions(aspects: pd.DataFrame) -> None:
    """Reviews that praise one aspect and damn another — ABSA's reason to exist."""
    if aspects.empty:
        return
    rows = []
    for rid, grp in aspects.groupby("review_id"):
        best = grp.loc[grp["aspect_compound"].idxmax()]
        worst = grp.loc[grp["aspect_compound"].idxmin()]
        if best["aspect_compound"] >= 0.3 and worst["aspect_compound"] <= -0.3:
            rows.append((best["aspect_compound"] - worst["aspect_compound"], rid, best, worst))
    if not rows:
        return
    rows.sort(key=lambda t: -t[0])

    render_section(
        "03", "In one breath",
        f"{len(rows)} contradictory review{'s' if len(rows) != 1 else ''}",
        "The same customer, the same review — opposite verdicts. This is what a "
        "single sentiment score erases.",
    )
    cards = []
    for spread, rid, best, worst in rows[:4]:
        cards.append(
            f"""<div class="rl-contra">
                 <div class="rl-contra__meta">
                   <span>review {html_lib.escape(str(rid))}</span>
                   <span>spread {spread:.2f}</span>
                 </div>
                 <div class="rl-contra__line rl-contra__line--pos">
                   <span class="rl-contra__aspect">{html_lib.escape(str(best['aspect']))}</span>
                   <span class="rl-contra__quote">{html_lib.escape(_clip(best['sentence']))}</span>
                   <span class="rl-contra__score">{best['aspect_compound']:+.2f}</span>
                 </div>
                 <div class="rl-contra__line rl-contra__line--neg">
                   <span class="rl-contra__aspect">{html_lib.escape(str(worst['aspect']))}</span>
                   <span class="rl-contra__quote">{html_lib.escape(_clip(worst['sentence']))}</span>
                   <span class="rl-contra__score">{worst['aspect_compound']:+.2f}</span>
                 </div>
               </div>"""
        )
    st.markdown(f'<div class="rl-contras">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_trend(aspects: pd.DataFrame) -> None:
    trend = sentiment_over_time(aspects, freq="ME")
    if trend.empty:
        return
    render_section(
        "04", "Over time", "mentions per month",
        "How the volume of praise and complaint moves, month by month.",
    )
    long = trend.melt(id_vars="period", var_name="sentiment", value_name="count")
    fig = px.area(
        long, x="period", y="count", color="sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        category_orders={"sentiment": ["positive", "neutral", "negative"]},
    )
    for tr in fig.data:
        base = SENTIMENT_COLORS.get(tr.name, PALETTE["ink_muted"])
        tr.update(line=dict(width=2, color=base), fillcolor=_rgba(base, 0.18))
    fig.update_traces(hovertemplate="%{x|%b %Y} · <b>%{y}</b><extra></extra>")
    _style_fig(fig, height=340, legend=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_quotes(aspects: pd.DataFrame, group_by: str) -> None:
    quotes = representative_quotes(aspects, group_col=group_by, per_side=1)
    if quotes.empty:
        return
    render_section(
        "05", "In their own words", "most polarized quote per group",
        "The sentences behind the numbers — one per theme, strongest verdict first.",
    )

    def _cards(subset: pd.DataFrame, side: str) -> str:
        cards = []
        for _, r in subset.head(6).iterrows():
            cards.append(
                f"""<div class="rl-quote rl-quote--{side}">
                     <p class="rl-quote__text">{html_lib.escape(str(r['sentence']))}</p>
                     <div class="rl-quote__meta">
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
        st.markdown('<div class="rl-colhead rl-colhead--loved">The praise</div>',
                    unsafe_allow_html=True)
        st.markdown(_cards(loved, "loved"), unsafe_allow_html=True)
    with col_r:
        st.markdown('<div class="rl-colhead rl-colhead--hated">The complaints</div>',
                    unsafe_allow_html=True)
        st.markdown(_cards(hated, "hated"), unsafe_allow_html=True)


def render_llm_summary(aspects: pd.DataFrame, group_by: str) -> None:
    """Optional LLM executive summary — Ollama or an OpenAI-compatible endpoint."""
    render_section(
        "06", "The executive summary", "optional · LLM",
        "A model reads the aggregated numbers — never the raw reviews — and "
        "writes the so-what.",
    )
    with st.expander("Ask a model to read the numbers"):
        st.caption(
            "A local Ollama model — or any OpenAI-compatible endpoint from your "
            ".env — receives the stats digest above and returns a short memo "
            "with recommendations. See .env.example for setup."
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
          <div class="rl-footer__rule"></div>
          <div class="rl-footer__rule rl-footer__rule--thick"></div>
          <div class="rl-footer__row">
            <span>ReviewLens</span><span>·</span><span>MIT © 2026 Saud Satopay</span>
            <span class="rl-right">set in Gloock &amp; Instrument · fine-tuned RoBERTa ABSA
            · measured on SemEval-2014</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page 1 — the journal (the analysis)
# ---------------------------------------------------------------------------
def page_journal() -> None:
    render_masthead()
    render_hero()

    reviews, aspects, product_meta = load_data()
    if aspects.empty:
        st.warning("No aspect data available. Run the pipeline first.")
        return

    f = sidebar_filters(aspects)
    filtered = apply_filters(aspects, f)

    if product_meta:
        render_product_strip(product_meta)
    render_overview(reviews, filtered)
    render_distribution(filtered, f["group_by"], f["min_mentions"])
    render_contradictions(filtered)
    render_trend(filtered)
    render_quotes(filtered, f["group_by"])
    render_llm_summary(filtered, f["group_by"])
    render_footer()


# ---------------------------------------------------------------------------
# Page 2 — the procurement desk (Amazon lookup)
# ---------------------------------------------------------------------------
def _amazon_dir() -> pathlib.Path:
    return resolve_path(load_config()["paths"]["data_raw"]) / "amazon"


def _load_case_files() -> list[dict]:
    """Every fetched product's meta, newest first."""
    import json

    metas = []
    for path in sorted(
        _amazon_dir().glob("amazon_*_meta.json"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    ):
        try:
            metas.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return metas


def _analyze_in_journal(asin: str) -> None:
    st.session_state["data_source"] = f"live: amazon_{asin}_reviews"
    st.switch_page(PAGE_JOURNAL)


def render_dossier(meta: dict) -> None:
    """The product, in full: photo, title, brand, price, rating anatomy."""
    asin = html_lib.escape(str(meta.get("asin", "")))
    count = meta.get("ratings_count")
    average = meta.get("average_rating")
    brand = meta.get("brand")
    price = meta.get("price")
    facts = [f"ASIN {asin}"]
    if brand:
        facts.append(html_lib.escape(str(brand)))
    if count:
        facts.append(f"{count:,} ratings")
    facts.append(f"{meta.get('fetched_reviews', '—')} reviews fetched")
    price_html = (
        f'<div class="rl-dossier__price">{html_lib.escape(str(price))}</div>' if price else ""
    )
    st.markdown(
        f"""
        <div class="rl-dossier">
          {_photo_html(meta, "rl-dossier__photo")}
          <div class="rl-dossier__info">
            <div class="rl-kicker">The dossier</div>
            <div class="rl-dossier__title">{html_lib.escape(str(meta.get('title', '')))}</div>
            <div class="rl-dossier__facts">{' · '.join(facts)}</div>
            {price_html}
            <a class="rl-dossier__link" href="{html_lib.escape(str(meta.get('url', '#')))}"
               target="_blank" rel="noopener">view on amazon ↗</a>
          </div>
          <div class="rl-dossier__stats">
            <div class="rl-product__avg">{average if average is not None else '—'}<span>/5
            </span></div>
            <div class="rl-dossier__stars">{_stars(average)}</div>
            <div class="rl-product__bars">{_histogram_rows(meta)}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_fetched_reviews(asin: str) -> None:
    csv_path = _amazon_dir() / f"amazon_{asin}_reviews.csv"
    if not csv_path.exists():
        return
    reviews = pd.read_csv(csv_path)
    render_section(
        "B", "What the page says", f"{len(reviews)} public top reviews",
        "The raw material — every review the product page shows without "
        "signing in, before the pipeline reads it.",
    )
    cols = st.columns(2, gap="large")
    for i, row in reviews.iterrows():
        with cols[i % 2]:
            stars = _stars(row["rating"] if pd.notna(row["rating"]) else None)
            date = "" if pd.isna(row.get("date")) else str(row["date"])
            st.markdown(
                f"""<div class="rl-quote rl-quote--plain">
                     <p class="rl-quote__text">{html_lib.escape(str(row['text']))}</p>
                     <div class="rl-quote__meta">
                       <span class="rl-quote__starline">{stars}</span>
                       <span>{date}</span>
                     </div>
                   </div>""",
                unsafe_allow_html=True,
            )


def render_case_files(metas: list[dict], current_asin: str | None) -> None:
    others = [m for m in metas if m.get("asin") != current_asin]
    if not others:
        return
    render_section(
        "C", "Case files", f"{len(metas)} products on record",
        "Everything fetched so far — reopen a dossier or send it to the journal.",
    )
    columns = st.columns(3, gap="medium")
    for i, meta in enumerate(others):
        asin = str(meta.get("asin", ""))
        with columns[i % 3]:
            title = html_lib.escape(str(meta.get("title", "")))
            st.markdown(
                f"""<div class="rl-casecard">
                     {_photo_html(meta, "rl-casecard__photo")}
                     <div class="rl-casecard__title">{title}</div>
                     <div class="rl-casecard__meta">★ {meta.get('average_rating', '—')} ·
                       {meta.get('fetched_reviews', '—')} reviews ·
                       {html_lib.escape(str(meta.get('price') or ''))}</div>
                   </div>""",
                unsafe_allow_html=True,
            )
            if st.button("Open dossier", key=f"open_{asin}"):
                st.session_state["lookup_asin"] = asin
                st.rerun()


def page_amazon() -> None:
    render_masthead()
    st.markdown(
        """
        <div class="rl-hero rl-hero--single">
          <div class="rl-hero__left">
            <div class="rl-kicker">The procurement desk</div>
            <h1>Point it at any product<span class="rl-dot">.</span></h1>
            <p class="rl-dek">Paste an Amazon link. One polite fetch of the public
            page returns the rating anatomy — average, histogram, count — and its
            top reviews, ready for the full aspect reading.</p>
            <div class="rl-rules">
              <div class="rl-rule"></div><div class="rl-rule rl-rule--thin"></div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("amazon_lookup", border=False):
        col_input, col_button = st.columns([5, 1], gap="small", vertical_alignment="bottom")
        url = col_input.text_input(
            "Amazon product URL", placeholder="https://www.amazon.in/dp/B097JJ2CK6",
        )
        submitted = col_button.form_submit_button("Fetch")

    if submitted and url.strip():
        from reviewlens.data.amazon_live import save_fetch

        try:
            with st.spinner("Fetching the product page…"):
                _, stats = save_fetch(url.strip(), _amazon_dir())
        except (ValueError, RuntimeError, OSError) as exc:
            st.error(str(exc))
        else:
            st.session_state["lookup_asin"] = stats["asin"]

    metas = _load_case_files()
    current_asin = st.session_state.get("lookup_asin") or (
        metas[0].get("asin") if metas else None
    )
    current = next((m for m in metas if m.get("asin") == current_asin), None)

    if current:
        render_section(
            "A", "The product", "fresh from the page",
            "What Amazon's own page discloses about this product, verbatim.",
        )
        render_dossier(current)
        if st.button("Read the full aspect analysis in the journal →"):
            _analyze_in_journal(str(current["asin"]))
        render_fetched_reviews(str(current["asin"]))
    else:
        st.caption("No products fetched yet — paste a link above to open the first dossier.")

    render_case_files(metas, current_asin)
    render_footer()


# ---------------------------------------------------------------------------
PAGE_JOURNAL = st.Page(page_journal, title="The Journal", url_path="journal", default=True)
PAGE_AMAZON = st.Page(page_amazon, title="Amazon Lookup", url_path="amazon")


def _sidebar_shell() -> None:
    """Wordmark + hand-styled navigation, on every page."""
    st.sidebar.markdown(
        """
        <div class="rl-wordmark">Review<br>Lens<span class="rl-dot">.</span></div>
        <div class="rl-wordmark-sub">every aspect, its own verdict</div>
        """,
        unsafe_allow_html=True,
    )
    _side_head("Pages")
    st.sidebar.page_link(PAGE_JOURNAL, label="The Journal")
    st.sidebar.page_link(PAGE_AMAZON, label="Amazon Lookup")


def main() -> None:
    _inject_css()
    navigation = st.navigation([PAGE_JOURNAL, PAGE_AMAZON], position="hidden")
    _sidebar_shell()
    navigation.run()


if __name__ == "__main__":
    main()
else:
    # `streamlit run` executes the module top-to-bottom without __main__.
    main()
