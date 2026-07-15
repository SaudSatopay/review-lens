"""End-to-end baseline pipeline: reviews CSV -> per-aspect sentiment table.

    ingest -> clean -> doc-level VADER -> sentence-split -> aspect extraction ->
    per-aspect sentiment -> theme grouping

Run it::

    python -m reviewlens.pipeline --input data/sample/sample_reviews.csv
    reviewlens --input data/sample/sample_reviews.csv     # console-script alias
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from reviewlens.aggregate.summary import aspect_distribution, top_hated, top_loved
from reviewlens.aspects.baseline import extract_aspects
from reviewlens.clustering.themes import add_theme_column
from reviewlens.config import load_config, resolve_path
from reviewlens.data.clean import clean_reviews
from reviewlens.data.ingest import load_reviews
from reviewlens.data.split import explode_sentences
from reviewlens.sentiment.aspect_sentiment import score_aspect_sentiment
from reviewlens.sentiment.vader_baseline import document_sentiment

ASPECT_COLUMNS = [
    "review_id", "product", "rating", "date", "sentence_id", "sentence",
    "doc_sentiment", "doc_compound",
    "aspect", "theme", "aspect_sentiment", "aspect_compound",
]


@dataclass
class PipelineResult:
    """Container for the three frames the pipeline produces."""

    reviews: pd.DataFrame     # one row per review, with doc-level VADER baseline
    sentences: pd.DataFrame   # one row per sentence
    aspects: pd.DataFrame     # one row per (sentence, aspect) — the ABSA output

    def aspect_summary(self, group_col: str = "aspect") -> pd.DataFrame:
        """Per-aspect (or per-theme) sentiment distribution."""
        return aspect_distribution(self.aspects, group_col=group_col)

    def save(self, output_dir: str | Path) -> dict[str, Path]:
        """Write frames to ``output_dir`` as parquet. Returns written paths."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = {
            "reviews": out / "reviews.parquet",
            "sentences": out / "sentences.parquet",
            "aspects": out / "aspects.parquet",
        }
        self.reviews.to_parquet(paths["reviews"], index=False)
        self.sentences.to_parquet(paths["sentences"], index=False)
        self.aspects.to_parquet(paths["aspects"], index=False)
        return paths


def run_pipeline(
    source: str | Path | pd.DataFrame | None = None,
    config: dict[str, Any] | None = None,
) -> PipelineResult:
    """Run the full baseline pipeline and return a :class:`PipelineResult`."""
    cfg = config or load_config()

    if source is None:
        source = resolve_path(cfg["paths"]["data_sample"])

    # 1-2. ingest + clean
    reviews = clean_reviews(load_reviews(source, cfg))

    # 3. document-level baseline (VADER) — the comparison point for ABSA
    reviews = document_sentiment(reviews, config=cfg)

    # 4. sentence split
    sentences = explode_sentences(reviews)

    # 5-6. aspect extraction + per-aspect sentiment
    rows: list[dict] = []
    for row in sentences.itertuples(index=False):
        sentence = row.sentence
        for aspect in extract_aspects(sentence, cfg):
            label, compound = score_aspect_sentiment(sentence, aspect, cfg)
            rows.append(
                {
                    "review_id": row.review_id,
                    "product": row.product,
                    "rating": row.rating,
                    "date": row.date,
                    "sentence_id": row.sentence_id,
                    "sentence": sentence,
                    "doc_sentiment": row.doc_sentiment,
                    "doc_compound": row.doc_compound,
                    "aspect": aspect,
                    "aspect_sentiment": label,
                    "aspect_compound": compound,
                }
            )

    aspects = pd.DataFrame(rows)
    # 7. theme grouping (baseline)
    aspects = add_theme_column(aspects, aspect_col="aspect", config=cfg)
    if aspects.empty:
        aspects = pd.DataFrame(columns=ASPECT_COLUMNS)
    else:
        aspects = aspects[ASPECT_COLUMNS]

    return PipelineResult(reviews=reviews, sentences=sentences, aspects=aspects)


def _print_report(result: PipelineResult, group_col: str, cfg: dict[str, Any]) -> None:
    s_cfg = cfg["summary"]
    k, min_mentions = s_cfg["top_k_aspects"], s_cfg["min_mentions"]

    print("\n=== ReviewLens baseline pipeline ===")
    print(f"Reviews processed : {len(result.reviews)}")
    print(f"Sentences         : {len(result.sentences)}")
    print(f"Aspect mentions   : {len(result.aspects)}")

    if result.aspects.empty:
        print("\nNo aspects extracted.")
        return

    loved = top_loved(result.aspects, group_col=group_col, k=k, min_mentions=min_mentions)
    hated = top_hated(result.aspects, group_col=group_col, k=k, min_mentions=min_mentions)

    print(f"\n-- Top loved ({group_col}) --")
    for _, r in loved.iterrows():
        print(f"  {r[group_col]:<18} net={r['net_score']:+.2f}  (n={int(r['mentions'])})")

    print(f"\n-- Top hated ({group_col}) --")
    for _, r in hated.iterrows():
        print(f"  {r[group_col]:<18} net={r['net_score']:+.2f}  (n={int(r['mentions'])})")

    # Show ABSA's value vs the doc-level baseline: reviews whose aspects disagree.
    mixed = (
        result.aspects.groupby("review_id")["aspect_sentiment"].nunique().gt(1).sum()
    )
    print(
        f"\n{mixed} review(s) contain mixed per-aspect sentiment that a single "
        "document-level VADER score would flatten."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the ReviewLens baseline pipeline.")
    parser.add_argument("-i", "--input", default=None, help="Path to a reviews CSV.")
    parser.add_argument(
        "-o", "--output-dir", default=None, help="Where to write processed parquet files."
    )
    parser.add_argument(
        "--group-by", choices=["aspect", "theme"], default="theme",
        help="Grouping level for the printed summary (default: theme).",
    )
    parser.add_argument("--no-save", action="store_true", help="Skip writing output files.")
    args = parser.parse_args(argv)

    cfg = load_config()
    result = run_pipeline(source=args.input, config=cfg)
    _print_report(result, group_col=args.group_by, cfg=cfg)

    if not args.no_save:
        output_dir = args.output_dir or resolve_path(cfg["paths"]["data_processed"])
        paths = result.save(output_dir)
        print("\nSaved:")
        for name, path in paths.items():
            print(f"  {name:<10} {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
