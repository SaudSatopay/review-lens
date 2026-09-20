"""End-to-end baseline pipeline: reviews CSV -> per-aspect sentiment table.

    ingest -> clean -> doc-level VADER -> sentence-split -> aspect extraction ->
    per-aspect sentiment -> theme grouping

Run it::

    python -m reviewlens.pipeline --input data/sample/sample_reviews.csv
    reviewlens --input data/sample/sample_reviews.csv     # console-script alias
"""

from __future__ import annotations

import argparse
from copy import deepcopy
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


def _extract_terms(texts: list[str], cfg: dict[str, Any]) -> list[list[str]]:
    """One aspect-term list per sentence, via the configured extractor.

    ``transformer`` batches all sentences through the fine-tuned BIO tagger,
    ``crf`` through the classical CRF sequence labeler; both are imported
    lazily so the baseline path never pays for them.
    """
    kind = cfg["aspects"].get("extractor", "baseline")
    if kind == "transformer":
        from reviewlens.aspects.absa import get_aspect_extractor

        model_dir = str(resolve_path(cfg["aspects"]["transformer_model_dir"]))
        return get_aspect_extractor(model_dir).extract_batch(texts)
    if kind == "crf":
        from reviewlens.aspects.crf import get_crf_extractor

        model_path = str(resolve_path(cfg["aspects"]["crf_model_path"]))
        return get_crf_extractor(model_path).extract_batch(texts)
    return [extract_aspects(text, cfg) for text in texts]


def _absa_model_name(cfg: dict[str, Any]) -> str:
    """Resolve which ABSA weights to load, honoring ``sentiment.absa_checkpoint``.

    ``pretrained`` -> ``absa_model_name`` (hub checkpoint); ``finetuned`` -> our
    own ``absa_finetuned_dir``, with a train-it-first error if it isn't there.
    """
    s_cfg = cfg["sentiment"]
    if s_cfg.get("absa_checkpoint", "pretrained") == "finetuned":
        path = resolve_path(s_cfg["absa_finetuned_dir"])
        if not path.exists():
            raise FileNotFoundError(
                f"No fine-tuned ABSA classifier at {path}. Train it first:\n"
                "    python scripts/download_semeval.py\n"
                "    python scripts/train_absa.py"
            )
        return str(path)
    return s_cfg["absa_model_name"]


def _score_aspect_rows(rows: list[dict], cfg: dict[str, Any]) -> None:
    """Attach ``aspect_sentiment`` / ``aspect_compound`` to each row, in place.

    Dispatches on ``sentiment.aspect_model``: ``baseline`` scores each aspect with
    VADER on its sentence; ``absa`` runs the transformer cross-encoder, which is
    batched (hence scoring every row at once rather than inside the extract loop).
    """
    if not rows:
        return

    aspect_model = cfg["sentiment"].get("aspect_model", "baseline")
    if aspect_model == "absa":
        # Imported here so the baseline path never requires torch.
        from reviewlens.sentiment.transformer_absa import get_absa_model

        model = get_absa_model(_absa_model_name(cfg))
        preds = model.predict_batch([(r["sentence"], r["aspect"]) for r in rows])
    elif aspect_model == "nb":
        from reviewlens.sentiment.naive_bayes import get_nb_model

        nb = get_nb_model(str(resolve_path(cfg["sentiment"]["nb_model_path"])))
        preds = nb.predict_batch([(r["sentence"], r["aspect"]) for r in rows])
    else:
        preds = [score_aspect_sentiment(r["sentence"], r["aspect"], cfg) for r in rows]

    # strict=True: a model returning the wrong number of predictions should be a
    # loud error, not a silently half-scored table.
    for row, (label, score) in zip(rows, preds, strict=True):
        row["aspect_sentiment"] = label
        row["aspect_compound"] = score


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

    # 5. aspect extraction (noun-phrase baseline, CRF, or fine-tuned BIO tagger)
    terms_per_sentence = _extract_terms(sentences["sentence"].tolist(), cfg)

    # 5b. optional discourse step: pronoun-initial sentences with no aspects
    # inherit the review's most recent aspect ("The battery is huge. It drains…").
    if cfg["aspects"].get("anaphora", False):
        from reviewlens.aspects.anaphora import resolve_pronoun_aspects

        terms_per_sentence = resolve_pronoun_aspects(
            sentences["review_id"].tolist(),
            sentences["sentence"].tolist(),
            terms_per_sentence,
        )
    rows: list[dict] = []
    for row, terms in zip(sentences.itertuples(index=False), terms_per_sentence, strict=True):
        for aspect in terms:
            rows.append(
                {
                    "review_id": row.review_id,
                    "product": row.product,
                    "rating": row.rating,
                    "date": row.date,
                    "sentence_id": row.sentence_id,
                    "sentence": row.sentence,
                    "doc_sentiment": row.doc_sentiment,
                    "doc_compound": row.doc_compound,
                    "aspect": aspect,
                }
            )

    # 6. per-aspect sentiment (VADER baseline or transformer ABSA)
    _score_aspect_rows(rows, cfg)

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

    extractor = cfg["aspects"].get("extractor", "baseline")
    themes = cfg["clustering"].get("method", "normalized")
    print(
        f"\n=== ReviewLens pipeline (extractor={extractor}, "
        f"sentiment={cfg['sentiment']['aspect_model']}, themes={themes}) ==="
    )
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
    parser.add_argument(
        "--aspect-model", choices=["baseline", "nb", "absa"], default=None,
        help="Override sentiment.aspect_model: 'baseline' (VADER), 'nb' (Naive "
        "Bayes), or 'absa' (transformer).",
    )
    parser.add_argument(
        "--extractor", choices=["baseline", "crf", "transformer"], default=None,
        help="Override aspects.extractor: 'baseline' (noun-phrase), 'crf' "
        "(classical sequence labeler), or 'transformer' (BIO fine-tune).",
    )
    parser.add_argument(
        "--anaphora", action="store_true",
        help="Resolve pronoun-initial sentences to the review's most recent "
        "aspect (heuristic discourse step; off by default).",
    )
    parser.add_argument(
        "--absa-checkpoint", choices=["pretrained", "finetuned"], default=None,
        help="Which ABSA weights to use with --aspect-model absa: the pretrained "
        "hub checkpoint or our own fine-tune from scripts/train_absa.py.",
    )
    parser.add_argument(
        "--clustering", choices=["normalized", "wordnet", "kmeans", "hdbscan"], default=None,
        help="Override clustering.method: keyword mapping, WordNet "
        "synonym/hyponym grouping, or MiniLM embedding clusters.",
    )
    parser.add_argument(
        "--n-clusters", type=int, default=None,
        help="Override clustering.n_clusters (kmeans theme count; 8 suits the "
        "sample, real corpora want more).",
    )
    parser.add_argument(
        "--llm-summary", action="store_true",
        help="Also generate an LLM executive summary (needs Ollama or an "
        "OpenAI-compatible endpoint — see .env.example).",
    )
    parser.add_argument("--no-save", action="store_true", help="Skip writing output files.")
    args = parser.parse_args(argv)

    # Deep-copy: load_config() is cached and returns a shared dict.
    cfg = deepcopy(load_config())
    if args.aspect_model:
        cfg["sentiment"]["aspect_model"] = args.aspect_model
    if args.extractor:
        cfg["aspects"]["extractor"] = args.extractor
    if args.absa_checkpoint:
        cfg["sentiment"]["absa_checkpoint"] = args.absa_checkpoint
    if args.anaphora:
        cfg["aspects"]["anaphora"] = True
    if args.clustering:
        cfg["clustering"]["method"] = args.clustering
    if args.n_clusters:
        cfg["clustering"]["n_clusters"] = args.n_clusters

    result = run_pipeline(source=args.input, config=cfg)
    _print_report(result, group_col=args.group_by, cfg=cfg)

    if not args.no_save:
        output_dir = args.output_dir or resolve_path(cfg["paths"]["data_processed"])
        paths = result.save(output_dir)
        print("\nSaved:")
        for name, path in paths.items():
            print(f"  {name:<10} {path}")

    if args.llm_summary:
        # Optional step: a missing local LLM shouldn't undo a finished run.
        from reviewlens.aggregate.llm_summary import generate_summary

        print("\n=== LLM executive summary ===")
        try:
            summary = generate_summary(result.aspects, group_col=args.group_by, config=cfg)
        except RuntimeError as exc:
            print(f"[skipped] {exc}")
        else:
            print(summary)
            out_path = resolve_path("reports") / "executive_summary.md"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(summary + "\n", encoding="utf-8")
            print(f"\nSaved: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
