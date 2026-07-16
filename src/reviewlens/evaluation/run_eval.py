"""Run the SemEval-2014 evaluation and write a results JSON.

Usage (after ``python scripts/download_semeval.py``)::

    python scripts/evaluate_semeval.py                     # everything available
    python scripts/evaluate_semeval.py --models baseline absa_pretrained
    python scripts/evaluate_semeval.py --limit 50          # smoke run

Compares, per dataset (Restaurants / Laptops test gold):

* extraction  — ``baseline`` (noun-phrase chunker) vs ``finetuned`` (our BIO
  tagger, if ``models/aspect-extractor`` exists)
* sentiment   — ``baseline`` (VADER) vs ``absa_pretrained`` (yangheng
  checkpoint; upper bound, see caveat) vs ``absa_finetuned`` (our classifier,
  trained on SemEval-2014 train only — the clean number)

By default every evaluator whose model artifact is available runs. Results
merge into ``reports/semeval2014_results.json`` so passes can run piecemeal.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from reviewlens.aspects.baseline import extract_aspects
from reviewlens.config import load_config, resolve_path
from reviewlens.evaluation.metrics import extraction_scores, sentiment_scores
from reviewlens.evaluation.semeval import DATASETS, load_semeval
from reviewlens.sentiment.aspect_sentiment import score_aspect_sentiment

EXTRACTORS = ("baseline", "finetuned")
SENTIMENT_MODELS = ("baseline", "absa_pretrained", "absa_finetuned")

# The pretrained checkpoint's own training mix includes SemEval-2014 train data,
# so treat its test scores as an optimistic upper bound, not a fair zero-shot
# number. Our fine-tuned model (SemEval train only) is the clean measurement.
PRETRAINED_CAVEAT = (
    "The pretrained ABSA checkpoint (yangheng/deberta-v3-base-absa-v1.1) was "
    "trained on a merged ABSA corpus that includes SemEval-2014 train splits; "
    "its test scores are an upper bound rather than a zero-shot measurement. "
    "The absa_finetuned model was trained on the official train splits only."
)


def evaluate_extraction(
    dataset: str,
    extractor_kind: str,
    cfg: dict[str, Any],
    semeval_dir: Path,
    limit: int | None = None,
) -> dict[str, Any]:
    """Score an aspect extractor against gold terms on the test split."""
    sentences, gold = load_semeval(dataset, "test", semeval_dir, drop_conflict=False)
    if limit:
        sentences = sentences.head(limit)
        gold = gold[gold["sentence_id"].isin(sentences["sentence_id"])]

    texts = sentences["text"].tolist()
    started = time.perf_counter()

    if extractor_kind == "finetuned":
        from reviewlens.aspects.absa import get_aspect_extractor

        model_dir = str(resolve_path(cfg["aspects"]["transformer_model_dir"]))
        terms_per_sentence = get_aspect_extractor(model_dir).extract_batch(texts)
        extractor_name = f"fine-tuned BIO tagger ({cfg['training']['base_model']})"
    elif extractor_kind == "baseline":
        terms_per_sentence = [
            extract_aspects(text, cfg)
            for text in tqdm(texts, desc=f"extract:{dataset}", unit="sent")
        ]
        extractor_name = "noun-phrase baseline"
    else:
        raise ValueError(f"Unknown extractor kind: {extractor_kind!r}")

    rows = [
        {"sentence_id": sid, "term": term}
        for sid, terms in zip(sentences["sentence_id"], terms_per_sentence, strict=True)
        for term in terms
    ]
    pred = pd.DataFrame(rows, columns=["sentence_id", "term"])

    scores = extraction_scores(gold, pred)
    scores["extractor"] = extractor_name
    scores["n_sentences"] = len(sentences)
    scores["elapsed_s"] = round(time.perf_counter() - started, 1)
    return scores


def _absa_model_path(model_kind: str, cfg: dict[str, Any]) -> str:
    if model_kind == "absa_pretrained":
        return cfg["sentiment"]["absa_model_name"]
    return str(resolve_path(cfg["sentiment"]["absa_finetuned_dir"]))


def _predict_absa(
    pairs: list[tuple[str, str]], model_kind: str, cfg: dict[str, Any]
) -> list[str]:
    from reviewlens.sentiment.transformer_absa import get_absa_model

    model = get_absa_model(_absa_model_path(model_kind, cfg))
    chunk_size = 4 * cfg["eval"].get("absa_batch_size", 16)
    labels: list[str] = []
    for start in tqdm(range(0, len(pairs), chunk_size), desc=model_kind, unit="chunk"):
        chunk = pairs[start : start + chunk_size]
        labels.extend(label for label, _ in model.predict_batch(chunk))
    return labels


def evaluate_sentiment(
    dataset: str,
    model_kind: str,
    cfg: dict[str, Any],
    semeval_dir: Path,
    limit: int | None = None,
) -> dict[str, Any]:
    """Score gold (sentence, aspect) pairs with one of the sentiment models."""
    sentences, terms = load_semeval(
        dataset, "test", semeval_dir, drop_conflict=cfg["eval"].get("drop_conflict", True)
    )
    examples = terms.merge(sentences, on="sentence_id")
    if limit:
        examples = examples.head(limit)

    pairs = list(zip(examples["text"], examples["term"], strict=True))
    y_true = examples["polarity"].tolist()

    started = time.perf_counter()
    if model_kind in ("absa_pretrained", "absa_finetuned"):
        y_pred = _predict_absa(pairs, model_kind, cfg)
        model_name = _absa_model_path(model_kind, cfg)
    elif model_kind == "baseline":
        y_pred = [
            score_aspect_sentiment(text, term, cfg)[0]
            for text, term in tqdm(pairs, desc=f"vader:{dataset}", unit="pair")
        ]
        model_name = "VADER (sentence-level)"
    else:
        raise ValueError(f"Unknown model kind: {model_kind!r}")

    scores = sentiment_scores(y_true, y_pred)
    scores["model"] = model_name
    scores["elapsed_s"] = round(time.perf_counter() - started, 1)
    return scores


def _available(cfg: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Default extractors / sentiment models: fine-tuned entries only when trained.

    Explicit --extractors/--models requests bypass this (missing artifacts then
    raise, the right failure mode for an explicit ask).
    """
    extractor_dir = resolve_path(cfg["aspects"]["transformer_model_dir"])
    finetuned_absa_dir = resolve_path(cfg["sentiment"]["absa_finetuned_dir"])

    extractors = ["baseline"] + (["finetuned"] if extractor_dir.exists() else [])
    models = ["baseline", "absa_pretrained"] + (
        ["absa_finetuned"] if finetuned_absa_dir.exists() else []
    )
    return extractors, models


def _deep_update(base: dict, new: dict) -> dict:
    for key, value in new.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _write_results(path: Path, results: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    merged = _deep_update(existing, results)
    path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")


def _print_summary(results: dict) -> None:
    print("\n=== SemEval-2014 Task 4 — results ===")
    for dataset in DATASETS:
        block = results.get(dataset)
        if not block:
            continue
        print(f"\n[{dataset}]")
        for kind, scores in block.get("extraction", {}).items():
            print(
                f"  extraction/{kind:<10}: P={scores['precision']:.3f} "
                f"R={scores['recall']:.3f} F1={scores['f1']:.3f}"
            )
        for kind, scores in block.get("sentiment", {}).items():
            print(
                f"  sentiment/{kind:<15}: acc={scores['accuracy']:.3f} "
                f"macro-F1={scores['macro_f1']:.3f}  (n={scores['n']})"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate ReviewLens on SemEval-2014 Task 4.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument(
        "--tasks", nargs="+", default=["extraction", "sentiment"],
        choices=["extraction", "sentiment"],
    )
    parser.add_argument(
        "--extractors", nargs="+", default=None, choices=EXTRACTORS,
        help="Default: baseline, plus finetuned when its model directory exists.",
    )
    parser.add_argument(
        "--models", nargs="+", default=None, choices=SENTIMENT_MODELS,
        help="Default: baseline + absa_pretrained, plus absa_finetuned when it exists.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap examples (smoke runs).")
    parser.add_argument("--output", default=None, help="Results JSON path override.")
    args = parser.parse_args(argv)

    cfg = load_config()
    semeval_dir = resolve_path(cfg["eval"]["semeval_dir"])
    output = Path(args.output) if args.output else resolve_path(cfg["eval"]["results_path"])

    default_extractors, default_models = _available(cfg)
    extractors = args.extractors or default_extractors
    models = args.models or default_models

    results: dict[str, Any] = {
        "meta": {
            "generated_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
            "drop_conflict": cfg["eval"].get("drop_conflict", True),
            "limit": args.limit,
            "caveat": PRETRAINED_CAVEAT,
        }
    }

    for dataset in args.datasets:
        results[dataset] = {}
        if "extraction" in args.tasks:
            results[dataset]["extraction"] = {
                kind: evaluate_extraction(dataset, kind, cfg, semeval_dir, args.limit)
                for kind in extractors
            }
        if "sentiment" in args.tasks:
            results[dataset]["sentiment"] = {
                kind: evaluate_sentiment(dataset, kind, cfg, semeval_dir, args.limit)
                for kind in models
            }

    _write_results(output, results)
    _print_summary(results)
    print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
