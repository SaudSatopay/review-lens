"""Run the SemEval-2014 evaluation and write a results JSON.

Usage (after ``python scripts/download_semeval.py``)::

    python -m reviewlens.evaluation.run_eval                       # everything
    python -m reviewlens.evaluation.run_eval --tasks sentiment --models absa
    python -m reviewlens.evaluation.run_eval --limit 50            # smoke run

Results merge into ``reports/semeval2014_results.json`` (config:
``eval.results_path``), so tasks can be run piecemeal — e.g. the fast VADER
pass first and the slower transformer pass later — without losing anything.
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

# The pretrained checkpoint's own training mix includes SemEval-2014 train data,
# so treat its test scores as an optimistic upper bound, not a fair zero-shot
# number. Recorded into the results JSON so the caveat travels with the numbers.
PRETRAINED_CAVEAT = (
    "The pretrained ABSA checkpoint (yangheng/deberta-v3-base-absa-v1.1) was "
    "trained on a merged ABSA corpus that includes SemEval-2014 train splits; "
    "its test scores are an upper bound rather than a zero-shot measurement."
)


def evaluate_extraction(
    dataset: str,
    cfg: dict[str, Any],
    semeval_dir: Path,
    limit: int | None = None,
) -> dict[str, Any]:
    """Noun-phrase baseline vs gold aspect terms on the test split."""
    sentences, gold = load_semeval(dataset, "test", semeval_dir, drop_conflict=False)
    if limit:
        sentences = sentences.head(limit)
        gold = gold[gold["sentence_id"].isin(sentences["sentence_id"])]

    started = time.perf_counter()
    rows: list[dict] = []
    for row in tqdm(
        sentences.itertuples(index=False),
        total=len(sentences),
        desc=f"extract:{dataset}",
        unit="sent",
    ):
        for term in extract_aspects(row.text, cfg):
            rows.append({"sentence_id": row.sentence_id, "term": term})

    pred = pd.DataFrame(rows, columns=["sentence_id", "term"])
    scores = extraction_scores(gold, pred)
    scores["extractor"] = "noun-phrase baseline"
    scores["n_sentences"] = len(sentences)
    scores["elapsed_s"] = round(time.perf_counter() - started, 1)
    return scores


def _predict_absa(pairs: list[tuple[str, str]], cfg: dict[str, Any]) -> list[str]:
    from reviewlens.sentiment.transformer_absa import get_absa_model

    model = get_absa_model(cfg["sentiment"]["absa_model_name"])
    chunk_size = 4 * cfg["eval"].get("absa_batch_size", 16)
    labels: list[str] = []
    for start in tqdm(range(0, len(pairs), chunk_size), desc="absa", unit="chunk"):
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
    """Score gold (sentence, aspect) pairs with VADER or the ABSA transformer."""
    sentences, terms = load_semeval(
        dataset, "test", semeval_dir, drop_conflict=cfg["eval"].get("drop_conflict", True)
    )
    examples = terms.merge(sentences, on="sentence_id")
    if limit:
        examples = examples.head(limit)

    pairs = list(zip(examples["text"], examples["term"], strict=True))
    y_true = examples["polarity"].tolist()

    started = time.perf_counter()
    if model_kind == "absa":
        y_pred = _predict_absa(pairs, cfg)
    elif model_kind == "baseline":
        y_pred = [
            score_aspect_sentiment(text, term, cfg)[0]
            for text, term in tqdm(pairs, desc=f"vader:{dataset}", unit="pair")
        ]
    else:
        raise ValueError(f"Unknown model kind: {model_kind!r}")

    scores = sentiment_scores(y_true, y_pred)
    scores["model"] = (
        cfg["sentiment"]["absa_model_name"] if model_kind == "absa" else "VADER (sentence-level)"
    )
    scores["elapsed_s"] = round(time.perf_counter() - started, 1)
    return scores


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
        if "extraction" in block:
            e = block["extraction"]
            print(
                f"  extraction (noun-phrase): P={e['precision']:.3f} "
                f"R={e['recall']:.3f} F1={e['f1']:.3f}"
            )
        for kind in ("baseline", "absa"):
            s = block.get("sentiment", {}).get(kind)
            if s:
                print(
                    f"  sentiment/{kind:<8}: acc={s['accuracy']:.3f} "
                    f"macro-F1={s['macro_f1']:.3f}  (n={s['n']})"
                )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate ReviewLens on SemEval-2014 Task 4.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument(
        "--tasks", nargs="+", default=["extraction", "sentiment"],
        choices=["extraction", "sentiment"],
    )
    parser.add_argument(
        "--models", nargs="+", default=["baseline", "absa"], choices=["baseline", "absa"],
        help="Sentiment models to evaluate (ignored for extraction).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap examples (smoke runs).")
    parser.add_argument("--output", default=None, help="Results JSON path override.")
    args = parser.parse_args(argv)

    cfg = load_config()
    semeval_dir = resolve_path(cfg["eval"]["semeval_dir"])
    output = Path(args.output) if args.output else resolve_path(cfg["eval"]["results_path"])

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
            results[dataset]["extraction"] = evaluate_extraction(
                dataset, cfg, semeval_dir, args.limit
            )
        if "sentiment" in args.tasks:
            results[dataset]["sentiment"] = {}
            for kind in args.models:
                results[dataset]["sentiment"][kind] = evaluate_sentiment(
                    dataset, kind, cfg, semeval_dir, args.limit
                )

    _write_results(output, results)
    _print_summary(results)
    print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
