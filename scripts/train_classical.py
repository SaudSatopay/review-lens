#!/usr/bin/env python
"""Train the classical (no-torch) models on SemEval-2014 train splits.

    python scripts/download_semeval.py     # once
    python scripts/train_classical.py      # CRF extractor + Naive Bayes sentiment

Both train on CPU in seconds and land in models/ (git-ignored):

* CRF aspect extractor   -> models/crf-extractor.joblib   (aspects.extractor: crf)
* Naive Bayes sentiment  -> models/nb-sentiment.joblib    (sentiment.aspect_model: nb)

Score them against the same gold sets as everything else:

    python scripts/evaluate_semeval.py --extractors crf --models nb
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import joblib  # noqa: E402

from reviewlens.aspects.crf import train_crf  # noqa: E402
from reviewlens.config import load_config, resolve_path  # noqa: E402
from reviewlens.evaluation.semeval import DATASETS, load_semeval  # noqa: E402
from reviewlens.sentiment.naive_bayes import train_naive_bayes  # noqa: E402


def _extraction_data(semeval_dir) -> tuple[list[str], list[list[tuple[int, int]]]]:
    sentences_all: list[str] = []
    spans_all: list[list[tuple[int, int]]] = []
    for dataset in DATASETS:
        sentences, terms = load_semeval(dataset, "train", semeval_dir, drop_conflict=False)
        spans_by_sentence: dict[str, list[tuple[int, int]]] = {}
        for row in terms.itertuples(index=False):
            spans_by_sentence.setdefault(row.sentence_id, []).append((row.start, row.end))
        for row in sentences.itertuples(index=False):
            sentences_all.append(row.text)
            spans_all.append(spans_by_sentence.get(row.sentence_id, []))
    return sentences_all, spans_all


def _sentiment_data(semeval_dir) -> tuple[list[tuple[str, str]], list[str]]:
    pairs: list[tuple[str, str]] = []
    labels: list[str] = []
    for dataset in DATASETS:
        sentences, terms = load_semeval(dataset, "train", semeval_dir, drop_conflict=True)
        text_by_id = dict(zip(sentences["sentence_id"], sentences["text"], strict=True))
        for row in terms.itertuples(index=False):
            pairs.append((text_by_id[row.sentence_id], row.term))
            labels.append(row.polarity)
    return pairs, labels


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", default=["crf", "nb"], choices=["crf", "nb"],
        help="Which classical models to train (default: both).",
    )
    args = parser.parse_args(argv)

    cfg = load_config()
    semeval_dir = resolve_path(cfg["eval"]["semeval_dir"])
    models_dir = resolve_path(cfg["paths"]["models"])
    models_dir.mkdir(parents=True, exist_ok=True)

    if "crf" in args.models:
        sentences, spans = _extraction_data(semeval_dir)
        print(f"[crf] training on {len(sentences)} sentences ...")
        started = time.perf_counter()
        crf = train_crf(sentences, spans)
        path = resolve_path(cfg["aspects"]["crf_model_path"])
        joblib.dump(crf, path)
        print(f"[crf] done in {time.perf_counter() - started:.1f}s -> {path}")

    if "nb" in args.models:
        pairs, labels = _sentiment_data(semeval_dir)
        print(f"[nb] training on {len(pairs)} (sentence, aspect) pairs ...")
        started = time.perf_counter()
        nb = train_naive_bayes(pairs, labels)
        path = resolve_path(cfg["sentiment"]["nb_model_path"])
        joblib.dump(nb, path)
        print(f"[nb] done in {time.perf_counter() - started:.1f}s -> {path}")

    print("\nScore them on the gold test sets:")
    print("    python scripts/evaluate_semeval.py --extractors crf --models nb")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
