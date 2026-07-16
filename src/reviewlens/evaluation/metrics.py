"""Metrics for the SemEval-2014 evaluation.

Two tasks, two scorers:

* :func:`extraction_scores` — aspect-term extraction, micro P/R/F1 with
  case-insensitive exact **term-set** matching per sentence. This deliberately
  differs from the official offset-based SemEval scorer: the noun-phrase
  baseline returns de-duplicated lowercase terms with no character offsets, so
  duplicate mentions of the same string in one sentence collapse to one. The
  numbers are therefore comparable to, but not identical with, the official
  scorer's.

* :func:`sentiment_scores` — 3-class aspect-sentiment classification
  (positive / negative / neutral): accuracy, macro-F1, and per-class P/R/F1.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import classification_report

SENTIMENT_LABELS = ("negative", "neutral", "positive")


def _term_sets(df: pd.DataFrame) -> dict[str, set[str]]:
    """Per-sentence sets of normalized (lowercased, stripped) terms."""
    out: dict[str, set[str]] = {}
    if df.empty:
        return out
    for sid, group in df.groupby("sentence_id"):
        terms = {str(t).lower().strip() for t in group["term"] if str(t).strip()}
        if terms:
            out[str(sid)] = terms
    return out


def extraction_scores(gold_terms: pd.DataFrame, pred_terms: pd.DataFrame) -> dict[str, Any]:
    """Micro-averaged P/R/F1 over per-sentence term sets.

    Both frames need ``sentence_id`` and ``term`` columns.
    """
    gold = _term_sets(gold_terms)
    pred = _term_sets(pred_terms)

    tp = fp = fn = 0
    for sid in gold.keys() | pred.keys():
        g = gold.get(sid, set())
        p = pred.get(sid, set())
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "n_gold_terms": int(sum(len(s) for s in gold.values())),
        "n_pred_terms": int(sum(len(s) for s in pred.values())),
        "matching": "case-insensitive exact term-set per sentence (not offset-based)",
    }


def sentiment_scores(
    y_true: list[str],
    y_pred: list[str],
    labels: tuple[str, ...] = SENTIMENT_LABELS,
) -> dict[str, Any]:
    """Accuracy, macro-F1, and per-class P/R/F1 for 3-class aspect sentiment."""
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: {len(y_true)} gold vs {len(y_pred)} predictions")
    if not y_true:
        raise ValueError("no examples to score")

    report = classification_report(
        y_true,
        y_pred,
        labels=list(labels),
        output_dict=True,
        zero_division=0,
    )
    per_class = {
        label: {
            "precision": round(report[label]["precision"], 4),
            "recall": round(report[label]["recall"], 4),
            "f1": round(report[label]["f1-score"], 4),
            "support": int(report[label]["support"]),
        }
        for label in labels
    }
    return {
        "n": len(y_true),
        "accuracy": round(report["accuracy"], 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "per_class": per_class,
    }
