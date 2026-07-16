"""Evaluation: SemEval-2014 Task 4 benchmark for extraction + aspect sentiment."""

from reviewlens.evaluation.metrics import extraction_scores, sentiment_scores
from reviewlens.evaluation.semeval import download_semeval, load_semeval, parse_semeval_xml

__all__ = [
    "download_semeval",
    "load_semeval",
    "parse_semeval_xml",
    "extraction_scores",
    "sentiment_scores",
]
