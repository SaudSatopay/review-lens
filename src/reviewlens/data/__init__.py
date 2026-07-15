"""Data layer: ingest -> clean -> sentence-split."""

from reviewlens.data.clean import clean_reviews, clean_text
from reviewlens.data.ingest import load_reviews
from reviewlens.data.split import explode_sentences, split_into_sentences

__all__ = [
    "load_reviews",
    "clean_text",
    "clean_reviews",
    "split_into_sentences",
    "explode_sentences",
]
