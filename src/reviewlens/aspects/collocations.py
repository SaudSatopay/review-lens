"""PMI collocation mining — Manning & Schütze, applied (syllabus Module 2).

Pointwise mutual information over corpus bigrams:

    PMI(x, y) = log2( P(x, y) / (P(x) · P(y)) )

High-PMI bigrams are word pairs that co-occur far more than chance —
collocations like "battery life", "customer service", "noise cancellation".
A ``min_count`` floor guards against PMI's classic failure mode (rare pairs
score astronomically), per the textbook.

Used by the Amazon fetcher's report and available to any caller; the noun-
phrase extractor's multi-word terms are the grammar-driven cousin of the same
idea.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable

import pandas as pd

_WORD_RE = re.compile(r"[a-z][a-z']+")

# Function words that make grammatical but uninteresting collocates.
_STOPWORDS = {
    "is", "be", "a", "an", "of", "to", "in", "on", "at", "by", "or", "if", "so",
    "the", "and", "for", "with", "that", "this", "was", "are", "but", "not",
    "you", "your", "have", "has", "had", "its", "it's", "very", "from", "they",
    "them", "were", "been", "will", "would", "when", "than", "then", "there",
    "what", "which", "can", "could", "just", "all", "out", "about", "into",
    "after", "before", "because", "while", "over", "only", "also", "more",
    "does", "did", "don't", "doesn't", "didn't", "get", "got", "one", "two",
}


def _tokenize(sentence: str) -> list[str]:
    return _WORD_RE.findall(str(sentence).lower())


def pmi_collocations(
    sentences: Iterable[str],
    min_count: int = 3,
    top_k: int = 25,
) -> pd.DataFrame:
    """Rank the corpus's bigram collocations by PMI.

    Columns: ``term, count, pmi`` — sorted by PMI descending, stopword-headed
    pairs excluded, only bigrams seen at least ``min_count`` times.
    """
    unigrams: Counter[str] = Counter()
    bigrams: Counter[tuple[str, str]] = Counter()
    for sentence in sentences:
        tokens = _tokenize(sentence)
        unigrams.update(tokens)
        bigrams.update(zip(tokens, tokens[1:], strict=False))

    n_uni = sum(unigrams.values())
    n_bi = sum(bigrams.values())
    cols = ["term", "count", "pmi"]
    if not n_bi:
        return pd.DataFrame(columns=cols)

    rows = []
    for (w1, w2), count in bigrams.items():
        if count < min_count or w1 in _STOPWORDS or w2 in _STOPWORDS:
            continue
        p_xy = count / n_bi
        p_x, p_y = unigrams[w1] / n_uni, unigrams[w2] / n_uni
        rows.append({"term": f"{w1} {w2}", "count": count, "pmi": math.log2(p_xy / (p_x * p_y))})

    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values("pmi", ascending=False).head(top_k).reset_index(drop=True)
