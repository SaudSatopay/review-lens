"""Noun-phrase aspect-extraction baseline.

Classic, dependency-light approach: POS-tag the sentence and chunk maximal noun
sequences (``battery``, ``battery life``, ``customer service``). Adjectives are
intentionally excluded from the aspect term because they carry the *opinion*
("great"), not the *aspect* ("camera").

This is the baseline the fine-tuned BIO token classifier will be benchmarked
against (see aspect-extraction F1 in the evaluation plan).
"""

from __future__ import annotations

from typing import Any

from nltk import pos_tag, word_tokenize
from nltk.chunk import RegexpParser

from reviewlens.config import load_config
from reviewlens.nltk_setup import ensure_nltk_data

# Maximal runs of nouns => candidate aspect terms.
_GRAMMAR = r"NP: {<NN.*>+}"
_CHUNKER = RegexpParser(_GRAMMAR)

# Generic nouns that pass the POS filter but carry no aspect meaning.
_NOISE_TERMS = {
    "thing", "things", "stuff", "lot", "lots", "bit", "everything", "something",
    "anything", "nothing", "one", "ones", "time", "times", "way", "ways", "day",
    "days", "week", "weeks", "month", "months", "year", "years", "today",
    "everyone", "someone", "anyone", "review", "reviews", "product", "item",
    "items", "money", "people", "guy", "guys", "everybody", "nobody",
}


def _is_valid_term(term: str, min_len: int, max_words: int) -> bool:
    if len(term) < min_len:
        return False
    words = term.split()
    if len(words) > max_words:
        return False
    if term in _NOISE_TERMS:
        return False
    # Require at least one word with 2+ alphabetic characters.
    return any(sum(ch.isalpha() for ch in w) >= 2 for w in words)


def extract_aspects(
    sentence: str,
    config: dict[str, Any] | None = None,
) -> list[str]:
    """Extract candidate aspect terms from a single sentence.

    Returns lowercase terms, de-duplicated while preserving first-seen order.
    """
    if not sentence or not str(sentence).strip():
        return []

    cfg = (config or load_config())["aspects"]
    min_len = cfg.get("min_term_length", 3)
    max_words = cfg.get("max_term_words", 4)

    ensure_nltk_data()
    tagged = pos_tag(word_tokenize(str(sentence)))
    tree = _CHUNKER.parse(tagged)

    seen: set[str] = set()
    aspects: list[str] = []
    for subtree in tree.subtrees(filter=lambda t: t.label() == "NP"):
        term = " ".join(w for w, _ in subtree.leaves()).lower().strip()
        if _is_valid_term(term, min_len, max_words) and term not in seen:
            seen.add(term)
            aspects.append(term)
    return aspects
