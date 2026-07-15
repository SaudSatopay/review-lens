"""Per-aspect sentiment scoring.

Baseline strategy: score the *sentence* that mentions the aspect with VADER and
map it to a label. Because we already split reviews into sentences, this gives a
meaningfully per-aspect signal ("great camera" and "terrible battery" live in
different sentences and get opposite labels).

The ``aspect`` argument is unused by the baseline but is part of the signature
on purpose: the transformer slice swaps in a true ``(sentence, aspect) ->
polarity`` classifier (e.g. yangheng/deberta-v3-base-absa-v1.1) behind the same
interface.
"""

from __future__ import annotations

from typing import Any

from reviewlens.config import load_config
from reviewlens.sentiment.vader_baseline import label_sentiment, score_compound


def score_aspect_sentiment(
    sentence: str,
    aspect: str | None = None,  # noqa: ARG001 - part of the shared interface
    config: dict[str, Any] | None = None,
) -> tuple[str, float]:
    """Return ``(label, compound)`` for an aspect mentioned in ``sentence``."""
    cfg = (config or load_config())["sentiment"]
    compound = score_compound(sentence)
    label = label_sentiment(compound, cfg["pos_threshold"], cfg["neg_threshold"])
    return label, compound
