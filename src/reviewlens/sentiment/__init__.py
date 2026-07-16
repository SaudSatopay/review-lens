"""Sentiment scoring: VADER doc-level baseline, per-aspect sentiment, ABSA model.

``transformer_absa`` is safe to import without torch installed — the heavy imports
happen inside :class:`TransformerAspectSentiment.__init__`, not at module load.
"""

from reviewlens.sentiment.aspect_sentiment import score_aspect_sentiment
from reviewlens.sentiment.transformer_absa import (
    TransformerAspectSentiment,
    get_absa_model,
)
from reviewlens.sentiment.vader_baseline import (
    document_sentiment,
    label_sentiment,
    score_compound,
)

__all__ = [
    "document_sentiment",
    "label_sentiment",
    "score_compound",
    "score_aspect_sentiment",
    "TransformerAspectSentiment",
    "get_absa_model",
]
