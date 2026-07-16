"""Transformer aspect-sentiment classification — true ``(sentence, aspect) -> polarity``.

This is the model the VADER baseline exists to be measured against. VADER scores a
whole *sentence*, so every aspect in "the display is stunning but the battery is a
dealbreaker" receives the same label. A cross-encoder reads the sentence **and the
aspect together**, so it can hold opposite opinions about two aspects in one clause.

Default checkpoint: ``yangheng/deberta-v3-base-absa-v1.1`` (config:
``sentiment.absa_model_name``).

torch / transformers are imported lazily, so the baseline install never pays for
them. Install the extras with ``pip install -e ".[ml]"``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from reviewlens.config import load_config

# Checkpoints label classes inconsistently (Positive / POS / LABEL_2 / ...).
_LABEL_ALIASES = {
    "positive": "positive", "pos": "positive", "label_2": "positive",
    "negative": "negative", "neg": "negative", "label_0": "negative",
    "neutral": "neutral", "neu": "neutral", "label_1": "neutral",
}


def _normalize_label(raw: str) -> str:
    key = str(raw).strip().lower()
    if key in _LABEL_ALIASES:
        return _LABEL_ALIASES[key]
    raise ValueError(f"Unrecognized sentiment label from model: {raw!r}")


class TransformerAspectSentiment:
    """Cross-encoder ABSA classifier.

    Exposes the same ``(label, score)`` contract as the VADER baseline so the two
    are drop-in interchangeable. ``score`` is ``P(positive) - P(negative)`` in
    [-1, 1] — a direct analogue of VADER's compound, which keeps every downstream
    aggregation (rankings, representative quotes) working unchanged.
    """

    def __init__(
        self,
        model_name: str | None = None,
        config: dict[str, Any] | None = None,
        device: str | None = None,
        batch_size: int = 16,
        max_length: int = 256,
    ):
        cfg = config or load_config()
        self.model_name = model_name or cfg["sentiment"]["absa_model_name"]
        self.batch_size = batch_size
        self.max_length = max_length

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise ImportError(
                "Transformer ABSA needs torch + transformers. Install the extras:\n"
                '    pip install -e ".[ml]"   (or: pip install -r requirements.txt)'
            ) from exc

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(self.model_name)
            .to(self.device)
            .eval()
        )
        self._labels = {i: _normalize_label(x) for i, x in self.model.config.id2label.items()}

    def predict_batch(self, pairs: list[tuple[str, str]]) -> list[tuple[str, float]]:
        """Classify ``[(sentence, aspect), ...]`` -> ``[(label, score), ...]``."""
        if not pairs:
            return []

        out: list[tuple[str, float]] = []
        for start in range(0, len(pairs), self.batch_size):
            chunk = pairs[start : start + self.batch_size]
            encoded = self.tokenizer(
                [s for s, _ in chunk],
                [a for _, a in chunk],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            with self._torch.no_grad():
                probs = self.model(**encoded).logits.softmax(dim=-1)

            for row in probs:
                scores = {self._labels[i]: float(p) for i, p in enumerate(row)}
                label = max(scores, key=scores.get)
                signed = scores.get("positive", 0.0) - scores.get("negative", 0.0)
                out.append((label, signed))
        return out

    def predict(self, sentence: str, aspect: str) -> tuple[str, float]:
        """Classify a single ``(sentence, aspect)`` pair."""
        return self.predict_batch([(sentence, aspect)])[0]


@lru_cache(maxsize=2)
def get_absa_model(model_name: str | None = None) -> TransformerAspectSentiment:
    """Load (and cache) the ABSA model so weights load once per process."""
    return TransformerAspectSentiment(model_name=model_name)
