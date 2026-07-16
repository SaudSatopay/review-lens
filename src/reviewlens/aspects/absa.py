"""Transformer aspect extraction — fine-tuned BIO token classification.

Inference side of :mod:`reviewlens.training.train_extractor`: loads the
fine-tuned token classifier and turns per-token B/I/O predictions back into
aspect-term strings via character offsets (see
:func:`reviewlens.training.bio.decode_bio_spans`).

torch / transformers import lazily inside the class, so the baseline install
never pays for them. Train the model first::

    python scripts/download_semeval.py
    python scripts/train_extractor.py
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from reviewlens.config import load_config, resolve_path
from reviewlens.training.bio import decode_bio_spans


class TransformerAspectExtractor:
    """Fine-tuned BIO aspect-term extractor with batched inference."""

    def __init__(
        self,
        model_dir: str | Path,
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 128,
    ):
        model_path = Path(model_dir)
        if not model_path.exists():
            raise FileNotFoundError(
                f"No fine-tuned extractor at {model_path}. Train it first:\n"
                "    python scripts/download_semeval.py\n"
                "    python scripts/train_extractor.py"
            )

        try:
            import torch
            from transformers import AutoModelForTokenClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise ImportError(
                "The transformer extractor needs torch + transformers. Install:\n"
                '    pip install -e ".[ml]"'
            ) from exc

        self._torch = torch
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model = (
            AutoModelForTokenClassification.from_pretrained(str(model_path))
            .to(self.device)
            .eval()
        )

    def extract_batch(self, sentences: list[str]) -> list[list[str]]:
        """Extract aspect terms for each sentence. Returns one term list per input."""
        results: list[list[str]] = []
        for start in range(0, len(sentences), self.batch_size):
            chunk = [str(s) for s in sentences[start : start + self.batch_size]]
            encoded = self.tokenizer(
                chunk,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            offset_mapping = encoded.pop("offset_mapping")
            encoded = encoded.to(self.device)

            with self._torch.no_grad():
                label_ids = self.model(**encoded).logits.argmax(dim=-1).cpu()

            for text, offsets, labels in zip(
                chunk, offset_mapping.tolist(), label_ids.tolist(), strict=True
            ):
                offset_tuples = [(int(s), int(e)) for s, e in offsets]
                results.append(decode_bio_spans(text, offset_tuples, labels))
        return results

    def extract(self, sentence: str) -> list[str]:
        """Extract aspect terms from a single sentence."""
        return self.extract_batch([sentence])[0]


@lru_cache(maxsize=2)
def get_aspect_extractor(model_dir: str | None = None) -> TransformerAspectExtractor:
    """Load (and cache) the fine-tuned extractor so weights load once per process."""
    if model_dir is None:
        model_dir = str(resolve_path(load_config()["aspects"]["transformer_model_dir"]))
    return TransformerAspectExtractor(model_dir)
