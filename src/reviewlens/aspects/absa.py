"""Transformer aspect extraction (BIO token classification) -- NEXT SLICE.

Placeholder module wired for the fine-tuned aspect extractor. The plan:

* Fine-tune ``deberta-v3-base`` / ``roberta-base`` as a token classifier with
  BIO tags (B-ASP, I-ASP, O) on SemEval-2014 Task 4 (Restaurants + Laptops).
* Evaluate with span-level F1 (``seqeval``) against the noun-phrase baseline in
  :mod:`reviewlens.aspects.baseline`.

Torch / transformers are imported lazily so importing this module never forces
the heavy ML stack on users running only the baseline. Calling the extractor
before that slice is implemented raises a clear error.
"""

from __future__ import annotations

from typing import Any


class TransformerAspectExtractor:
    """Placeholder for the fine-tuned BIO aspect-term extractor."""

    def __init__(self, model_name_or_path: str, config: dict[str, Any] | None = None):
        self.model_name_or_path = model_name_or_path
        self.config = config

    def extract(self, sentence: str) -> list[str]:  # noqa: ARG002
        raise NotImplementedError(
            "Transformer aspect extractor is not implemented yet. "
            "Use reviewlens.aspects.baseline.extract_aspects for the baseline, "
            "or implement the BIO token-classification slice."
        )
