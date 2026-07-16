"""Fine-tuning: BIO aspect-term extractor + ABSA polarity classifier.

Heavy imports (torch / transformers) happen inside the train entry points, not
at package import, so the baseline install can import this package freely.
"""

from reviewlens.training.bio import (
    ID2LABEL,
    LABEL2ID,
    bio_labels_for_offsets,
    decode_bio_spans,
)

__all__ = ["ID2LABEL", "LABEL2ID", "bio_labels_for_offsets", "decode_bio_spans"]
