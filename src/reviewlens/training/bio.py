"""BIO tag alignment between character spans and tokenizer offsets.

Pure functions — no torch, no tokenizer objects — so the span/tag logic is unit
testable with hand-written offset mappings. The encoder in
:mod:`reviewlens.training.examples` feeds real tokenizer offsets through these.

Scheme: ``O`` (0), ``B-ASP`` (1), ``I-ASP`` (2); special/padding tokens get
``IGNORE_INDEX`` (-100), which the loss and metrics skip.
"""

from __future__ import annotations

LABELS = ["O", "B-ASP", "I-ASP"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = dict(enumerate(LABELS))

O_ID = LABEL2ID["O"]
B_ID = LABEL2ID["B-ASP"]
I_ID = LABEL2ID["I-ASP"]
IGNORE_INDEX = -100


def _overlapping_span(
    token_start: int, token_end: int, spans: list[tuple[int, int]]
) -> int | None:
    """Index of the first gold span overlapping [token_start, token_end), else None."""
    for i, (span_start, span_end) in enumerate(spans):
        if token_start < span_end and token_end > span_start:
            return i
    return None


def bio_labels_for_offsets(
    offsets: list[tuple[int, int]],
    special_tokens_mask: list[int],
    spans: list[tuple[int, int]],
) -> list[int]:
    """Label ids for each token given gold aspect character spans.

    The first token overlapping a span gets B-ASP; subsequent tokens of the
    same span get I-ASP. Special tokens (and empty offsets, e.g. padding) get
    IGNORE_INDEX.
    """
    labels: list[int] = []
    previous_span: int | None = None
    for (start, end), special in zip(offsets, special_tokens_mask, strict=True):
        if special or start == end:
            labels.append(IGNORE_INDEX)
            previous_span = None
            continue
        span_idx = _overlapping_span(start, end, spans)
        if span_idx is None:
            labels.append(O_ID)
            previous_span = None
        elif span_idx == previous_span:
            labels.append(I_ID)
        else:
            labels.append(B_ID)
            previous_span = span_idx
    return labels


def decode_bio_spans(
    text: str,
    offsets: list[tuple[int, int]],
    label_ids: list[int],
) -> list[str]:
    """Reconstruct aspect terms from predicted per-token label ids.

    Uses character offsets to slice the original text, so subword joining is
    exact. Tolerates I-without-B (IOB2 repair: treats it as a span start).
    Returns lowercase terms, de-duplicated in order of first appearance.
    """
    char_spans: list[list[int]] = []
    current: list[int] | None = None

    for (start, end), label in zip(offsets, label_ids, strict=True):
        if start == end:  # special or padding token
            current = None
            continue
        if label == B_ID:
            current = [start, end]
            char_spans.append(current)
        elif label == I_ID:
            if current is None:  # IOB2 repair
                current = [start, end]
                char_spans.append(current)
            else:
                current[1] = end
        else:
            current = None

    seen: set[str] = set()
    terms: list[str] = []
    for start, end in char_spans:
        term = text[start:end].strip().lower()
        if term and term not in seen:
            seen.add(term)
            terms.append(term)
    return terms
