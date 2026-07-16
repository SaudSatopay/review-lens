"""Build tokenized training examples from SemEval-2014 for both fine-tunes.

One multi-domain model per task: examples from the Restaurants and Laptops
*train* splits are pooled (ReviewLens should work on arbitrary product reviews,
not one domain), with a seeded 90/10 train/dev split for epoch-level monitoring.
The official *test* splits are never touched here — final numbers come from
``reviewlens.evaluation.run_eval``.

Everything is padded to ``max_length`` at encode time so the default collator
can batch plain dicts — no dependency on collator classes that shift API
between transformers versions.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from reviewlens.evaluation.semeval import DATASETS, load_semeval
from reviewlens.training.bio import bio_labels_for_offsets

SENTIMENT_LABELS = ["negative", "neutral", "positive"]
SENTIMENT_LABEL2ID = {label: i for i, label in enumerate(SENTIMENT_LABELS)}
SENTIMENT_ID2LABEL = dict(enumerate(SENTIMENT_LABELS))


def _train_dev_split(
    examples: list[dict], seed: int, dev_fraction: float = 0.1
) -> tuple[list[dict], list[dict]]:
    indices = list(range(len(examples)))
    random.Random(seed).shuffle(indices)
    n_dev = max(1, int(len(indices) * dev_fraction))
    dev_idx = set(indices[:n_dev])
    train = [ex for i, ex in enumerate(examples) if i not in dev_idx]
    dev = [ex for i, ex in enumerate(examples) if i in dev_idx]
    return train, dev


def build_extraction_examples(
    tokenizer: Any,
    semeval_dir: str | Path,
    max_length: int = 128,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Token-classification examples: every train sentence, BIO-labelled.

    Sentences without any gold aspect are kept — all-O negatives teach the
    model that most noun phrases are *not* aspects (exactly where the chunker
    baseline loses its precision). Extraction uses all gold terms regardless of
    polarity, conflict included.
    """
    examples: list[dict] = []
    for dataset in DATASETS:
        sentences, terms = load_semeval(dataset, "train", semeval_dir, drop_conflict=False)
        spans_by_sentence: dict[str, list[tuple[int, int]]] = {}
        for row in terms.itertuples(index=False):
            spans_by_sentence.setdefault(row.sentence_id, []).append((row.start, row.end))

        for row in sentences.itertuples(index=False):
            encoded = tokenizer(
                row.text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_offsets_mapping=True,
                return_special_tokens_mask=True,
            )
            labels = bio_labels_for_offsets(
                encoded["offset_mapping"],
                encoded["special_tokens_mask"],
                spans_by_sentence.get(row.sentence_id, []),
            )
            example = {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "labels": labels,
            }
            if "token_type_ids" in encoded:
                example["token_type_ids"] = encoded["token_type_ids"]
            examples.append(example)

    return _train_dev_split(examples, seed)


def build_absa_examples(
    tokenizer: Any,
    semeval_dir: str | Path,
    max_length: int = 128,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Sequence-classification examples: (sentence, aspect) -> polarity.

    Same cross-encoder input format the inference class uses (sentence as text,
    aspect as text pair). Conflict labels are dropped — the standard 3-class
    setup, matching the evaluation.
    """
    examples: list[dict] = []
    for dataset in DATASETS:
        sentences, terms = load_semeval(dataset, "train", semeval_dir, drop_conflict=True)
        text_by_id = dict(
            zip(sentences["sentence_id"], sentences["text"], strict=True)
        )
        for row in terms.itertuples(index=False):
            encoded = tokenizer(
                text_by_id[row.sentence_id],
                row.term,
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
            example = {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "labels": SENTIMENT_LABEL2ID[row.polarity],
            }
            if "token_type_ids" in encoded:
                example["token_type_ids"] = encoded["token_type_ids"]
            examples.append(example)

    return _train_dev_split(examples, seed)
