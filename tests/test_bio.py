"""Tests for BIO span alignment and decoding (pure functions, no tokenizer)."""

from __future__ import annotations

from reviewlens.training.bio import (
    B_ID,
    I_ID,
    IGNORE_INDEX,
    O_ID,
    bio_labels_for_offsets,
    decode_bio_spans,
)

#                     [CLS]  The     battery  life      rocks    [SEP]
OFFSETS = [(0, 0), (0, 3), (4, 11), (12, 16), (17, 22), (0, 0)]
SPECIAL = [1, 0, 0, 0, 0, 1]
TEXT = "The battery life rocks"


def test_single_token_span():
    labels = bio_labels_for_offsets(OFFSETS, SPECIAL, [(4, 11)])  # "battery"
    assert labels == [IGNORE_INDEX, O_ID, B_ID, O_ID, O_ID, IGNORE_INDEX]


def test_multi_token_span_gets_b_then_i():
    labels = bio_labels_for_offsets(OFFSETS, SPECIAL, [(4, 16)])  # "battery life"
    assert labels == [IGNORE_INDEX, O_ID, B_ID, I_ID, O_ID, IGNORE_INDEX]


def test_adjacent_spans_get_two_bs():
    # Two distinct gold spans back to back must both start with B, not run into I.
    labels = bio_labels_for_offsets(OFFSETS, SPECIAL, [(4, 11), (12, 16)])
    assert labels == [IGNORE_INDEX, O_ID, B_ID, B_ID, O_ID, IGNORE_INDEX]


def test_token_partially_overlapping_span_is_labelled():
    # Annotation quirk: span starts mid-token -> the overlapping token still gets B.
    labels = bio_labels_for_offsets(OFFSETS, SPECIAL, [(6, 16)])
    assert labels == [IGNORE_INDEX, O_ID, B_ID, I_ID, O_ID, IGNORE_INDEX]


def test_no_spans_is_all_o():
    labels = bio_labels_for_offsets(OFFSETS, SPECIAL, [])
    assert labels == [IGNORE_INDEX, O_ID, O_ID, O_ID, O_ID, IGNORE_INDEX]


def test_decode_roundtrip_multi_token():
    label_ids = [IGNORE_INDEX, O_ID, B_ID, I_ID, O_ID, IGNORE_INDEX]
    assert decode_bio_spans(TEXT, OFFSETS, label_ids) == ["battery life"]


def test_decode_two_spans():
    label_ids = [IGNORE_INDEX, O_ID, B_ID, B_ID, O_ID, IGNORE_INDEX]
    assert decode_bio_spans(TEXT, OFFSETS, label_ids) == ["battery", "life"]


def test_decode_iob_repair_i_without_b_starts_a_span():
    label_ids = [IGNORE_INDEX, O_ID, O_ID, I_ID, O_ID, IGNORE_INDEX]
    assert decode_bio_spans(TEXT, OFFSETS, label_ids) == ["life"]


def test_decode_dedupes_repeated_terms():
    text = "battery vs battery"
    offsets = [(0, 0), (0, 7), (8, 10), (11, 18), (0, 0)]
    label_ids = [IGNORE_INDEX, B_ID, O_ID, B_ID, IGNORE_INDEX]
    assert decode_bio_spans(text, offsets, label_ids) == ["battery"]
