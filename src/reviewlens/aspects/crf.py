"""CRF aspect extraction — the classical statistical middle rung.

Conditional Random Fields (syllabus Module 3) are the pre-transformer state of
the art for sequence labeling: the same BIO formulation our RoBERTa tagger
uses, but with hand-crafted features (word shape, affixes, POS tags, a ±1
context window) instead of learned representations. Slotting it between the
noun-phrase chunker and the fine-tune turns the extraction benchmark into the
textbook three-rung story — rule-based → statistical → neural — with all three
scored on the same SemEval-2014 gold sets.

Trains in seconds on CPU (no torch): ``python scripts/train_classical.py``.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from reviewlens.nltk_setup import ensure_nltk_data
from reviewlens.training.bio import B_ID, I_ID, O_ID, bio_labels_for_offsets, decode_bio_spans

# Word-ish runs or single punctuation marks, with character offsets — the same
# span convention the transformer tokenizer provides, so the BIO helpers in
# ``training.bio`` are reused unchanged.
_TOKEN_RE = re.compile(r"\w+(?:'\w+)?|[^\w\s]")

_LABELS = ["O", "B-ASP", "I-ASP"]
_ID2LABEL = {O_ID: "O", B_ID: "B-ASP", I_ID: "I-ASP"}
_LABEL2ID = {v: k for k, v in _ID2LABEL.items()}


def tokenize_with_offsets(text: str) -> list[tuple[str, int, int]]:
    """``(token, start, end)`` triples for a sentence."""
    return [(m.group(), m.start(), m.end()) for m in _TOKEN_RE.finditer(str(text))]


def _token_features(tokens: list[str], tags: list[str], i: int) -> dict:
    """Classic CRF feature template: shape + affixes + POS, with ±1 context."""
    word, tag = tokens[i], tags[i]
    feats = {
        "bias": 1.0,
        "word.lower": word.lower(),
        "word.istitle": word.istitle(),
        "word.isupper": word.isupper(),
        "word.isdigit": word.isdigit(),
        "word.suffix3": word[-3:].lower(),
        "word.suffix2": word[-2:].lower(),
        "word.prefix2": word[:2].lower(),
        "pos": tag,
        "pos2": tag[:2],
    }
    if i > 0:
        feats.update(
            {"-1:word.lower": tokens[i - 1].lower(), "-1:pos2": tags[i - 1][:2]}
        )
    else:
        feats["BOS"] = True
    if i < len(tokens) - 1:
        feats.update(
            {"+1:word.lower": tokens[i + 1].lower(), "+1:pos2": tags[i + 1][:2]}
        )
    else:
        feats["EOS"] = True
    return feats


def featurize(text: str) -> tuple[list[dict], list[tuple[int, int]]]:
    """Feature dicts + character offsets for one sentence."""
    ensure_nltk_data()
    from nltk import pos_tag

    triples = tokenize_with_offsets(text)
    if not triples:
        return [], []
    tokens = [t for t, _, _ in triples]
    tags = [tag for _, tag in pos_tag(tokens)]
    feats = [_token_features(tokens, tags, i) for i in range(len(tokens))]
    return feats, [(s, e) for _, s, e in triples]


def bio_tags_for_sentence(text: str, spans: list[tuple[int, int]]) -> list[str]:
    """Gold BIO string labels for a sentence's tokens (training helper)."""
    offsets = [(s, e) for _, s, e in tokenize_with_offsets(text)]
    ids = bio_labels_for_offsets(offsets, [0] * len(offsets), spans)
    return [_ID2LABEL[i] for i in ids]


def train_crf(
    sentences: list[str],
    spans_per_sentence: list[list[tuple[int, int]]],
    max_iterations: int = 100,
    c1: float = 0.1,
    c2: float = 0.1,
):
    """Fit an L-BFGS CRF with elastic-net regularization. Returns the model."""
    import sklearn_crfsuite

    x, y = [], []
    for text, spans in zip(sentences, spans_per_sentence, strict=True):
        feats, _ = featurize(text)
        if not feats:
            continue
        x.append(feats)
        y.append(bio_tags_for_sentence(text, spans))

    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs", c1=c1, c2=c2,
        max_iterations=max_iterations, all_possible_transitions=True,
    )
    crf.fit(x, y)
    return crf


class CRFAspectExtractor:
    """Trained-CRF extractor with the same ``extract_batch`` contract as the
    transformer, so the pipeline can swap it in via ``aspects.extractor: crf``."""

    def __init__(self, model_path: str | Path):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"No trained CRF at {path}. Train it first (seconds, CPU-only):\n"
                "    python scripts/download_semeval.py\n"
                "    python scripts/train_classical.py"
            )
        import joblib

        self.model = joblib.load(path)

    def extract_batch(self, sentences: list[str]) -> list[list[str]]:
        results: list[list[str]] = []
        featurized = [featurize(str(s)) for s in sentences]
        non_empty = [(i, f) for i, (f, _) in enumerate(featurized) if f]
        predictions = (
            self.model.predict([f for _, f in non_empty]) if non_empty else []
        )
        tags_by_index = {i: tags for (i, _), tags in zip(non_empty, predictions, strict=True)}

        for i, sentence in enumerate(sentences):
            tags = tags_by_index.get(i)
            if tags is None or len(tags) == 0:  # crfsuite may hand back arrays
                results.append([])
                continue
            _, offsets = featurized[i]
            ids = [_LABEL2ID[t] for t in tags]
            results.append(decode_bio_spans(str(sentence), offsets, ids))
        return results


@lru_cache(maxsize=1)
def get_crf_extractor(model_path: str) -> CRFAspectExtractor:
    """Cached accessor (mirrors the transformer extractor's)."""
    return CRFAspectExtractor(model_path)
