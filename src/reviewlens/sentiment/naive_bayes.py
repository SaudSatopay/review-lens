"""Naïve Bayes aspect sentiment — the classical supervised middle rung.

Multinomial Naïve Bayes over bag-of-words features (syllabus Module 4:
supervised classification, the same family as NB word-sense disambiguation).
It sits between VADER (lexicon, no training) and the transformer cross-encoder
(contextual): it *is* trained on SemEval, but a bag of words still cannot scope
"great" to one aspect and not another — measuring it makes that limitation a
number instead of a claim.

The aspect enters the feature space as a single synthetic token
(``aspecttok_battery_life``), so the model can learn aspect priors while the
sentence contributes ordinary uni/bigrams.

Trains in seconds on CPU (no torch): ``python scripts/train_classical.py``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

SENTIMENT_CLASSES = ["negative", "neutral", "positive"]


def _pair_text(sentence: str, aspect: str) -> str:
    marker = "aspecttok_" + "_".join(str(aspect).lower().split())
    return f"{sentence} {marker}"


def train_naive_bayes(pairs: list[tuple[str, str]], labels: list[str], alpha: float = 1.0):
    """Fit CountVectorizer(1-2 grams) -> MultinomialNB. Returns the sklearn Pipeline."""
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline

    model = Pipeline(
        [
            ("vectorizer", CountVectorizer(ngram_range=(1, 2), min_df=2, lowercase=True)),
            ("nb", MultinomialNB(alpha=alpha)),
        ]
    )
    model.fit([_pair_text(s, a) for s, a in pairs], labels)
    return model


class NaiveBayesAspectSentiment:
    """Trained-NB classifier with the same ``predict_batch`` contract as the
    transformer: one ``(label, score)`` per pair, score = P(pos) − P(neg)."""

    def __init__(self, model_path: str | Path):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"No trained Naïve Bayes model at {path}. Train it first "
                "(seconds, CPU-only):\n"
                "    python scripts/download_semeval.py\n"
                "    python scripts/train_classical.py"
            )
        import joblib

        self.model = joblib.load(path)
        self._classes = list(self.model.classes_)

    def predict_batch(self, pairs: list[tuple[str, str]]) -> list[tuple[str, float]]:
        if not pairs:
            return []
        probs = self.model.predict_proba([_pair_text(s, a) for s, a in pairs])
        i_pos = self._classes.index("positive")
        i_neg = self._classes.index("negative")
        out: list[tuple[str, float]] = []
        for row in probs:
            label = self._classes[int(row.argmax())]
            out.append((label, float(row[i_pos] - row[i_neg])))
        return out


@lru_cache(maxsize=1)
def get_nb_model(model_path: str) -> NaiveBayesAspectSentiment:
    """Cached accessor (mirrors the transformer's)."""
    return NaiveBayesAspectSentiment(model_path)
