"""One-shot NLTK data bootstrap.

The baseline aspect extractor and sentence splitter rely on a few NLTK corpora.
NLTK renamed some resources across versions (``punkt`` -> ``punkt_tab``,
``averaged_perceptron_tagger`` -> ``averaged_perceptron_tagger_eng``), so we try
each name defensively and only download what is missing.
"""

from __future__ import annotations

import nltk

# (download_id, resource lookup path) pairs. Newer + older names both listed so
# this works across NLTK releases; missing ones are simply skipped.
_RESOURCES: list[tuple[str, str]] = [
    ("punkt", "tokenizers/punkt"),
    ("punkt_tab", "tokenizers/punkt_tab"),
    ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger"),
    ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
]


def ensure_nltk_data(quiet: bool = True) -> None:
    """Download required NLTK corpora if they are not already present.

    Idempotent and safe to call at import time or from CI.
    """
    for download_id, lookup_path in _RESOURCES:
        try:
            nltk.data.find(lookup_path)
        except LookupError:
            # Some names only exist on certain NLTK versions; ignore failures
            # for the variant that doesn't apply to the installed version.
            try:
                nltk.download(download_id, quiet=quiet)
            except Exception:  # noqa: BLE001 - best-effort across versions
                pass
