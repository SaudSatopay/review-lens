"""Heuristic anaphora resolution — recovering pronoun-borne opinions.

Syllabus Module 5 (discourse, reference resolution), in its lightest useful
form. The pipeline scopes opinions to sentences, so a review like

    "The battery is huge. **It** drains in an hour."

loses the complaint: sentence two extracts no aspect. This resolver applies a
Hobbs-flavored recency heuristic: a sentence that yields no aspect terms and
*starts* with a third-person pronoun inherits the most recent aspect mentioned
earlier in the same review. That is deliberately conservative — only empty
sentences are touched, so every existing extraction stays byte-identical.

Off by default (``aspects.anaphora: false``): published benchmark numbers stay
reproducible. Enable with ``--anaphora`` on the CLI or the dashboard toggle.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

# Third-person pronouns in subject position at the sentence start.
_PRONOUN_RE = re.compile(
    r"^(it|its|it's|they|their|them|this|that|these|those)\b", re.IGNORECASE
)


def resolve_pronoun_aspects(
    review_ids: Iterable,
    sentences: Iterable[str],
    terms_per_sentence: list[list[str]],
) -> list[list[str]]:
    """Fill empty, pronoun-initial sentences with the review's last aspect.

    Inputs are aligned per sentence, in document order (as the pipeline's
    sentence explode produces them). Returns a new terms list; rows that
    already have aspects are returned unchanged.
    """
    resolved = [list(terms) for terms in terms_per_sentence]
    last_aspect: dict = {}  # review_id -> most recent aspect term (recency)

    for i, (rid, sentence) in enumerate(zip(review_ids, sentences, strict=True)):
        if resolved[i]:
            last_aspect[rid] = resolved[i][-1]
        elif rid in last_aspect and _PRONOUN_RE.match(str(sentence).lstrip()):
            resolved[i] = [last_aspect[rid]]
    return resolved
