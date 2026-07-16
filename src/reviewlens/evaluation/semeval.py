"""SemEval-2014 Task 4 (ABSA) data: download, parse, load.

The official distribution lives behind MetaShare registration, which is
frequently unavailable, so :func:`download_semeval` pulls the standard XML files
from public research mirrors on GitHub (several well-cited ABSA repos ship
them). Files land in ``data/raw/semeval2014/`` and are git-ignored — the data
carries its own license terms and is not redistributed from this repo.

XML shape (per sentence)::

    <sentence id="813">
      <text>All the appetizers and salads were fabulous.</text>
      <aspectTerms>
        <aspectTerm term="appetizers" polarity="positive" from="8" to="18"/>
      </aspectTerms>
    </sentence>
"""

from __future__ import annotations

import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

DATASETS = ("restaurants", "laptops")
SPLITS = ("train", "test")
POLARITIES = {"positive", "negative", "neutral", "conflict"}

# Official file names from the SemEval-2014 distribution.
OFFICIAL_FILES: dict[tuple[str, str], str] = {
    ("restaurants", "train"): "Restaurants_Train_v2.xml",
    ("restaurants", "test"): "Restaurants_Test_Gold.xml",
    ("laptops", "train"): "Laptop_Train_v2.xml",
    ("laptops", "test"): "Laptops_Test_Gold.xml",
}

# Public research mirrors, tried in order. mem_absa carries all four files;
# the others are fallbacks for the subsets they host.
_GH = "https://raw.githubusercontent.com"
MIRRORS: dict[tuple[str, str], list[str]] = {
    ("restaurants", "train"): [
        f"{_GH}/ganeshjawahar/mem_absa/master/data/Restaurants_Train_v2.xml",
        f"{_GH}/davidsbatista/Aspect-Based-Sentiment-Analysis/master/datasets/ABSA-SemEval2014/Restaurants_Train_v2.xml",
    ],
    ("restaurants", "test"): [
        f"{_GH}/ganeshjawahar/mem_absa/master/data/Restaurants_Test_Gold.xml",
        f"{_GH}/HSLCY/ABSA-BERT-pair/master/data/semeval2014/Restaurants_Test_Gold.xml",
    ],
    ("laptops", "train"): [
        f"{_GH}/ganeshjawahar/mem_absa/master/data/Laptop_Train_v2.xml",
        f"{_GH}/davidsbatista/Aspect-Based-Sentiment-Analysis/master/datasets/ABSA-SemEval2014/Laptop_Train_v2.xml",
    ],
    ("laptops", "test"): [
        f"{_GH}/ganeshjawahar/mem_absa/master/data/Laptops_Test_Gold.xml",
    ],
}

_MIN_PLAUSIBLE_BYTES = 50_000  # all four files are >100KB; catches HTML error pages


def _fetch(url: str, dest: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "reviewlens-eval"})
    with urllib.request.urlopen(request, timeout=60) as resp:
        dest.write_bytes(resp.read())


def download_semeval(
    dest_dir: str | Path,
    datasets: tuple[str, ...] = DATASETS,
    splits: tuple[str, ...] = SPLITS,
    force: bool = False,
) -> dict[tuple[str, str], Path]:
    """Download SemEval-2014 XML files that are missing from ``dest_dir``.

    Tries each mirror in order and validates the payload is real XML with a
    ``<sentences>`` root before accepting it. Returns ``{(dataset, split): path}``.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    out: dict[tuple[str, str], Path] = {}

    for dataset in datasets:
        for split in splits:
            key = (dataset, split)
            path = dest / OFFICIAL_FILES[key]
            out[key] = path
            if path.exists() and not force:
                continue

            errors: list[str] = []
            for url in MIRRORS[key]:
                try:
                    _fetch(url, path)
                    if path.stat().st_size < _MIN_PLAUSIBLE_BYTES:
                        raise ValueError(f"implausibly small file ({path.stat().st_size} bytes)")
                    if ET.parse(path).getroot().tag != "sentences":
                        raise ValueError("root tag is not <sentences>")
                    break
                except Exception as exc:  # noqa: BLE001 - collect and try next mirror
                    errors.append(f"{url} -> {exc}")
                    path.unlink(missing_ok=True)
            else:
                raise RuntimeError(
                    f"Could not download {OFFICIAL_FILES[key]} from any mirror:\n  "
                    + "\n  ".join(errors)
                )
    return out


def parse_semeval_xml(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse one SemEval XML file.

    Returns ``(sentences, terms)``:
        sentences -- sentence_id, text            (every sentence, even term-less)
        terms     -- sentence_id, term, polarity, start, end
    """
    root = ET.parse(path).getroot()
    sentence_rows: list[dict] = []
    term_rows: list[dict] = []

    for sentence in root.iter("sentence"):
        sid = sentence.get("id")
        text = sentence.findtext("text") or ""
        sentence_rows.append({"sentence_id": sid, "text": text})

        container = sentence.find("aspectTerms")
        if container is None:
            continue
        for term in container.findall("aspectTerm"):
            polarity = term.get("polarity")
            if polarity is not None and polarity not in POLARITIES:
                raise ValueError(f"Unexpected polarity {polarity!r} in {path}")
            term_rows.append(
                {
                    "sentence_id": sid,
                    "term": term.get("term"),
                    "polarity": polarity,
                    "start": int(term.get("from")),
                    "end": int(term.get("to")),
                }
            )

    sentences = pd.DataFrame(sentence_rows, columns=["sentence_id", "text"])
    terms = pd.DataFrame(term_rows, columns=["sentence_id", "term", "polarity", "start", "end"])
    return sentences, terms


def load_semeval(
    dataset: str,
    split: str,
    semeval_dir: str | Path,
    drop_conflict: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load one dataset/split from ``semeval_dir`` as ``(sentences, terms)``.

    ``drop_conflict=True`` removes gold terms labelled ``conflict`` (the standard
    3-class evaluation setup used by virtually all ABSA papers).
    """
    key = (dataset, split)
    if key not in OFFICIAL_FILES:
        raise ValueError(f"Unknown dataset/split {key}. Valid: {sorted(OFFICIAL_FILES)}")

    path = Path(semeval_dir) / OFFICIAL_FILES[key]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the data first:\n"
            "    python scripts/download_semeval.py"
        )

    sentences, terms = parse_semeval_xml(path)
    if drop_conflict and not terms.empty:
        terms = terms[terms["polarity"] != "conflict"].reset_index(drop=True)
    return sentences, terms
