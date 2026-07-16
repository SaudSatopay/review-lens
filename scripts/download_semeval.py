#!/usr/bin/env python
"""Download the SemEval-2014 Task 4 XML files from public research mirrors.

    python scripts/download_semeval.py                 # everything missing
    python scripts/download_semeval.py --force         # re-download
    python scripts/download_semeval.py --datasets laptops

Files land in data/raw/semeval2014/ (git-ignored — the dataset has its own
license terms and is not redistributed from this repository).
"""

from __future__ import annotations

import argparse
import pathlib
import sys

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reviewlens.config import load_config, resolve_path  # noqa: E402
from reviewlens.evaluation.semeval import DATASETS, SPLITS, download_semeval  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    parser.add_argument("--force", action="store_true", help="Re-download existing files.")
    parser.add_argument("--dir", default=None, help="Destination directory override.")
    args = parser.parse_args(argv)

    dest = args.dir or resolve_path(load_config()["eval"]["semeval_dir"])
    paths = download_semeval(
        dest, datasets=tuple(args.datasets), splits=tuple(args.splits), force=args.force
    )

    print(f"SemEval-2014 files in {dest}:")
    for (dataset, split), path in sorted(paths.items()):
        size_kb = path.stat().st_size / 1024
        print(f"  {dataset:<12} {split:<6} {path.name:<28} {size_kb:8.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
