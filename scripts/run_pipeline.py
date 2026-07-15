#!/usr/bin/env python
"""Thin CLI wrapper around reviewlens.pipeline.

Lets you run the pipeline without installing the package (adds ``src`` to the
path), e.g.::

    python scripts/run_pipeline.py --input data/sample/sample_reviews.csv
"""

from __future__ import annotations

import pathlib
import sys

# Allow running straight from a clone without `pip install -e .`
SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reviewlens.pipeline import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
