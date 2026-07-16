#!/usr/bin/env python
"""Thin CLI wrapper around reviewlens.evaluation.run_eval.

Run the SemEval-2014 evaluation without installing the package::

    python scripts/evaluate_semeval.py
    python scripts/evaluate_semeval.py --tasks sentiment --models absa
"""

from __future__ import annotations

import pathlib
import sys

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reviewlens.evaluation.run_eval import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
