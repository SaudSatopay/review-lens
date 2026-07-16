#!/usr/bin/env python
"""Wrapper: fine-tune the ABSA polarity classifier without installing the package."""

from __future__ import annotations

import pathlib
import sys

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reviewlens.training.train_absa import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
