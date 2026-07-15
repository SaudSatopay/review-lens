"""Configuration loading and path resolution.

All tunables live in ``config.yaml`` at the project root. Modules call
:func:`load_config` rather than hard-coding paths or model names, so the whole
pipeline can be reconfigured from one file.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

import yaml

# project root = .../review-lens  (this file is src/reviewlens/config.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@cache
def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load ``config.yaml`` as a nested dict.

    Result is cached per path. Pass an explicit ``config_path`` to override the
    default (useful in tests).
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(relative: str | Path) -> Path:
    """Resolve a config-relative path against the project root.

    Absolute paths are returned unchanged.
    """
    p = Path(relative)
    return p if p.is_absolute() else (PROJECT_ROOT / p)
