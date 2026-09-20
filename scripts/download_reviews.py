#!/usr/bin/env python
"""Download a real Amazon review dataset and convert it to a ReviewLens CSV.

    python scripts/download_reviews.py                        # software, 500 reviews
    python scripts/download_reviews.py --category appliances
    python scripts/download_reviews.py --limit 0              # keep everything

Files land in data/raw/amazon/ (git-ignored — the corpus has its own license
terms and is not redistributed from this repository). Then point the pipeline
at the CSV:

    python scripts/run_pipeline.py -i data/raw/amazon/software_reviews.csv \
        --extractor transformer --aspect-model absa --clustering kmeans
"""

from __future__ import annotations

import argparse
import pathlib
import sys

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reviewlens.config import load_config, resolve_path  # noqa: E402
from reviewlens.data.amazon import (  # noqa: E402
    AMAZON_CATEGORIES,
    download_amazon,
    parse_amazon_jsonl,
    to_canonical,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--category", default="software", choices=sorted(AMAZON_CATEGORIES),
        help="Which 5-core Amazon category to fetch (default: software).",
    )
    parser.add_argument(
        "--limit", type=int, default=500,
        help="Sample size for the demo CSV; 0 keeps every review (default: 500).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument("--force", action="store_true", help="Re-download the archive.")
    parser.add_argument("--dir", default=None, help="Destination directory override.")
    args = parser.parse_args(argv)

    dest_dir = pathlib.Path(args.dir) if args.dir else (
        resolve_path(load_config()["paths"]["data_raw"]) / "amazon"
    )

    stem, blurb = AMAZON_CATEGORIES[args.category]
    print(f"Fetching {stem}.json.gz ({blurb}) ...")
    archive = download_amazon(args.category, dest_dir, force=args.force)

    raw = parse_amazon_jsonl(archive)
    reviews = to_canonical(raw, limit=args.limit, seed=args.seed)

    csv_path = dest_dir / f"{args.category}_reviews.csv"
    reviews.to_csv(csv_path, index=False)

    print(f"  raw records   : {len(raw)}")
    print(f"  demo reviews  : {len(reviews)}  ->  {csv_path}")
    print("\nNext:")
    print(f"  python scripts/run_pipeline.py -i {csv_path.as_posix()} \\")
    print("      --extractor transformer --aspect-model absa --clustering kmeans")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
