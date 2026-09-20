#!/usr/bin/env python
"""Paste an Amazon product link — get its reviews and their stats.

    python scripts/fetch_amazon.py https://www.amazon.in/dp/B097JJ2CK6

Fetches the public product page once, prints the product's rating stats
(average, total ratings, 5→1 star histogram), the on-page top reviews, and the
corpus's PMI collocations — then saves the reviews in the canonical schema so
the pipeline and dashboard can analyze them:

    python scripts/run_pipeline.py -i data/raw/amazon/amazon_<ASIN>_reviews.csv
"""

from __future__ import annotations

import argparse
import pathlib
import sys

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reviewlens.aspects.collocations import pmi_collocations  # noqa: E402
from reviewlens.config import load_config, resolve_path  # noqa: E402
from reviewlens.data.amazon_live import save_fetch  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="Amazon product URL (any /dp/<ASIN> form).")
    parser.add_argument("--dir", default=None, help="Destination directory override.")
    args = parser.parse_args(argv)

    dest_dir = pathlib.Path(args.dir) if args.dir else (
        resolve_path(load_config()["paths"]["data_raw"]) / "amazon"
    )

    csv_path, stats = save_fetch(args.url, dest_dir)

    print(f"\n{stats['title']}")
    print(f"ASIN {stats['asin']}  ·  {stats['url']}")
    if stats.get("average_rating") is not None:
        count = stats.get("ratings_count")
        print(f"\n  average rating : {stats['average_rating']} / 5"
              + (f"  ({count:,} ratings)" if count else ""))
    histogram = stats.get("histogram") or {}
    for stars in sorted(histogram, reverse=True):
        pct = histogram[stars]
        print(f"  {stars} star : {'#' * (pct // 2):<50} {pct}%")
    print(f"\n  fetched reviews : {stats['fetched_reviews']} (the page's public top reviews)")

    import pandas as pd

    reviews = pd.read_csv(csv_path)
    collocations = pmi_collocations(reviews["text"], min_count=2, top_k=8)
    if not collocations.empty:
        print("\n  PMI collocations in these reviews:")
        for _, row in collocations.iterrows():
            print(f"    {row['term']:<28} pmi={row['pmi']:.1f}  (n={int(row['count'])})")

    print(f"\nSaved: {csv_path}")
    print("\nNext:")
    print(f"  python scripts/run_pipeline.py -i {csv_path.as_posix()} --group-by theme")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
