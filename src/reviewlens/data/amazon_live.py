"""Live Amazon product fetch: paste a product link, get reviews + stats.

Fetches ONE public product page (the kind any browser shows without signing
in) and parses what Amazon displays there:

* product title, overall rating, total ratings count,
* the 5→1 star rating histogram (percentages),
* the on-page "top reviews" cards — typically 8–10 full review texts.

That is the honest ceiling of an anonymous fetch: Amazon requires sign-in for
review pagination, so this feature is a *live specimen collector*, not a
crawler — one polite GET per call, personal/educational use. For volume, the
5-core research corpus path (``scripts/download_reviews.py``) is the right
tool.

Both current markup dialects are handled (``review-body`` / ``reviewText``
hook spellings). A bot-check interstitial raises a clear error — retrying
once from a normal residential connection usually resolves it.
"""

from __future__ import annotations

import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

_ASIN_RE = re.compile(r"/(?:dp|gp/product|gp/aw/d)/([A-Z0-9]{10})", re.IGNORECASE)
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-IN,en-US;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
_HISTOGRAM_RE = re.compile(r"(\d+)\s*percent of reviews have (\d) star", re.IGNORECASE)
_BOT_CHECK_MARKERS = ("Robot Check", "Enter the characters you see below", "captcha")
# Accessibility/widget boilerplate Amazon embeds inside review bodies.
_BOILERPLATE_RE = re.compile(
    r"(?:(?:brief|full) content visible, double tap to read (?:full|brief) content\.?"
    r"|the media could not be loaded\.?"
    r"|videos? for this product)",
    re.IGNORECASE,
)


def parse_asin(url: str) -> str:
    """Extract the 10-character ASIN from any Amazon product URL form."""
    match = _ASIN_RE.search(str(url))
    if not match:
        raise ValueError(
            "Could not find an ASIN in that URL. Paste a product link like "
            "https://www.amazon.in/dp/B097JJ2CK6"
        )
    return match.group(1).upper()


def product_url(url: str) -> str:
    """Normalize to the canonical ``https://<host>/dp/<ASIN>`` form."""
    host_match = re.search(r"https?://([^/]*amazon\.[a-z.]+)", str(url), re.IGNORECASE)
    host = host_match.group(1) if host_match else "www.amazon.in"
    return f"https://{host}/dp/{parse_asin(url)}"


def fetch_html(url: str, timeout: int = 30) -> str:
    """One GET with browser-like headers. Raises on bot-check interstitials."""
    request = urllib.request.Request(product_url(url), headers=_HEADERS)
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        html = resp.read().decode("utf-8", errors="replace")
    if any(marker.lower() in html.lower() for marker in _BOT_CHECK_MARKERS):
        raise RuntimeError(
            "Amazon served a bot-check page instead of the product. Wait a "
            "moment and retry (works best from a normal home connection)."
        )
    return html


def _text(node) -> str:
    if not node:
        return ""
    text = _BOILERPLATE_RE.sub(" ", node.get_text(" ", strip=True))
    return re.sub(r"\s+", " ", text).strip()


def _first_float(text: str) -> float | None:
    match = re.search(r"(\d+(?:[.,]\d+)?)", str(text))
    return float(match.group(1).replace(",", ".")) if match else None


def parse_product_page(html: str) -> tuple[dict[str, Any], pd.DataFrame]:
    """Parse a product page -> (stats dict, raw reviews DataFrame).

    Reviews columns: ``rating, title, body, date`` (strings/floats; date may
    be NaT when the locale format is unknown).
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")

    title = _text(soup.select_one("#productTitle"))
    average = _first_float(_text(soup.select_one('[data-hook="rating-out-of-text"]')))
    count_node = soup.select_one("#acrCustomerReviewText")
    ratings_count = None
    if count_node:
        digits = re.sub(r"[^\d]", "", count_node.get("aria-label") or _text(count_node))
        ratings_count = int(digits) if digits else None

    histogram = {
        int(stars): int(pct) for pct, stars in _HISTOGRAM_RE.findall(html)
    }

    rows = []
    for card in soup.select('[data-hook="review"]'):
        star_node = card.select_one(
            '[data-hook="review-star-rating"] .a-icon-alt, '
            '[data-hook="cmps-review-star-rating"] .a-icon-alt'
        )
        body_node = card.select_one(
            '[data-hook="review-body"], [data-hook="reviewText"]'
        )
        title_node = card.select_one(
            '[data-hook="review-title"], [data-hook="reviewTitle"]'
        )
        date_text = _text(card.select_one('[data-hook="review-date"]'))
        # "Reviewed in India on 6 September 2026" -> "6 September 2026"
        date_part = date_text.rsplit(" on ", 1)[-1] if " on " in date_text else date_text
        rows.append(
            {
                "rating": _first_float(_text(star_node) if star_node else ""),
                "title": _text(title_node),
                "body": _text(body_node),
                "date": date_part,
            }
        )

    reviews = pd.DataFrame(rows, columns=["rating", "title", "body", "date"])
    reviews = reviews[reviews["body"].astype(str).str.strip() != ""]

    stats = {
        "title": title,
        "average_rating": average,
        "ratings_count": ratings_count,
        "histogram": histogram,
        "fetched_reviews": len(reviews),
    }
    return stats, reviews.reset_index(drop=True)


def reviews_to_canonical(reviews: pd.DataFrame, asin: str) -> pd.DataFrame:
    """Map parsed review cards to the canonical pipeline schema."""
    text = (reviews["title"].fillna("") + ". " + reviews["body"].fillna("")).str.strip(". ")
    out = pd.DataFrame(
        {
            "review_id": range(1, len(reviews) + 1),
            "text": text,
            "rating": reviews["rating"],
            "date": pd.to_datetime(reviews["date"], errors="coerce", format="mixed").dt.date,
            "product": asin,
        }
    )
    return out[out["text"].astype(str).str.strip() != ""].reset_index(drop=True)


def fetch_product(url: str) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fetch + parse one product URL -> (stats, canonical reviews frame)."""
    asin = parse_asin(url)
    stats, raw = parse_product_page(fetch_html(url))
    stats.update({"asin": asin, "url": product_url(url)})
    return stats, reviews_to_canonical(raw, asin)


def save_fetch(url: str, dest_dir: str | Path) -> tuple[Path, dict[str, Any]]:
    """Fetch a product and persist it like the demo datasets do.

    Writes ``amazon_<asin>_reviews.csv`` (canonical schema — the dashboard
    picks it up as a data source) and ``amazon_<asin>_meta.json`` (title,
    average rating, ratings count, histogram) next to it.
    """
    stats, reviews = fetch_product(url)
    if reviews.empty:
        raise RuntimeError(
            "The product page parsed, but no review texts were found on it "
            "(some listings show none without signing in)."
        )
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    asin = stats["asin"]
    csv_path = dest / f"amazon_{asin}_reviews.csv"
    reviews.to_csv(csv_path, index=False)
    stats["fetched_at"] = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    (dest / f"amazon_{asin}_meta.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    return csv_path, stats
