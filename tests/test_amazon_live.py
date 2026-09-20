"""Tests for the live Amazon fetch — parsing only, no network.

The synthetic page mirrors the real markup observed on amazon.in product
pages (data-hook attributes, histogram aria-labels), including the newer
``reviewText``/``reviewTitle`` hook spellings and the accessibility
boilerplate Amazon embeds inside review bodies.
"""

from __future__ import annotations

import pytest

from reviewlens.data.amazon_live import (
    parse_asin,
    parse_product_page,
    product_url,
    reviews_to_canonical,
)

_PAGE = """
<html><body>
<span id="productTitle">  Acme Blender 3000,   1200W </span>
<a id="bylineInfo" href="/stores/x">Visit the Acme Store</a>
<img id="landingImage" src="data:image/png;base64,xxxx"
     data-a-dynamic-image='{"https://m.media-amazon.com/images/I/x._SY355_.jpg":[355,355],"https://m.media-amazon.com/images/I/x._SX466_.jpg":[466,466]}'>
<div id="corePriceDisplay_desktop_feature_div">
  <span class="a-price"><span class="a-offscreen">₹709.00</span></span>
</div>
<span data-hook="rating-out-of-text">4.2 out of 5</span>
<span id="acrCustomerReviewText" aria-label="17,134 Reviews">(17,134)</span>
<a aria-label="58 percent of reviews have 5 stars"></a>
<a aria-label="22 percent of reviews have 4 stars"></a>
<a aria-label="10 percent of reviews have 3 stars"></a>
<a aria-label="3 percent of reviews have 2 stars"></a>
<a aria-label="7 percent of reviews have 1 stars"></a>

<div data-hook="review">
  <i data-hook="review-star-rating"><span class="a-icon-alt">5.0 out of 5 stars</span></i>
  <h5 data-hook="reviewTitle">Great motor</h5>
  <span data-hook="review-date">Reviewed in India on 6 September 2026</span>
  <span data-hook="reviewText">Brief content visible, double tap to read full content.
    The motor is powerful and the jar is huge.</span>
</div>
<div data-hook="review">
  <i data-hook="review-star-rating"><span class="a-icon-alt">2.0 out of 5 stars</span></i>
  <a data-hook="review-title"><span>Lid cracked</span></a>
  <span data-hook="review-date">Reviewed in the United States on March 3, 2026</span>
  <span data-hook="review-body"><span>The lid cracked after a week.</span></span>
</div>
<div data-hook="review">
  <i data-hook="review-star-rating"><span class="a-icon-alt">4.0 out of 5 stars</span></i>
  <h5 data-hook="reviewTitle">Video only</h5>
  <span data-hook="review-date">Reviewed in India on 1 May 2026</span>
  <span data-hook="reviewText">The media could not be loaded.</span>
</div>
</body></html>
"""


def test_parse_asin_accepts_common_url_forms():
    assert parse_asin("https://www.amazon.in/dp/B097JJ2CK6?th=1") == "B097JJ2CK6"
    assert parse_asin("https://amazon.com/gp/product/b08LHTJTBB/ref=x") == "B08LHTJTBB"
    with pytest.raises(ValueError, match="ASIN"):
        parse_asin("https://www.amazon.in/s?k=blender")


def test_product_url_normalizes_and_keeps_host():
    url = product_url("https://www.amazon.in/Some-Name/dp/B097JJ2CK6/ref=sr_1_1?qid=9")
    assert url == "https://www.amazon.in/dp/B097JJ2CK6"
    assert product_url("https://amazon.com/dp/B097JJ2CK6").startswith("https://amazon.com/")


def test_parse_product_page_stats_and_reviews():
    stats, reviews = parse_product_page(_PAGE)

    assert stats["title"] == "Acme Blender 3000, 1200W"
    assert stats["brand"] == "Acme"
    assert stats["price"] == "₹709.00"
    # largest entry of the dynamic-image map, never the base64 src placeholder
    assert stats["image_url"] == "https://m.media-amazon.com/images/I/x._SX466_.jpg"
    assert stats["average_rating"] == 4.2
    assert stats["ratings_count"] == 17134
    assert stats["histogram"] == {5: 58, 4: 22, 3: 10, 2: 3, 1: 7}

    # Third card is boilerplate-only -> dropped; both hook dialects parsed.
    assert len(reviews) == 2
    assert reviews.loc[0, "rating"] == 5.0
    assert "double tap" not in reviews.loc[0, "body"].lower()
    assert reviews.loc[0, "body"].startswith("The motor is powerful")
    assert reviews.loc[1, "title"] == "Lid cracked"
    assert reviews.loc[0, "date"] == "6 September 2026"


def test_reviews_to_canonical_schema():
    _, reviews = parse_product_page(_PAGE)
    out = reviews_to_canonical(reviews, "B00TEST123")

    assert list(out.columns) == ["review_id", "text", "rating", "date", "product"]
    assert list(out["review_id"]) == [1, 2]
    assert out.loc[0, "text"].startswith("Great motor. The motor is powerful")
    assert str(out.loc[0, "date"]) == "2026-09-06"
    assert str(out.loc[1, "date"]) == "2026-03-03"   # US date format parses too
    assert set(out["product"]) == {"B00TEST123"}
