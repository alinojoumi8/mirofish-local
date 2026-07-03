"""Tests for website-URL ingestion: parsing, SSRF guard, and HTML-to-text."""

import pytest

from app.utils.url_fetcher import parse_urls, _assert_public_host, _HTMLTextExtractor


def test_parse_urls_normalizes_and_dedupes():
    result = parse_urls("example.com\nhttps://foo.com/page, bar.org  https://foo.com/page")
    assert result == ["https://example.com", "https://foo.com/page", "https://bar.org"]


def test_parse_urls_accepts_list():
    assert parse_urls(["https://a.com", "b.com"]) == ["https://a.com", "https://b.com"]


def test_parse_urls_empty():
    assert parse_urls("") == []
    assert parse_urls(None) == []


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "169.254.169.254", "10.0.0.5", "::1", "0.0.0.0"])
def test_ssrf_guard_blocks_internal_hosts(host):
    with pytest.raises(ValueError):
        _assert_public_host(host)


def test_html_extractor_strips_tags_and_captures_title():
    html = (
        "<html><head><title> NovaCity News </title><style>.x{}</style></head>"
        "<body><nav>menu</nav><h1>Plan Approved</h1>"
        "<p>The council voted 7-2 to <b>approve</b> the plan.</p>"
        "<script>evil()</script><p>Opponents warned of costs.</p></body></html>"
    )
    ex = _HTMLTextExtractor()
    ex.feed(html)
    assert ex.title.strip() == "NovaCity News"
    text = ex.get_text()
    assert "Plan Approved" in text
    assert "approve the plan" in text
    assert "Opponents warned of costs" in text
    # Skipped content must not leak through.
    assert "evil()" not in text
    assert "menu" not in text
