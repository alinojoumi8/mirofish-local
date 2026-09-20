"""Tests for website-URL ingestion: parsing, SSRF guard, and HTML-to-text."""

from dataclasses import dataclass, field

import pytest

from app.utils.url_fetcher import fetch_url, parse_urls, _assert_public_host, _HTMLTextExtractor


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


@dataclass
class _FakeResponse:
    url: str
    status_code: int
    headers: dict
    body: bytes = b""
    history: list = field(default_factory=list)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size=65536):
        del chunk_size
        yield self.body

    def close(self):
        return None


def test_fetch_url_rejects_private_redirect_before_requesting_it(monkeypatch):
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        if url == "https://public.example/start":
            return _FakeResponse(
                url=url,
                status_code=302,
                headers={"Location": "http://127.0.0.1:11434/api/tags"},
            )
        raise AssertionError(f"private redirect was requested: {url}")

    monkeypatch.setattr("app.utils.url_fetcher._assert_public_host", lambda host: (
        (_ for _ in ()).throw(ValueError("private host")) if host == "127.0.0.1" else None
    ))
    monkeypatch.setattr("app.utils.url_fetcher.requests.get", fake_get)

    with pytest.raises(ValueError, match="private host"):
        fetch_url("https://public.example/start")

    assert [url for url, _ in calls] == ["https://public.example/start"]
    assert calls[0][1]["allow_redirects"] is False


def test_fetch_url_follows_public_redirect_and_preserves_readable_content(monkeypatch):
    responses = {
        "https://public.example/start": _FakeResponse(
            url="https://public.example/start",
            status_code=301,
            headers={"Location": "/article"},
        ),
        "https://public.example/article": _FakeResponse(
            url="https://public.example/article",
            status_code=200,
            headers={"Content-Type": "text/plain"},
            body=b"A legitimate public article with enough readable text.",
        ),
    }

    monkeypatch.setattr("app.utils.url_fetcher._assert_public_host", lambda host: None)
    monkeypatch.setattr(
        "app.utils.url_fetcher.requests.get",
        lambda url, **kwargs: responses[url],
    )

    result = fetch_url("https://public.example/start")

    assert result == {
        "url": "https://public.example/article",
        "title": "public.example",
        "text": "A legitimate public article with enough readable text.",
    }
