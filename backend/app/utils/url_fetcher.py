"""Fetch and extract readable text from a web URL.

Lets users seed a simulation from a website link alongside uploaded PDF/MD/TXT
files. Uses only `requests` plus the stdlib HTML parser (no extra dependencies)
and applies an SSRF guard so a user-supplied URL cannot be used to reach internal
services (Neo4j, Ollama, cloud metadata endpoints, etc.).
"""

from __future__ import annotations

import ipaddress
import re
import socket
from html.parser import HTMLParser
from typing import Dict, List, Optional
from urllib.parse import urlparse

import requests

# Conservative caps for a user-initiated fetch.
MAX_BYTES = 5 * 1024 * 1024  # 5 MB
REQUEST_TIMEOUT = 20  # seconds
USER_AGENT = "MiroFish-Offline/1.0 (+document ingestion)"

# Tags whose inner content is never readable page text. (Not "head": <title> lives
# there and is captured separately; head's script/style/meta carry no body text.)
_SKIP_TAGS = {"script", "style", "noscript", "template", "svg", "nav", "footer"}
# Block-level tags that should introduce a line break in the extracted text.
_BLOCK_TAGS = {
    "p", "div", "br", "li", "tr", "section", "article", "header",
    "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre", "ul", "ol", "table",
}


class _HTMLTextExtractor(HTMLParser):
    """Strip HTML to plain text, skipping non-content tags and capturing <title>."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._in_title = False
        self.title = ""
        self._parts: List[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True
        if tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False
        if tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data):
        if self._skip_depth > 0:
            return
        if self._in_title:
            self.title += data
        if data.strip():
            self._parts.append(data)

    def get_text(self) -> str:
        text = "".join(self._parts)
        # Collapse runs of spaces/tabs and excess blank lines.
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n[ \t]+", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def parse_urls(raw) -> List[str]:
    """Split a free-form string (or list) of URLs into a clean, de-duplicated list."""
    items: List[str] = []
    if isinstance(raw, (list, tuple)):
        candidates = raw
    elif isinstance(raw, str):
        candidates = re.split(r"[\s,]+", raw)
    else:
        return []
    seen = set()
    for candidate in candidates:
        url = (candidate or "").strip()
        if not url:
            continue
        if not re.match(r"^https?://", url, re.IGNORECASE):
            url = "https://" + url
        if url not in seen:
            seen.add(url)
            items.append(url)
    return items


def _assert_public_host(host: str) -> None:
    """Raise ValueError if the host resolves to a private/loopback/link-local address
    (SSRF guard)."""
    if not host:
        raise ValueError("URL has no host")
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve host: {host}") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (
            ip.is_private or ip.is_loopback or ip.is_link_local
            or ip.is_reserved or ip.is_multicast or ip.is_unspecified
        ):
            raise ValueError(f"Refusing to fetch internal/private address for host: {host}")


def _domain_name(url: str) -> str:
    netloc = urlparse(url).netloc
    return netloc or url


def fetch_url(url: str) -> Dict[str, str]:
    """Fetch a URL and return ``{url, title, text}``.

    Supports HTML (stripped to text), plain text / markdown, and PDF (via PyMuPDF).
    Raises ValueError on invalid/blocked URLs or empty content.
    """
    parsed = urlparse(url)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError(f"Only http/https URLs are supported: {url}")
    _assert_public_host(parsed.hostname or "")

    response = requests.get(
        url,
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": USER_AGENT, "Accept": "text/html,text/plain,application/pdf,*/*"},
        stream=True,
        allow_redirects=True,
    )
    response.raise_for_status()

    # Re-validate the final host after any redirects.
    final_host = urlparse(response.url).hostname or ""
    _assert_public_host(final_host)

    content_type = (response.headers.get("Content-Type") or "").lower()

    # Read at most MAX_BYTES.
    chunks: List[bytes] = []
    total = 0
    for chunk in response.iter_content(chunk_size=65536):
        if not chunk:
            continue
        chunks.append(chunk)
        total += len(chunk)
        if total >= MAX_BYTES:
            break
    raw = b"".join(chunks)[:MAX_BYTES]

    title = ""
    if "application/pdf" in content_type or url.lower().endswith(".pdf"):
        text = _extract_pdf_bytes(raw)
    elif "html" in content_type or raw[:512].lstrip().lower().startswith((b"<!doctype html", b"<html")):
        extractor = _HTMLTextExtractor()
        try:
            extractor.feed(raw.decode("utf-8", errors="replace"))
        except Exception:
            pass
        text = extractor.get_text()
        title = (extractor.title or "").strip()
    else:
        text = raw.decode("utf-8", errors="replace").strip()

    if not text or len(text.strip()) < 20:
        raise ValueError(f"No readable text extracted from URL: {url}")

    if not title:
        title = _domain_name(response.url)
    return {"url": response.url, "title": title[:200], "text": text}


def _extract_pdf_bytes(data: bytes) -> str:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover
        raise ValueError("PDF URL ingestion requires PyMuPDF") from exc
    parts: List[str] = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        for page in doc:
            page_text = page.get_text()
            if page_text.strip():
                parts.append(page_text)
    return "\n\n".join(parts)
