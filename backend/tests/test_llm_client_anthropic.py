import logging

import requests

from app.utils.llm_client import LLMClient


class FailingOpenAI:
    class Chat:
        class Completions:
            @staticmethod
            def create(**kwargs):
                raise AssertionError("OpenAI transport should not be used")

        completions = Completions()

    chat = Chat()


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "content": [
                {"type": "thinking", "thinking": "hidden reasoning"},
                {"type": "text", "text": "OK"},
            ]
        }


class CapturingHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


def capture_logger(name):
    target_logger = logging.getLogger(name)
    handler = CapturingHandler()
    previous_level = target_logger.level
    target_logger.setLevel(logging.DEBUG)
    target_logger.addHandler(handler)
    return target_logger, handler, previous_level


def test_anthropic_transport_posts_messages_and_extracts_text(monkeypatch):
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("app.utils.llm_client.OpenAI", lambda **kwargs: FailingOpenAI())
    monkeypatch.setattr(requests, "post", fake_post)

    client = LLMClient(
        api_key="test-key",
        base_url="https://api.minimax.io/anthropic",
        model="MiniMax-M2.7",
        timeout=12,
    )

    result = client.chat(
        [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "Reply with exactly OK."},
        ],
        max_tokens=32,
        temperature=0,
    )

    assert result == "OK"
    assert captured["url"] == "https://api.minimax.io/anthropic/v1/messages"
    assert captured["headers"]["x-api-key"] == "test-key"
    assert captured["headers"]["anthropic-version"] == "2023-06-01"
    assert captured["json"]["model"] == "MiniMax-M2.7"
    assert captured["json"]["system"] == "You are terse."
    assert captured["json"]["messages"] == [
        {"role": "user", "content": "Reply with exactly OK."}
    ]


def test_anthropic_transport_retries_temporary_dns_failures(monkeypatch):
    attempts = {"count": 0}

    def flaky_post(url, headers, json, timeout):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise requests.exceptions.ConnectionError("temporary DNS failure")
        return FakeResponse()

    monkeypatch.setattr("app.utils.llm_client.OpenAI", lambda **kwargs: FailingOpenAI())
    monkeypatch.setattr(requests, "post", flaky_post)
    monkeypatch.setattr("app.utils.llm_client.time.sleep", lambda _seconds: None)

    client = LLMClient(
        api_key="test-key",
        base_url="https://api.minimax.io/anthropic",
        model="MiniMax-M2.7",
        timeout=12,
    )

    target_logger, handler, previous_level = capture_logger("mirofish.llm_client")
    try:
        result = client.chat([{"role": "user", "content": "Reply OK"}])
    finally:
        target_logger.removeHandler(handler)
        target_logger.setLevel(previous_level)

    assert result == "OK"
    assert attempts["count"] == 2
    assert any(
        record.levelno == logging.WARNING
        and "Anthropic-compatible LLM request failed" in record.getMessage()
        and "attempt 1/5" in record.getMessage()
        and "temporary DNS failure" in record.getMessage()
        for record in handler.records
    )
