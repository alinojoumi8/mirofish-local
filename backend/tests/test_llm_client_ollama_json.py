import pytest
import requests

from app.utils.llm_client import LLMClient


class FakeResponse:
    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        return None

    def json(self):
        return self._body


def test_ollama_json_uses_native_endpoint_with_thinking_disabled(monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured.update({"url": url, **kwargs})
        return FakeResponse({"message": {"content": '{"decisions": []}'}})

    monkeypatch.setattr("app.utils.llm_client.requests.post", fake_post)
    client = LLMClient(
        api_key="ollama",
        base_url="http://localhost:11434/v1",
        model="qwen3.5:4b",
        timeout=17,
    )

    result = client.chat_json(
        [{"role": "user", "content": "Return JSON"}],
        temperature=0,
        max_tokens=321,
        disable_reasoning=True,
    )

    assert result == {"decisions": []}
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["timeout"] == 17
    assert captured["json"]["think"] is False
    assert captured["json"]["format"] == "json"
    assert captured["json"]["stream"] is False
    assert captured["json"]["options"]["num_predict"] == 321


def test_ollama_native_json_propagates_timeout(monkeypatch):
    def timeout(_url, **kwargs):
        assert kwargs["timeout"] == 2
        raise requests.Timeout("native request timed out")

    monkeypatch.setattr("app.utils.llm_client.requests.post", timeout)
    client = LLMClient(
        api_key="ollama",
        base_url="http://localhost:11434/v1",
        model="qwen3.5:4b",
        timeout=2,
    )

    with pytest.raises(requests.Timeout, match="timed out"):
        client.chat_json(
            [{"role": "user", "content": "Return JSON"}],
            disable_reasoning=True,
        )


def test_non_ollama_json_transport_is_unchanged(monkeypatch):
    calls = []
    client = LLMClient(
        api_key="cloud-key",
        base_url="https://api.example.test/v1",
        model="cloud-model",
    )

    def fake_chat(**kwargs):
        calls.append(kwargs)
        return '{"ok": true}'

    monkeypatch.setattr(client, "chat", fake_chat)
    monkeypatch.setattr(
        "app.utils.llm_client.requests.post",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("native path used")),
    )

    assert client.chat_json(
        [{"role": "user", "content": "Return JSON"}],
        disable_reasoning=True,
    ) == {"ok": True}
    assert calls == [{
        "messages": [{"role": "user", "content": "Return JSON"}],
        "temperature": 0.3,
        "max_tokens": 4096,
        "response_format": {"type": "json_object"},
    }]


def test_ollama_native_json_rejects_empty_content(monkeypatch):
    monkeypatch.setattr(
        "app.utils.llm_client.requests.post",
        lambda *_args, **_kwargs: FakeResponse({"message": {"content": ""}}),
    )
    client = LLMClient(
        api_key="ollama",
        base_url="http://localhost:11434/v1",
        model="qwen3.5:4b",
    )

    with pytest.raises(ValueError, match="no JSON content"):
        client.chat_json(
            [{"role": "user", "content": "Return JSON"}],
            disable_reasoning=True,
        )
