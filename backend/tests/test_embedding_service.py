import pytest
import requests

from app.storage.embedding_service import EmbeddingError, EmbeddingService, RETRIEVAL_DOCUMENT


class FakeResponse:
    def __init__(self, data, status_code=200, text=""):
        self._data = data
        self.status_code = status_code
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            error = requests.exceptions.HTTPError(self.text)
            error.response = self
            raise error

    def json(self):
        return self._data


def test_ollama_auto_pulls_missing_model_and_returns_768d(monkeypatch):
    calls = []
    monkeypatch.setattr("app.storage.embedding_service.Config.GRAPH_EMBEDDING_CACHE_ENABLED", False)

    def fake_get(url, timeout):
        calls.append(("get", url))
        return FakeResponse({"models": []})

    def fake_post(url, json, headers=None, timeout=30):
        calls.append(("post", url, json))
        if url.endswith("/api/pull"):
            return FakeResponse({"status": "success"})
        return FakeResponse({"embeddings": [[0.1] * 768 for _ in json["input"]]})

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(requests, "post", fake_post)

    service = EmbeddingService(provider="ollama", model="nomic-embed-text", auto_pull=True)
    assert len(service.embed("hello")) == 768
    assert any(call[1].endswith("/api/pull") for call in calls)


def test_ollama_tagless_model_matches_latest_tag(monkeypatch):
    calls = []
    monkeypatch.setattr("app.storage.embedding_service.Config.GRAPH_EMBEDDING_CACHE_ENABLED", False)

    def fake_get(url, timeout):
        return FakeResponse({"models": [{"name": "nomic-embed-text:latest"}]})

    def fake_post(url, json, headers=None, timeout=30):
        calls.append(url)
        return FakeResponse({"embeddings": [[0.1] * 768 for _ in json["input"]]})

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(requests, "post", fake_post)

    service = EmbeddingService(provider="ollama", model="nomic-embed-text", auto_pull=True)
    assert len(service.embed("hello")) == 768
    assert not any(url.endswith("/api/pull") for url in calls)


def test_gemini_batch_payload_uses_768_dimensions_and_task_type(monkeypatch):
    captured = {}
    monkeypatch.setattr("app.storage.embedding_service.Config.GRAPH_EMBEDDING_CACHE_ENABLED", False)

    def fake_post(url, json, headers=None, timeout=30):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return FakeResponse({"embeddings": [{"values": [0.2] * 768}]})

    monkeypatch.setattr(requests, "post", fake_post)

    service = EmbeddingService(
        provider="gemini",
        model="gemini-embedding-2",
        api_key="test-key",
        dimensions=768,
    )
    assert len(service.embed_batch(["document text"], task_type=RETRIEVAL_DOCUMENT)[0]) == 768
    request = captured["json"]["requests"][0]
    assert request["model"] == "models/gemini-embedding-2"
    assert request["taskType"] == "RETRIEVAL_DOCUMENT"
    assert request["outputDimensionality"] == 768
    assert captured["headers"]["x-goog-api-key"] == "test-key"


def test_embedding_service_rejects_non_index_dimensions():
    with pytest.raises(EmbeddingError, match="Unsupported embedding dimension"):
        EmbeddingService(provider="ollama", dimensions=1536)
