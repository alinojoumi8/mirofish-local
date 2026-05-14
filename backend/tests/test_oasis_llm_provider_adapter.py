from openai.types.chat import ChatCompletion

from scripts.llm_provider_adapter import (
    AnthropicMessagesModel,
    OasisLLMConfig,
    create_oasis_model,
    resolve_oasis_llm_config,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_resolve_config_detects_anthropic_compatible_minimax(monkeypatch):
    monkeypatch.setenv("LLM_DEFAULT_PROVIDER", "minimax")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.minimax.io/anthropic")
    monkeypatch.setenv("LLM_MODEL_NAME", "MiniMax-M2.7")

    config = resolve_oasis_llm_config({})

    assert config.provider == "minimax"
    assert config.transport == "anthropic_messages"
    assert config.api_key == "test-key"
    assert config.base_url == "https://api.minimax.io/anthropic"
    assert config.model == "MiniMax-M2.7"


def test_anthropic_messages_model_posts_messages_and_returns_chat_completion(
    monkeypatch,
):
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse(
            {
                "id": "msg_test",
                "model": "MiniMax-M2.7",
                "usage": {"input_tokens": 8, "output_tokens": 2},
                "content": [
                    {"type": "thinking", "thinking": "internal"},
                    {"type": "text", "text": "Interview answer."},
                ],
            }
        )

    monkeypatch.setattr(
        "scripts.llm_provider_adapter.requests.post",
        fake_post,
    )

    model = AnthropicMessagesModel(
        model_type="MiniMax-M2.7",
        api_key="test-key",
        url="https://api.minimax.io/anthropic",
        timeout=12,
    )

    result = model.run(
        [
            {"role": "system", "content": "You are a legal agent."},
            {"role": "user", "content": "What is your position?"},
        ]
    )

    assert isinstance(result, ChatCompletion)
    assert result.choices[0].message.content == "Interview answer."
    assert captured["url"] == "https://api.minimax.io/anthropic/v1/messages"
    assert captured["headers"]["x-api-key"] == "test-key"
    assert captured["headers"]["anthropic-version"] == "2023-06-01"
    assert captured["json"]["model"] == "MiniMax-M2.7"
    assert captured["json"]["system"] == "You are a legal agent."
    assert captured["json"]["messages"] == [
        {"role": "user", "content": "What is your position?"}
    ]
    assert captured["json"]["max_tokens"] == 4096


def test_create_oasis_model_routes_openai_compatible_to_camel_factory(
    monkeypatch,
):
    calls = {}

    def fake_create(**kwargs):
        calls.update(kwargs)
        return "camel-model"

    monkeypatch.setattr(
        "scripts.llm_provider_adapter.ModelFactory.create",
        fake_create,
    )

    config = OasisLLMConfig(
        provider="openai-compatible",
        transport="openai_compatible",
        api_key="test-key",
        base_url="http://ollama:11434/v1",
        model="qwen2.5:7b",
    )

    model = create_oasis_model(config)

    assert model == "camel-model"
    assert calls["api_key"] == "test-key"
    assert calls["url"] == "http://ollama:11434/v1"
    assert calls["model_type"] == "qwen2.5:7b"


def test_create_oasis_model_routes_anthropic_transport_to_adapter():
    config = OasisLLMConfig(
        provider="minimax",
        transport="anthropic_messages",
        api_key="test-key",
        base_url="https://api.minimax.io/anthropic",
        model="MiniMax-M2.7",
    )

    model = create_oasis_model(config)

    assert isinstance(model, AnthropicMessagesModel)
