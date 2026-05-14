"""
Provider adapter for OASIS/CAMEL simulation models.

CAMEL's OpenAI backend only works with OpenAI-compatible chat-completions
endpoints. MiniMax and Kimi coding endpoints used by this app expose an
Anthropic Messages-compatible API, so OASIS needs a small model backend that
speaks that transport while still returning CAMEL/OpenAI-shaped responses.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

import requests
from camel.models import BaseModelBackend, ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.utils import BaseTokenCounter, OpenAITokenCounter
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai import AsyncStream, Stream
from pydantic import BaseModel


ANTHROPIC_COMPATIBLE_PROVIDERS = {"anthropic", "minimax", "kimi"}


@dataclass(frozen=True)
class OasisLLMConfig:
    provider: str
    transport: str
    api_key: str
    base_url: str
    model: str
    timeout: float = 180.0
    max_retries: int = 3


def _normalise_base_url(base_url: str) -> str:
    base_url = (base_url or "").strip()
    if base_url.endswith("/v1/messages"):
        return base_url[: -len("/v1/messages")]
    if base_url.endswith("/v1/chat/completions"):
        return base_url[: -len("/chat/completions")]
    return base_url.rstrip("/")


def _infer_provider(base_url: str, model: str) -> str:
    value = f"{base_url} {model}".lower()
    if "minimax" in value:
        return "minimax"
    if "kimi" in value or "moonshot" in value:
        return "kimi"
    if "anthropic" in value or "claude" in value:
        return "anthropic"
    if "ollama" in value or "11434" in value:
        return "ollama"
    return "openai-compatible"


def _infer_transport(provider: str, base_url: str) -> str:
    provider = (provider or "").lower()
    base_url = (base_url or "").lower().rstrip("/")

    explicit = os.environ.get("LLM_TRANSPORT", "").strip().lower()
    if explicit:
        return explicit

    if provider in ANTHROPIC_COMPATIBLE_PROVIDERS:
        return "anthropic_messages"
    if "anthropic" in base_url or base_url.endswith("/coding"):
        return "anthropic_messages"
    return "openai_compatible"


def resolve_oasis_llm_config(
    config: Dict[str, Any],
    *,
    use_boost: bool = False,
) -> OasisLLMConfig:
    """Resolve the configured model provider for an OASIS simulation."""
    if use_boost and os.environ.get("LLM_BOOST_API_KEY"):
        api_key = os.environ.get("LLM_BOOST_API_KEY", "")
        base_url = os.environ.get("LLM_BOOST_BASE_URL", "")
        model = os.environ.get("LLM_BOOST_MODEL_NAME") or os.environ.get(
            "LLM_MODEL_NAME", ""
        )
        provider = os.environ.get("LLM_BOOST_PROVIDER", "")
        transport = os.environ.get("LLM_BOOST_TRANSPORT", "")
    else:
        api_key = os.environ.get("LLM_API_KEY") or os.environ.get(
            "OPENAI_API_KEY", ""
        )
        base_url = (
            os.environ.get("LLM_BASE_URL")
            or os.environ.get("OPENAI_API_BASE_URL")
            or config.get("llm_base_url", "")
        )
        model = os.environ.get("LLM_MODEL_NAME", "")
        provider = os.environ.get("LLM_DEFAULT_PROVIDER", "")
        transport = os.environ.get("LLM_TRANSPORT", "")

    model = model or config.get("llm_model", "gpt-4o-mini")
    base_url = _normalise_base_url(base_url)
    provider = (provider or _infer_provider(base_url, model)).lower()
    transport = (transport or _infer_transport(provider, base_url)).lower()

    if not api_key:
        raise ValueError("Missing API key: set LLM_API_KEY or OPENAI_API_KEY")

    return OasisLLMConfig(
        provider=provider,
        transport=transport,
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout=float(os.environ.get("MODEL_TIMEOUT", "180")),
        max_retries=int(os.environ.get("LLM_MAX_RETRIES", "3")),
    )


def create_oasis_model_from_env(
    config: Dict[str, Any],
    *,
    use_boost: bool = False,
):
    return create_oasis_model(resolve_oasis_llm_config(config, use_boost=use_boost))


def create_oasis_model(config: OasisLLMConfig):
    """Create a CAMEL model backend for the configured provider transport."""
    if config.transport == "anthropic_messages":
        print(
            f"[{config.provider}] model={config.model}, "
            f"transport=anthropic_messages, base_url={config.base_url[:40]}..."
        )
        return AnthropicMessagesModel(
            model_type=config.model,
            api_key=config.api_key,
            url=config.base_url,
            timeout=config.timeout,
            max_retries=config.max_retries,
        )

    print(
        f"[{config.provider}] model={config.model}, "
        f"transport=openai_compatible, base_url={config.base_url[:40]}..."
    )
    return ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
        model_type=config.model,
        model_config_dict={},
        api_key=config.api_key,
        url=config.base_url or None,
        timeout=config.timeout,
        max_retries=config.max_retries,
    )


class AnthropicMessagesModel(BaseModelBackend):
    """CAMEL backend for Anthropic Messages-compatible providers."""

    def __init__(
        self,
        model_type: str,
        model_config_dict: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        token_counter: Optional[BaseTokenCounter] = None,
        timeout: Optional[float] = None,
        max_retries: int = 3,
        **_: Any,
    ) -> None:
        default_config = {
            "max_tokens": int(os.environ.get("LLM_MAX_TOKENS", "4096"))
        }
        default_config.update(model_config_dict or {})
        super().__init__(
            model_type=model_type,
            model_config_dict=default_config,
            api_key=api_key,
            url=_normalise_base_url(url or ""),
            token_counter=token_counter,
            timeout=timeout or 180,
            max_retries=max_retries,
        )
        if not self._api_key:
            raise ValueError("Anthropic Messages transport requires an API key")
        if not self._url:
            raise ValueError("Anthropic Messages transport requires a base URL")

    @property
    def token_counter(self) -> BaseTokenCounter:
        if not self._token_counter:
            self._token_counter = OpenAITokenCounter(ModelType.GPT_4O_MINI)
        return self._token_counter

    def _run(
        self,
        messages: List[Dict[str, Any]],
        response_format: Optional[Type[BaseModel]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        body = self._build_request_body(messages, response_format, tools)
        data = self._post_messages(body)
        return self._to_chat_completion(data)

    async def _arun(
        self,
        messages: List[Dict[str, Any]],
        response_format: Optional[Type[BaseModel]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ChatCompletion | AsyncStream[ChatCompletionChunk]:
        return await asyncio.to_thread(
            self._run,
            messages,
            response_format,
            tools,
        )

    def _build_request_body(
        self,
        messages: List[Dict[str, Any]],
        response_format: Optional[Type[BaseModel]],
        tools: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        system, chat_messages = self._convert_messages(messages)
        request_config = dict(self.model_config_dict)
        max_tokens = int(
            request_config.pop(
                "max_tokens",
                request_config.pop("max_completion_tokens", 4096),
            )
        )

        body: Dict[str, Any] = {
            "model": str(self.model_type),
            "max_tokens": max_tokens,
            "messages": chat_messages,
        }
        if system:
            body["system"] = system
        if "temperature" in request_config:
            body["temperature"] = request_config["temperature"]
        if "top_p" in request_config:
            body["top_p"] = request_config["top_p"]
        if "stop" in request_config:
            body["stop_sequences"] = request_config["stop"]
        if response_format:
            body["system"] = (
                (body.get("system", "") + "\n\n")
                + "Return a response that satisfies the requested schema."
            ).strip()

        anthropic_tools = self._convert_tools(tools or [])
        if anthropic_tools:
            body["tools"] = anthropic_tools

        return body

    def _post_messages(self, body: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self._url.rstrip('/')}/v1/messages"
        headers = {
            "x-api-key": str(self._api_key),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        last_error: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=body,
                    timeout=self._timeout,
                )
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= self._max_retries:
                    break
                time.sleep(min(2**attempt, 8))

        assert last_error is not None
        raise last_error

    def _convert_messages(
        self,
        messages: List[Dict[str, Any]],
    ) -> tuple[str, List[Dict[str, str]]]:
        system_parts: List[str] = []
        chat_messages: List[Dict[str, str]] = []

        for message in messages:
            role = str(message.get("role") or "user")
            content = self._content_to_text(message.get("content"))
            if not content:
                continue

            if role == "system":
                system_parts.append(content)
                continue

            role = "assistant" if role == "assistant" else "user"
            if chat_messages and chat_messages[-1]["role"] == role:
                chat_messages[-1]["content"] += f"\n\n{content}"
            else:
                chat_messages.append({"role": role, "content": content})

        if not chat_messages:
            chat_messages.append({"role": "user", "content": "Continue."})
        if chat_messages[0]["role"] == "assistant":
            chat_messages.insert(0, {"role": "user", "content": "Continue."})

        return "\n\n".join(system_parts), chat_messages

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if text:
                        parts.append(str(text))
                else:
                    parts.append(str(item))
            return "\n".join(parts).strip()
        return str(content).strip()

    @staticmethod
    def _convert_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        converted: List[Dict[str, Any]] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            function = tool.get("function") if tool.get("type") == "function" else tool
            name = function.get("name") if isinstance(function, dict) else None
            if not name:
                continue
            converted.append(
                {
                    "name": name,
                    "description": function.get("description", ""),
                    "input_schema": function.get(
                        "parameters",
                        {"type": "object", "properties": {}},
                    ),
                }
            )
        return converted

    def _to_chat_completion(self, data: Dict[str, Any]) -> ChatCompletion:
        text_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []

        for block in data.get("content", []):
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and block.get("text"):
                text_parts.append(str(block["text"]))
            elif block.get("type") == "tool_use":
                tool_calls.append(
                    {
                        "id": block.get("id") or f"toolu_{len(tool_calls)}",
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(
                                block.get("input") or {},
                                ensure_ascii=False,
                            ),
                        },
                    }
                )

        content = "\n".join(text_parts).strip()
        usage = data.get("usage") or {}
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)

        choice: Choice | Dict[str, Any] = {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls or None,
            },
            "finish_reason": "tool_calls" if tool_calls else "stop",
        }

        return ChatCompletion.construct(
            id=data.get("id", f"anthropic-messages-{int(time.time())}"),
            choices=[choice],
            created=int(time.time()),
            model=data.get("model", str(self.model_type)),
            object="chat.completion",
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        )
