"""
LLM Client Wrapper
Unified OpenAI format API calls
Supports Ollama num_ctx parameter to prevent prompt truncation
"""

import json
import os
import re
import time
from typing import Optional, Dict, Any, List
from openai import OpenAI
import requests

from ..config import Config
from ..utils.logger import get_logger


logger = get_logger('mirofish.llm_client')


class LLMClient:
    """LLM Client"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 300.0
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model = model or Config.LLM_MODEL_NAME
        self.timeout = timeout

        if not self.api_key:
            raise ValueError("LLM_API_KEY not configured")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout,
        )

        # Ollama context window size — prevents prompt truncation.
        # Read from env OLLAMA_NUM_CTX, default 8192 (Ollama default is only 2048).
        self._num_ctx = int(os.environ.get('OLLAMA_NUM_CTX', '8192'))

    def _is_ollama(self) -> bool:
        """Check if we're talking to an Ollama server."""
        return '11434' in (self.base_url or '')

    def _is_anthropic_compatible(self) -> bool:
        """Check if we're talking to an Anthropic Messages compatible endpoint."""
        return 'anthropic' in (self.base_url or '') or (self.base_url or '').rstrip('/').endswith('/coding')

    def _chat_anthropic(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Send chat request to an Anthropic Messages compatible endpoint."""
        system_messages = [m.get("content", "") for m in messages if m.get("role") == "system"]
        chat_messages = [
            {"role": m.get("role", "user"), "content": m.get("content", "")}
            for m in messages
            if m.get("role") != "system"
        ]

        body: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": chat_messages,
        }
        if system_messages:
            body["system"] = "\n\n".join(system_messages)
        if temperature is not None:
            body["temperature"] = temperature

        url = f"{self.base_url.rstrip('/')}/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        response = self._post_anthropic_with_retry(url, headers, body)
        data = response.json()

        text_parts = [
            item.get("text", "")
            for item in data.get("content", [])
            if item.get("type") == "text" and item.get("text")
        ]
        content = "\n".join(text_parts).strip()
        if not content:
            raise ValueError("Anthropic-compatible LLM returned no text content")
        return content

    def _post_anthropic_with_retry(
        self,
        url: str,
        headers: Dict[str, str],
        body: Dict[str, Any],
    ) -> requests.Response:
        max_retries = int(os.environ.get("LLM_HTTP_MAX_RETRIES", "4"))
        base_delay = float(os.environ.get("LLM_HTTP_RETRY_BASE_SECONDS", "2"))
        max_delay = float(os.environ.get("LLM_HTTP_RETRY_MAX_SECONDS", "30"))
        last_error: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=body,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return response
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.HTTPError,
            ) as exc:
                if isinstance(exc, requests.exceptions.HTTPError):
                    status_code = exc.response.status_code if exc.response is not None else 0
                    if status_code and status_code < 500 and status_code != 429:
                        raise
                last_error = exc
                if attempt >= max_retries:
                    raise
                logger.warning(
                    "Anthropic-compatible LLM request failed (attempt %s/%s): %s",
                    attempt + 1,
                    max_retries + 1,
                    exc,
                )
                delay = min(max_delay, base_delay * (2 ** attempt))
                time.sleep(delay)

        raise last_error or RuntimeError("Anthropic-compatible LLM request failed")

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None
    ) -> str:
        """
        Send chat request

        Args:
            messages: Message list
            temperature: Temperature parameter
            max_tokens: Max token count
            response_format: Response format (e.g., JSON mode)

        Returns:
            Model response text
        """
        if self._is_anthropic_compatible():
            return self._chat_anthropic(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if response_format:
            kwargs["response_format"] = response_format

        # For Ollama: pass num_ctx via extra_body to prevent prompt truncation
        if self._is_ollama() and self._num_ctx:
            kwargs["extra_body"] = {
                "options": {"num_ctx": self._num_ctx}
            }

        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        # Some models (like MiniMax M2.5) include <think>thinking content in response, need to remove
        content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
        return content

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """
        Send chat request and return JSON

        Args:
            messages: Message list
            temperature: Temperature parameter
            max_tokens: Max token count

        Returns:
            Parsed JSON object
        """
        response = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        # Clean markdown code block markers
        cleaned_response = response.strip()
        cleaned_response = re.sub(r'^```(?:json)?\s*\n?', '', cleaned_response, flags=re.IGNORECASE)
        cleaned_response = re.sub(r'\n?```\s*$', '', cleaned_response)
        cleaned_response = cleaned_response.strip()

        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format from LLM: {cleaned_response}")
