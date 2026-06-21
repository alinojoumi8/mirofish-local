"""
EmbeddingService - provider-aware embedding generation.

Default provider is local Ollama with nomic-embed-text at 768 dimensions.
Gemini is optional and uses the Gemini REST embedding API with 768 output
dimensions so it remains compatible with the existing Neo4j vector indexes.
"""

import logging
import hashlib
import json
import os
import time
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from ..config import Config

logger = logging.getLogger('mirofish.embedding')

RETRIEVAL_DOCUMENT = "RETRIEVAL_DOCUMENT"
RETRIEVAL_QUERY = "RETRIEVAL_QUERY"
SUPPORTED_DIMENSIONS = 768


@dataclass(frozen=True)
class EmbeddingProviderInfo:
    provider: str
    model: str
    dimensions: int
    base_url: str

    @property
    def provider_id(self) -> str:
        return f"{self.provider}:{self.model}:{self.dimensions}"

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["provider_id"] = self.provider_id
        return data


class EmbeddingCache:
    """Disk-backed embedding cache keyed by provider, task type, and text."""

    VERSION = "embedding-v1"

    def __init__(self, cache_dir: Optional[str | os.PathLike[str]] = None):
        self.cache_dir = Path(cache_dir or Config.GRAPH_EMBEDDING_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, provider_id: str, task_type: str, text: str) -> Optional[List[float]]:
        path = self._path(provider_id, task_type, text)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None

        metadata = payload.get("metadata", {})
        if (
            metadata.get("version") != self.VERSION
            or metadata.get("provider_id") != provider_id
            or metadata.get("task_type") != task_type
            or metadata.get("text_hash") != self._hash_text(text)
        ):
            return None
        vector = payload.get("vector")
        return vector if isinstance(vector, list) else None

    def set(self, provider_id: str, task_type: str, text: str, vector: List[float]) -> None:
        path = self._path(provider_id, task_type, text)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": {
                "version": self.VERSION,
                "provider_id": provider_id,
                "task_type": task_type,
                "text_hash": self._hash_text(text),
            },
            "vector": vector,
        }
        tmp_path = path.with_suffix(".tmp")
        with self._lock:
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False)
            os.replace(tmp_path, path)

    def _path(self, provider_id: str, task_type: str, text: str) -> Path:
        cache_key = self._hash_text("|".join([
            self.VERSION,
            provider_id,
            task_type,
            self._hash_text(text),
        ]))
        return self.cache_dir / cache_key[:2] / f"{cache_key}.json"


class EmbeddingService:
    """Generate embeddings using Ollama or Gemini."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        dimensions: Optional[int] = None,
        max_retries: int = 3,
        timeout: int = 30,
        auto_pull: Optional[bool] = None,
    ):
        self.provider = (provider or Config.EMBEDDING_PROVIDER or "ollama").strip().lower()
        self.dimensions = dimensions or Config.EMBEDDING_DIMENSIONS
        self.max_retries = max_retries
        self.timeout = timeout
        self.auto_pull = Config.EMBEDDING_AUTO_PULL if auto_pull is None else auto_pull
        self._cache: dict[tuple[str, str], List[float]] = {}
        self._cache_max_size = 2000
        self._persistent_cache = (
            EmbeddingCache() if Config.GRAPH_EMBEDDING_CACHE_ENABLED else None
        )
        self._ready_checked = False
        self._unavailable_until = 0.0
        self._last_error: Optional[str] = None

        if self.dimensions != SUPPORTED_DIMENSIONS:
            raise EmbeddingError(
                f"Unsupported embedding dimension {self.dimensions}; Neo4j indexes are {SUPPORTED_DIMENSIONS}d"
            )

        if self.provider == "ollama":
            self.model = model or Config.EMBEDDING_MODEL
            self.base_url = (base_url or Config.EMBEDDING_BASE_URL).rstrip('/')
            self.api_key = api_key
            self._embed_url = f"{self.base_url}/api/embed"
            self._tags_url = f"{self.base_url}/api/tags"
            self._pull_url = f"{self.base_url}/api/pull"
        elif self.provider == "gemini":
            self.model = model or Config.GEMINI_EMBEDDING_MODEL
            self.base_url = (base_url or Config.GEMINI_EMBEDDING_BASE_URL).rstrip('/')
            self.api_key = api_key or Config.GEMINI_API_KEY
            if not self.api_key:
                raise EmbeddingError("GEMINI_API_KEY is required when EMBEDDING_PROVIDER=gemini")
            self._embed_url = f"{self.base_url}/models/{self.model}:batchEmbedContents"
        else:
            raise EmbeddingError(f"Unsupported embedding provider: {self.provider}")

    @property
    def info(self) -> EmbeddingProviderInfo:
        return EmbeddingProviderInfo(
            provider=self.provider,
            model=self.model,
            dimensions=self.dimensions,
            base_url=self.base_url,
        )

    def provider_id(self) -> str:
        return self.info.provider_id

    def embed(self, text: str, task_type: str = RETRIEVAL_QUERY) -> List[float]:
        """Generate an embedding for one text."""
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty text")

        text = text.strip()
        cache_key = (task_type, text)
        if cache_key in self._cache:
            return self._cache[cache_key]

        vector = self._request_embeddings([text], task_type=task_type)[0]
        self._cache_put(cache_key, vector)
        return vector

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        task_type: str = RETRIEVAL_DOCUMENT,
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        results: List[Optional[List[float]]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        for i, text in enumerate(texts):
            text = text.strip() if text else ""
            cache_key = (task_type, text)
            if cache_key in self._cache:
                results[i] = self._cache[cache_key]
            elif text and self._persistent_cache:
                cached = self._persistent_cache.get(self.provider_id(), task_type, text)
                if cached is not None:
                    results[i] = cached
                    self._cache_put(cache_key, cached)
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(text)
            elif text:
                uncached_indices.append(i)
                uncached_texts.append(text)
            else:
                results[i] = [0.0] * self.dimensions

        if uncached_texts:
            all_vectors: List[List[float]] = []
            for start in range(0, len(uncached_texts), batch_size):
                batch = uncached_texts[start:start + batch_size]
                all_vectors.extend(self._request_embeddings(batch, task_type=task_type))

            for idx, vec, text in zip(uncached_indices, all_vectors, uncached_texts):
                results[idx] = vec
                self._cache_put((task_type, text), vec)
                if self._persistent_cache:
                    self._persistent_cache.set(self.provider_id(), task_type, text, vec)

        return results  # type: ignore

    def health_check(self) -> bool:
        """Return whether the configured embedding provider can generate vectors."""
        try:
            return self.health_status()["healthy"]
        except Exception:
            return False

    def health_status(self) -> Dict[str, Any]:
        """Detailed status used by /api/status."""
        status = self.info.to_dict()
        status.update({
            "healthy": False,
            "ready": False,
            "error": None,
        })
        try:
            self.ensure_ready(check_only=True)
            vec = self.embed("health check", task_type=RETRIEVAL_QUERY)
            status["healthy"] = len(vec) == self.dimensions
            status["ready"] = True
            status["vector_length"] = len(vec)
        except Exception as exc:
            status["error"] = str(exc)
        return status

    def ensure_ready(self, check_only: bool = False) -> None:
        """Verify provider availability; optionally pull the local Ollama model."""
        if self._ready_checked:
            return
        self._raise_if_circuit_open()

        if self.provider == "ollama":
            self._ensure_ollama_ready(check_only=check_only)
        elif self.provider == "gemini":
            # Gemini readiness is validated by the first embedding call.
            pass

        if not check_only:
            self._ready_checked = True

    def _ensure_ollama_ready(self, check_only: bool) -> None:
        try:
            response = requests.get(self._tags_url, timeout=min(self.timeout, 10))
            response.raise_for_status()
            models = response.json().get("models", [])
        except requests.exceptions.RequestException as exc:
            if not check_only:
                self._mark_unavailable(exc)
            raise EmbeddingError(f"Ollama embedding server is not reachable at {self.base_url}: {exc}") from exc

        model_names = {m.get("name") for m in models if isinstance(m, dict)}
        model_names.update((m.get("model") for m in models if isinstance(m, dict)))
        model_names = {name for name in model_names if name}
        if self._ollama_model_installed(model_names):
            return

        if check_only or not self.auto_pull:
            raise EmbeddingError(
                f"Ollama model '{self.model}' is not installed. Pull it with: ollama pull {self.model}"
            )

        logger.info("Ollama embedding model '%s' missing; pulling it before use", self.model)
        try:
            response = requests.post(
                self._pull_url,
                json={"model": self.model, "stream": False},
                timeout=max(self.timeout, 600),
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            self._mark_unavailable(exc)
            raise EmbeddingError(f"Failed to pull Ollama embedding model '{self.model}': {exc}") from exc

    def _request_embeddings(self, texts: List[str], task_type: str) -> List[List[float]]:
        self._raise_if_circuit_open()
        self.ensure_ready(check_only=False)

        if self.provider == "ollama":
            return self._request_ollama_embeddings(texts)
        if self.provider == "gemini":
            return self._request_gemini_embeddings(texts, task_type=task_type)
        raise EmbeddingError(f"Unsupported embedding provider: {self.provider}")

    def _ollama_model_installed(self, model_names: set[str]) -> bool:
        if self.model in model_names:
            return True
        if ":" not in self.model and f"{self.model}:latest" in model_names:
            return True
        return False

    def _request_ollama_embeddings(self, texts: List[str]) -> List[List[float]]:
        payload = {"model": self.model, "input": texts}
        return self._post_embeddings(self._embed_url, payload, headers=None)

    def _request_gemini_embeddings(self, texts: List[str], task_type: str) -> List[List[float]]:
        model_name = f"models/{self.model}"
        payload = {
            "requests": [
                {
                    "model": model_name,
                    "content": {"parts": [{"text": text}]},
                    "taskType": task_type,
                    "outputDimensionality": self.dimensions,
                }
                for text in texts
            ]
        }
        headers = {
            "x-goog-api-key": self.api_key or "",
            "content-type": "application/json",
        }
        return self._post_embeddings(self._embed_url, payload, headers=headers)

    def _post_embeddings(
        self,
        url: str,
        payload: Dict[str, Any],
        headers: Optional[Dict[str, str]],
    ) -> List[List[float]]:
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
                response.raise_for_status()
                vectors = self._parse_embedding_response(response.json(), expected_count=len(payload.get("input", payload.get("requests", []))))
                for vector in vectors:
                    if len(vector) != self.dimensions:
                        raise EmbeddingError(
                            f"Expected {self.dimensions}d embedding from {self.provider}, got {len(vector)}d"
                        )
                self._last_error = None
                return vectors
            except requests.exceptions.ConnectionError as exc:
                self._mark_unavailable(exc)
                raise EmbeddingError(f"{self.provider} embedding server connection failed: {exc}") from exc
            except requests.exceptions.Timeout as exc:
                self._mark_unavailable(exc)
                raise EmbeddingError(f"{self.provider} embedding request timed out: {exc}") from exc
            except requests.exceptions.HTTPError as exc:
                last_error = exc
                body = exc.response.text if exc.response is not None else str(exc)
                if exc.response is not None and exc.response.status_code < 500:
                    raise EmbeddingError(f"{self.provider} embedding failed: {body}") from exc
                logger.warning(
                    "%s embedding HTTP error (attempt %s/%s): %s",
                    self.provider,
                    attempt + 1,
                    self.max_retries,
                    body,
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise EmbeddingError(f"Invalid {self.provider} embedding response: {exc}") from exc

            if attempt < self.max_retries - 1:
                time.sleep(2 ** attempt)

        raise EmbeddingError(f"{self.provider} embedding failed after {self.max_retries} retries: {last_error}")

    def _parse_embedding_response(self, data: Dict[str, Any], expected_count: int) -> List[List[float]]:
        if self.provider == "ollama":
            embeddings = data.get("embeddings", [])
            if len(embeddings) != expected_count:
                raise EmbeddingError(f"Expected {expected_count} embeddings, got {len(embeddings)}")
            return embeddings

        embeddings = data.get("embeddings", [])
        vectors = [item.get("values", []) for item in embeddings]
        if len(vectors) != expected_count:
            raise EmbeddingError(f"Expected {expected_count} embeddings, got {len(vectors)}")
        return vectors

    def _cache_put(self, cache_key: tuple[str, str], vector: List[float]) -> None:
        if len(self._cache) >= self._cache_max_size:
            keys_to_remove = list(self._cache.keys())[:self._cache_max_size // 10]
            for key in keys_to_remove:
                del self._cache[key]
        self._cache[cache_key] = vector

    def _raise_if_circuit_open(self) -> None:
        if time.time() < self._unavailable_until:
            raise EmbeddingError(
                f"Embedding provider '{self.provider}' is temporarily disabled after a recent failure: {self._last_error}"
            )

    def _mark_unavailable(self, exc: Exception) -> None:
        self._last_error = str(exc)
        self._unavailable_until = time.time() + Config.EMBEDDING_CIRCUIT_BREAKER_SECONDS


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""
