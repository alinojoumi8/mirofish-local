# Provider Switching and Local Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let MiroFish switch cleanly between MiniMax M2.7 token-plan API, Kimi K2.6 API, and the existing local Ollama default, while using self-hosted Neo4j + Graphiti for temporal graph memory instead of Zep Cloud.

**Architecture:** Add a provider registry that resolves named LLM profiles from environment variables, then route all backend LLM calls and OASIS subprocess model creation through that registry. Kimi K2.6 uses its OpenAI-compatible coding endpoint; MiniMax M2.7 uses its Anthropic-compatible endpoint and needs an Anthropic Messages adapter rather than the current OpenAI SDK-only client. Provider selection should be saved with each graph build / simulation / report request so switching models does not require code edits or server restarts. Memory uses Graphiti as the temporal episode/entity/fact layer on top of local Neo4j; the existing Neo4j storage code becomes a compatibility adapter until callers are migrated.

**Tech Stack:** Flask, OpenAI Python SDK, CAMEL/OASIS, Vue 3, Axios, Neo4j Community Edition 5.26+, Graphiti Core, Ollama embeddings, pytest.

---

## Current Findings

- `backend/app/utils/llm_client.py` already uses the OpenAI SDK with configurable `api_key`, `base_url`, and `model`.
- `backend/app/services/simulation_config_generator.py` and `backend/app/services/oasis_profile_generator.py` bypass `LLMClient` and instantiate `OpenAI` directly. These must be moved behind the shared provider path.
- OASIS subprocess scripts read `LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL_NAME` directly in `backend/scripts/run_twitter_simulation.py`, `backend/scripts/run_reddit_simulation.py`, and `backend/scripts/run_parallel_simulation.py`.
- Local memory is already mostly implemented with `GraphStorage`, `Neo4jStorage`, `EmbeddingService`, and `GraphMemoryUpdater`, but the target memory layer is now Graphiti on Neo4j.
- Graphiti is self-hosted OSS, not Zep Cloud. It requires Neo4j 5.26+ in current docs, supports Neo4j via `Neo4jDriver`, and uses temporal episodes/provenance plus hybrid retrieval.
- Graphiti defaults to OpenAI-style LLM/embedding clients and works best with providers that support structured outputs. The implementation should start with the app's selected provider for LLM extraction, but keep the current local embedding path available if Graphiti provider compatibility becomes a blocker.
- Zep is still present as stale text in `README.md`, `frontend/CHINESE_TEXT_INVENTORY.md`, API progress copy, and `backend/uv.lock`; it is not in `backend/requirements.txt` or `backend/pyproject.toml`.
- Kimi K2.6 should use OpenAI-compatible base URL `https://api.kimi.com/coding/v1`.
- MiniMax M2.7 should use Anthropic-compatible base URL `https://api.minimax.io/anthropic`, so the provider switcher must support a non-OpenAI transport.
- Runtime smoke test note from 2026-05-12: Kimi coding endpoint responded with `403 access_terminated_error` and said Kimi For Coding is only available for supported coding-agent clients. The app can still run, but Kimi-backed LLM features may need client allowlisting or a supported client signature.

## File Structure

- Create `backend/app/utils/llm_provider.py`: provider dataclass, env resolver, provider validation, provider-specific `extra_body`.
- Modify `backend/app/utils/llm_client.py`: accept `provider_name`, resolve provider through `llm_provider.py`, and apply provider-specific request rules.
- Modify `backend/app/config.py`: add `LLM_DEFAULT_PROVIDER`, named provider env fields, and validation for selected providers.
- Modify `backend/app/services/simulation_config_generator.py`: replace direct `OpenAI` setup with `LLMClient`.
- Modify `backend/app/services/oasis_profile_generator.py`: replace direct `OpenAI` setup with `LLMClient`.
- Modify `backend/scripts/run_twitter_simulation.py`, `backend/scripts/run_reddit_simulation.py`, `backend/scripts/run_parallel_simulation.py`: resolve named provider into `OPENAI_API_KEY`, `OPENAI_API_BASE_URL`, and model before creating CAMEL models.
- Create `backend/app/api/settings.py`: expose safe provider metadata and optional active-provider validation.
- Modify `backend/app/api/__init__.py` and `backend/app/__init__.py`: register `settings_bp`.
- Modify `backend/app/api/graph.py`, `backend/app/api/simulation.py`, `backend/app/api/report.py`: accept optional `llm_provider` in request JSON and pass it into services.
- Modify `backend/app/services/simulation_manager.py` and generated `simulation_config.json`: persist `llm_provider`, `llm_model`, and `llm_base_url`.
- Create `backend/app/storage/graphiti_memory.py`: Graphiti initializer, episode ingestion, search, and health wrapper using the existing Neo4j credentials.
- Modify `backend/app/storage/graph_storage.py` or create an adapter beside it: route memory writes/searches through Graphiti while preserving the current API shape expected by graph/report/simulation code.
- Modify `backend/app/services/graph_builder.py` and `backend/app/services/graph_memory_updater.py`: write document chunks and agent activities as Graphiti episodes.
- Modify `docker-compose.yml`: upgrade Neo4j image to 5.26+ for Graphiti compatibility.
- Modify `backend/requirements.txt` and `backend/pyproject.toml`: add `graphiti-core>=0.28.2,<0.29.0` unless current testing proves a newer stable release is required.
- Create `frontend/src/api/settings.js`: fetch provider list.
- Modify `frontend/src/views/Home.vue`, `frontend/src/components/Step1GraphBuild.vue`, `frontend/src/components/Step2EnvSetup.vue`, `frontend/src/components/Step3Simulation.vue`, and report/chat entrypoints as needed: add provider picker and pass `llm_provider`.
- Modify `.env.example`: document MiniMax, Kimi, and local provider setup.
- Modify `README.md`: describe provider switching and confirm memory is local Neo4j, not Zep.
- Modify `backend/uv.lock`: regenerate after removing stale `zep-cloud` if `uv sync` still retains it.
- Create `backend/tests/test_llm_provider.py`: unit tests for provider resolution and safe metadata.
- Create `backend/tests/test_llm_client_provider.py`: mocked OpenAI calls proving MiniMax/Kimi extra bodies are applied.

---

### Task 1: Provider Registry

**Files:**
- Create: `backend/app/utils/llm_provider.py`
- Modify: `backend/app/config.py`
- Test: `backend/tests/test_llm_provider.py`

- [ ] **Step 1: Write failing provider resolution tests**

```python
# backend/tests/test_llm_provider.py
import os

from app.utils.llm_provider import get_llm_provider, list_llm_providers


def test_minimax_provider_resolves_from_env(monkeypatch):
    monkeypatch.setenv("LLM_DEFAULT_PROVIDER", "minimax")
    monkeypatch.setenv("MINIMAX_API_KEY", "minimax-secret")
    monkeypatch.setenv("MINIMAX_BASE_URL", "https://api.minimax.io/anthropic")
    monkeypatch.setenv("MINIMAX_MODEL", "MiniMax-M2.7")

    provider = get_llm_provider("minimax")

    assert provider.name == "minimax"
    assert provider.api_key == "minimax-secret"
    assert provider.base_url == "https://api.minimax.io/anthropic"
    assert provider.model == "MiniMax-M2.7"
    assert provider.extra_body == {"reasoning_split": True}


def test_kimi_provider_resolves_from_env(monkeypatch):
    monkeypatch.setenv("KIMI_API_KEY", "kimi-secret")
    monkeypatch.setenv("KIMI_BASE_URL", "https://api.kimi.com/coding/v1")
    monkeypatch.setenv("KIMI_MODEL", "kimi-k2.6")

    provider = get_llm_provider("kimi")

    assert provider.name == "kimi"
    assert provider.api_key == "kimi-secret"
    assert provider.base_url == "https://api.kimi.com/coding/v1"
    assert provider.model == "kimi-k2.6"
    assert provider.extra_body == {}


def test_provider_metadata_never_exposes_api_keys(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "minimax-secret")
    monkeypatch.setenv("KIMI_API_KEY", "kimi-secret")

    metadata = list_llm_providers()

    assert "minimax-secret" not in repr(metadata)
    assert "kimi-secret" not in repr(metadata)
    assert {item["name"] for item in metadata} >= {"ollama", "minimax", "kimi"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_llm_provider.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'app.utils.llm_provider'`.

- [ ] **Step 3: Implement provider registry**

```python
# backend/app/utils/llm_provider.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class LLMProvider:
    name: str
    label: str
    api_key: str
    base_url: str
    model: str
    extra_body: Dict[str, Any]
    configured: bool


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _build_provider(name: str) -> LLMProvider:
    normalized = (name or _env("LLM_DEFAULT_PROVIDER", "ollama")).strip().lower()

    if normalized == "minimax":
        api_key = _env("MINIMAX_API_KEY")
        return LLMProvider(
            name="minimax",
            label="MiniMax M2.7",
            api_key=api_key,
            base_url=_env("MINIMAX_BASE_URL", "https://api.minimax.io/anthropic"),
            model=_env("MINIMAX_MODEL", "MiniMax-M2.7"),
            extra_body={"reasoning_split": True},
            configured=bool(api_key),
        )

    if normalized == "kimi":
        api_key = _env("KIMI_API_KEY")
        return LLMProvider(
            name="kimi",
            label="Kimi K2.6",
            api_key=api_key,
            base_url=_env("KIMI_BASE_URL", "https://api.kimi.com/coding/v1"),
            model=_env("KIMI_MODEL", "kimi-k2.6"),
            extra_body={},
            configured=bool(api_key),
        )

    api_key = _env("OLLAMA_API_KEY", _env("LLM_API_KEY", "ollama"))
    return LLMProvider(
        name="ollama",
        label="Local Ollama",
        api_key=api_key,
        base_url=_env("OLLAMA_BASE_URL", _env("LLM_BASE_URL", "http://localhost:11434/v1")),
        model=_env("OLLAMA_MODEL", _env("LLM_MODEL_NAME", "qwen2.5:32b")),
        extra_body={},
        configured=bool(api_key),
    )


def get_llm_provider(name: Optional[str] = None) -> LLMProvider:
    provider = _build_provider(name or _env("LLM_DEFAULT_PROVIDER", "ollama"))
    if not provider.configured:
        raise ValueError(f"LLM provider '{provider.name}' is not configured; set its API key in .env")
    return provider


def list_llm_providers() -> List[Dict[str, Any]]:
    default_provider = _env("LLM_DEFAULT_PROVIDER", "ollama").lower()
    return [
        {
            "name": provider.name,
            "label": provider.label,
            "model": provider.model,
            "base_url": provider.base_url,
            "configured": provider.configured,
            "is_default": provider.name == default_provider,
        }
        for provider in (_build_provider("ollama"), _build_provider("minimax"), _build_provider("kimi"))
    ]
```

- [ ] **Step 4: Add config fields**

```python
# backend/app/config.py, inside class Config
LLM_DEFAULT_PROVIDER = os.environ.get("LLM_DEFAULT_PROVIDER", "ollama")

MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY")
MINIMAX_BASE_URL = os.environ.get("MINIMAX_BASE_URL", "https://api.minimax.io/anthropic")
MINIMAX_MODEL = os.environ.get("MINIMAX_MODEL", "MiniMax-M2.7")

KIMI_API_KEY = os.environ.get("KIMI_API_KEY")
KIMI_BASE_URL = os.environ.get("KIMI_BASE_URL", "https://api.kimi.com/coding/v1")
KIMI_MODEL = os.environ.get("KIMI_MODEL", "kimi-k2.6")

OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", os.environ.get("LLM_API_KEY", "ollama"))
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"))
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", os.environ.get("LLM_MODEL_NAME", "qwen2.5:32b"))
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_llm_provider.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/app/utils/llm_provider.py backend/app/config.py backend/tests/test_llm_provider.py
git commit -m "feat: add named llm provider registry"
```

---

### Task 2: Shared LLM Client Uses Providers

**Files:**
- Modify: `backend/app/utils/llm_client.py`
- Test: `backend/tests/test_llm_client_provider.py`

- [ ] **Step 1: Write mocked client tests**

```python
# backend/tests/test_llm_client_provider.py
from app.utils.llm_client import LLMClient


class FakeCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs

        class Message:
            content = '{"ok": true}'

        class Choice:
            message = Message()
            finish_reason = "stop"

        class Response:
            choices = [Choice()]

        return Response()


class FakeOpenAI:
    def __init__(self, *args, **kwargs):
        self.chat = type("Chat", (), {"completions": FakeCompletions()})()


def test_minimax_extra_body_is_sent(monkeypatch):
    fake = FakeOpenAI()
    monkeypatch.setenv("MINIMAX_API_KEY", "secret")
    monkeypatch.setattr("app.utils.llm_client.OpenAI", lambda **kwargs: fake)

    client = LLMClient(provider_name="minimax")
    assert client.chat([{"role": "user", "content": "return json"}]) == '{"ok": true}'

    assert fake.chat.completions.kwargs["model"] == "MiniMax-M2.7"
    assert fake.chat.completions.kwargs["extra_body"] == {"reasoning_split": True}


def test_kimi_uses_configured_openai_compatible_model(monkeypatch):
    fake = FakeOpenAI()
    monkeypatch.setenv("KIMI_API_KEY", "secret")
    monkeypatch.setattr("app.utils.llm_client.OpenAI", lambda **kwargs: fake)

    client = LLMClient(provider_name="kimi")
    client.chat([{"role": "user", "content": "hello"}])

    assert fake.chat.completions.kwargs["model"] == "kimi-k2.6"
    assert "extra_body" not in fake.chat.completions.kwargs
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_llm_client_provider.py -v`

Expected: FAIL with `TypeError: LLMClient.__init__() got an unexpected keyword argument 'provider_name'`.

- [ ] **Step 3: Update `LLMClient` constructor and request building**

```python
# backend/app/utils/llm_client.py
from .llm_provider import get_llm_provider

class LLMClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        provider_name: Optional[str] = None,
        timeout: float = 300.0
    ):
        self.provider = get_llm_provider(provider_name)
        self.api_key = api_key or self.provider.api_key
        self.base_url = base_url or self.provider.base_url
        self.model = model or self.provider.model
        self.provider_name = self.provider.name
        self.provider_extra_body = dict(self.provider.extra_body)
```

In `chat()`, merge `self.provider_extra_body` into `kwargs["extra_body"]`, then merge Ollama `num_ctx` options when `_is_ollama()` is true:

```python
extra_body = dict(self.provider_extra_body)
if self._is_ollama() and self._num_ctx:
    extra_body["options"] = {"num_ctx": self._num_ctx}
if extra_body:
    kwargs["extra_body"] = extra_body
```

- [ ] **Step 4: Keep JSON cleanup behavior**

Keep the existing cleanup:

```python
content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
```

This remains useful for MiniMax when callers do not need reasoning text.

- [ ] **Step 5: Run tests**

Run: `cd backend && uv run pytest tests/test_llm_provider.py tests/test_llm_client_provider.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/app/utils/llm_client.py backend/tests/test_llm_client_provider.py
git commit -m "feat: route llm client through providers"
```

---

### Task 3: Remove Direct OpenAI Calls From Backend Services

**Files:**
- Modify: `backend/app/services/simulation_config_generator.py`
- Modify: `backend/app/services/oasis_profile_generator.py`
- Test: existing pytest plus service-level smoke tests if added

- [ ] **Step 1: Change constructors to accept provider name**

For both service classes, add `provider_name: Optional[str] = None` and create `self.llm = LLMClient(provider_name=provider_name, api_key=api_key, base_url=base_url, model=model_name)`.

```python
from ..utils.llm_client import LLMClient

self.llm = LLMClient(
    provider_name=provider_name,
    api_key=api_key,
    base_url=base_url,
    model=model_name,
)
self.model_name = self.llm.model
self.base_url = self.llm.base_url
```

- [ ] **Step 2: Replace `self.client.chat.completions.create` in `SimulationConfigGenerator`**

Use:

```python
content = self.llm.chat(
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ],
    response_format={"type": "json_object"},
    temperature=0.7 - (attempt * 0.1),
)
return json.loads(content)
```

Keep the existing JSON repair fallback around this call.

- [ ] **Step 3: Replace `self.client.chat.completions.create` in `OasisProfileGenerator`**

Use:

```python
content = self.llm.chat(
    messages=[
        {"role": "system", "content": self._get_system_prompt(is_individual)},
        {"role": "user", "content": prompt},
    ],
    response_format={"type": "json_object"},
    temperature=0.7 - (attempt * 0.1),
)
```

Keep rule-based fallback behavior.

- [ ] **Step 4: Run targeted tests**

Run: `cd backend && uv run pytest tests/test_llm_provider.py tests/test_llm_client_provider.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/simulation_config_generator.py backend/app/services/oasis_profile_generator.py
git commit -m "refactor: use shared llm client in generation services"
```

---

### Task 4: Pass Provider Selection Through API Workflows

**Files:**
- Create: `backend/app/api/settings.py`
- Modify: `backend/app/api/__init__.py`
- Modify: `backend/app/__init__.py`
- Modify: `backend/app/api/graph.py`
- Modify: `backend/app/api/simulation.py`
- Modify: `backend/app/api/report.py`
- Modify: `backend/app/services/simulation_manager.py`

- [ ] **Step 1: Add safe provider metadata endpoint**

```python
# backend/app/api/settings.py
from flask import jsonify

from . import settings_bp
from ..utils.llm_provider import list_llm_providers


@settings_bp.route("/llm-providers", methods=["GET"])
def get_llm_providers():
    return jsonify({
        "success": True,
        "data": {
            "providers": list_llm_providers()
        }
    })
```

- [ ] **Step 2: Register `settings_bp`**

```python
# backend/app/api/__init__.py
settings_bp = Blueprint("settings", __name__)
from . import settings  # noqa: E402, F401
```

```python
# backend/app/__init__.py
from .api import graph_bp, simulation_bp, report_bp, settings_bp
app.register_blueprint(settings_bp, url_prefix="/api/settings")
```

- [ ] **Step 3: Accept `llm_provider` in request JSON**

In graph build, simulation prepare, profile generation, simulation start, report generation, report chat, and interview endpoints, read:

```python
llm_provider = data.get("llm_provider") or Config.LLM_DEFAULT_PROVIDER
```

Then pass it into any `LLMClient`, `OntologyGenerator`, `NERExtractor`, `SimulationConfigGenerator`, `OasisProfileGenerator`, or `ReportAgent` construction touched by that request.

- [ ] **Step 4: Persist provider choice in generated simulation config**

Add fields:

```python
"llm_provider": llm_provider,
"llm_model": self.model_name,
"llm_base_url": self.base_url,
```

The existing `llm_model` and `llm_base_url` fields already exist; add `llm_provider` beside them.

- [ ] **Step 5: Run backend smoke checks**

Run: `cd backend && uv run python -m compileall app`

Expected: no syntax errors.

- [ ] **Step 6: Commit**

```bash
git add backend/app/api/settings.py backend/app/api/__init__.py backend/app/__init__.py backend/app/api/graph.py backend/app/api/simulation.py backend/app/api/report.py backend/app/services/simulation_manager.py
git commit -m "feat: pass llm provider through backend workflows"
```

---

### Task 5: Make OASIS Subprocesses Use the Selected Provider

**Files:**
- Modify: `backend/scripts/run_twitter_simulation.py`
- Modify: `backend/scripts/run_reddit_simulation.py`
- Modify: `backend/scripts/run_parallel_simulation.py`
- Modify: `backend/app/services/simulation_runner.py`

- [ ] **Step 1: Add a shared helper in each script**

```python
def resolve_provider_for_oasis(config: Dict[str, Any], use_boost: bool = False) -> Dict[str, str]:
    provider = (config.get("llm_provider") or os.environ.get("LLM_DEFAULT_PROVIDER") or "ollama").lower()

    if use_boost and os.environ.get("LLM_BOOST_API_KEY"):
        return {
            "api_key": os.environ["LLM_BOOST_API_KEY"],
            "base_url": os.environ.get("LLM_BOOST_BASE_URL", ""),
            "model": os.environ.get("LLM_BOOST_MODEL_NAME") or config.get("llm_model", ""),
        }

    if provider == "minimax":
        return {
            "api_key": os.environ.get("MINIMAX_API_KEY", ""),
            "base_url": os.environ.get("MINIMAX_BASE_URL", "https://api.minimax.io/anthropic"),
            "model": os.environ.get("MINIMAX_MODEL", "MiniMax-M2.7"),
        }

    if provider == "kimi":
        return {
            "api_key": os.environ.get("KIMI_API_KEY", ""),
            "base_url": os.environ.get("KIMI_BASE_URL", "https://api.kimi.com/coding/v1"),
            "model": os.environ.get("KIMI_MODEL", "kimi-k2.6"),
        }

    return {
        "api_key": os.environ.get("OLLAMA_API_KEY") or os.environ.get("LLM_API_KEY", "ollama"),
        "base_url": os.environ.get("OLLAMA_BASE_URL") or os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"),
        "model": os.environ.get("OLLAMA_MODEL") or os.environ.get("LLM_MODEL_NAME") or config.get("llm_model", "qwen2.5:32b"),
    }
```

- [ ] **Step 2: Use helper inside existing model creation**

Replace direct reads of `LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL_NAME` with:

```python
resolved = resolve_provider_for_oasis(config, use_boost=use_boost)
llm_api_key = resolved["api_key"]
llm_base_url = resolved["base_url"]
llm_model = resolved["model"]
```

Keep:

```python
os.environ["OPENAI_API_KEY"] = llm_api_key
os.environ["OPENAI_API_BASE_URL"] = llm_base_url
```

- [ ] **Step 3: Ensure subprocess env contains provider vars**

In `SimulationRunner`, when building `env`, pass through `LLM_DEFAULT_PROVIDER`, `MINIMAX_*`, `KIMI_*`, and `OLLAMA_*` from parent `os.environ`.

- [ ] **Step 4: Run syntax checks**

Run:

```bash
cd backend
uv run python -m py_compile scripts/run_twitter_simulation.py scripts/run_reddit_simulation.py scripts/run_parallel_simulation.py
```

Expected: no syntax errors.

- [ ] **Step 5: Commit**

```bash
git add backend/scripts/run_twitter_simulation.py backend/scripts/run_reddit_simulation.py backend/scripts/run_parallel_simulation.py backend/app/services/simulation_runner.py
git commit -m "feat: apply llm provider selection to oasis scripts"
```

---

### Task 6: Frontend Provider Picker

**Files:**
- Create: `frontend/src/api/settings.js`
- Modify: `frontend/src/views/Home.vue`
- Modify: `frontend/src/components/Step1GraphBuild.vue`
- Modify: `frontend/src/components/Step2EnvSetup.vue`
- Modify: `frontend/src/components/Step3Simulation.vue`
- Modify report/chat entry components that send LLM-backed requests.

- [ ] **Step 1: Add settings API client**

```javascript
// frontend/src/api/settings.js
import request from './index'

export function getLlmProviders() {
  return request({
    url: '/api/settings/llm-providers',
    method: 'get'
  })
}
```

- [ ] **Step 2: Add provider selection state**

Use `localStorage` key `mirofish.llmProvider`.

```javascript
const selectedProvider = ref(localStorage.getItem('mirofish.llmProvider') || 'ollama')

watch(selectedProvider, value => {
  localStorage.setItem('mirofish.llmProvider', value)
})
```

- [ ] **Step 3: Render a compact provider select**

Use a normal select beside the existing model badge:

```html
<select v-model="selectedProvider" class="provider-select">
  <option
    v-for="provider in llmProviders"
    :key="provider.name"
    :value="provider.name"
    :disabled="!provider.configured"
  >
    {{ provider.label }} · {{ provider.model }}{{ provider.configured ? '' : ' · not configured' }}
  </option>
</select>
```

- [ ] **Step 4: Include `llm_provider` in API payloads**

Every LLM-backed request should include:

```javascript
llm_provider: selectedProvider.value
```

Start with graph ontology/build, simulation prepare/profile generation/start, report generation/chat, and interview requests.

- [ ] **Step 5: Build frontend**

Run: `cd frontend && npm run build`

Expected: Vite build completes successfully.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/api/settings.js frontend/src/views/Home.vue frontend/src/components/Step1GraphBuild.vue frontend/src/components/Step2EnvSetup.vue frontend/src/components/Step3Simulation.vue
git commit -m "feat: add llm provider picker"
```

---

### Task 7: Graphiti Memory on Neo4j

**Files:**
- Modify: `backend/requirements.txt`
- Modify: `backend/pyproject.toml`
- Modify: `docker-compose.yml`
- Create: `backend/app/storage/graphiti_memory.py`
- Modify: `backend/app/storage/__init__.py`
- Modify: `backend/app/services/graph_builder.py`
- Modify: `backend/app/services/graph_memory_updater.py`
- Test: `backend/tests/test_graphiti_memory_adapter.py`

- [ ] **Step 1: Add failing adapter tests**

```python
# backend/tests/test_graphiti_memory_adapter.py
from datetime import datetime, timezone

import pytest

from app.storage.graphiti_memory import GraphitiMemory


class FakeGraphiti:
    def __init__(self):
        self.indices_built = False
        self.episodes = []
        self.search_queries = []

    async def build_indices_and_constraints(self):
        self.indices_built = True

    async def add_episode(self, **kwargs):
        self.episodes.append(kwargs)

    async def search(self, query, **kwargs):
        self.search_queries.append((query, kwargs))
        return []


@pytest.mark.asyncio
async def test_add_text_episode_records_source_and_group():
    fake = FakeGraphiti()
    memory = GraphitiMemory(graphiti=fake)

    await memory.initialize()
    await memory.add_text_episode(
        group_id="project-123",
        name="chunk-1",
        text="Alice founded ExampleCo in 2024.",
        source_description="uploaded document",
    )

    assert fake.indices_built is True
    assert fake.episodes[0]["group_id"] == "project-123"
    assert fake.episodes[0]["name"] == "chunk-1"
    assert fake.episodes[0]["episode_body"] == "Alice founded ExampleCo in 2024."
    assert fake.episodes[0]["source_description"] == "uploaded document"
    assert isinstance(fake.episodes[0]["reference_time"], datetime)


@pytest.mark.asyncio
async def test_search_delegates_to_graphiti():
    fake = FakeGraphiti()
    memory = GraphitiMemory(graphiti=fake)

    results = await memory.search(group_id="project-123", query="Alice")

    assert results == []
    assert fake.search_queries[0][0] == "Alice"
    assert fake.search_queries[0][1]["group_ids"] == ["project-123"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_graphiti_memory_adapter.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'app.storage.graphiti_memory'`.

- [ ] **Step 3: Add Graphiti dependency and Neo4j version**

In `backend/requirements.txt`, add:

```text
graphiti-core>=0.28.2,<0.29.0
```

In `backend/pyproject.toml`, add to dependencies:

```toml
"graphiti-core>=0.28.2,<0.29.0",
```

In `docker-compose.yml`, change:

```yaml
image: neo4j:5.18-community
```

to:

```yaml
image: neo4j:5.26-community
```

- [ ] **Step 4: Implement Graphiti memory wrapper**

```python
# backend/app/storage/graphiti_memory.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from graphiti_core import Graphiti
from graphiti_core.driver.neo4j_driver import Neo4jDriver
from graphiti_core.nodes import EpisodeType

from ..config import Config


class GraphitiMemory:
    """Self-hosted Graphiti memory over local Neo4j."""

    def __init__(self, graphiti: Optional[Graphiti] = None):
        if graphiti is not None:
            self.graphiti = graphiti
        else:
            driver = Neo4jDriver(
                uri=Config.NEO4J_URI,
                user=Config.NEO4J_USER,
                password=Config.NEO4J_PASSWORD,
            )
            self.graphiti = Graphiti(graph_driver=driver)
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        await self.graphiti.build_indices_and_constraints()
        self._initialized = True

    async def add_text_episode(
        self,
        group_id: str,
        name: str,
        text: str,
        source_description: str,
        reference_time: Optional[datetime] = None,
    ) -> None:
        await self.initialize()
        await self.graphiti.add_episode(
            name=name,
            episode_body=text,
            source=EpisodeType.text,
            source_description=source_description,
            reference_time=reference_time or datetime.now(timezone.utc),
            group_id=group_id,
        )

    async def add_json_episode(
        self,
        group_id: str,
        name: str,
        payload: Dict[str, Any],
        source_description: str,
        reference_time: Optional[datetime] = None,
    ) -> None:
        await self.initialize()
        await self.graphiti.add_episode(
            name=name,
            episode_body=payload,
            source=EpisodeType.json,
            source_description=source_description,
            reference_time=reference_time or datetime.now(timezone.utc),
            group_id=group_id,
        )

    async def search(self, group_id: str, query: str, limit: int = 10) -> List[Any]:
        await self.initialize()
        return await self.graphiti.search(
            query,
            group_ids=[group_id],
            num_results=limit,
        )
```

- [ ] **Step 5: Export wrapper**

```python
# backend/app/storage/__init__.py
from .graphiti_memory import GraphitiMemory
```

- [ ] **Step 6: Route document graph build through Graphiti episodes**

In `backend/app/services/graph_builder.py`, after chunking the uploaded document, send each chunk to:

```python
await graphiti_memory.add_text_episode(
    group_id=graph_id,
    name=f"{project_id}-chunk-{chunk_index}",
    text=chunk_text,
    source_description=f"Uploaded document for project {project_id}",
)
```

Keep the existing `GraphStorage` reads temporarily if downstream UI endpoints still expect the old `nodes`/`edges` shape.

- [ ] **Step 7: Route dynamic simulation memory updates through Graphiti**

In `backend/app/services/graph_memory_updater.py`, convert each agent activity into a JSON episode:

```python
await graphiti_memory.add_json_episode(
    group_id=graph_id,
    name=f"{simulation_id}-{activity_id}",
    payload={
        "simulation_id": simulation_id,
        "agent_id": agent_id,
        "platform": platform,
        "action": action,
        "content": content,
        "timestamp": timestamp,
    },
    source_description="MiroFish simulation activity",
)
```

- [ ] **Step 8: Run tests and syntax checks**

Run:

```bash
cd backend
uv sync
uv run pytest tests/test_graphiti_memory_adapter.py -v
uv run python -m compileall app
```

Expected: tests pass and compileall reports no syntax errors.

- [ ] **Step 9: Commit**

```bash
git add backend/requirements.txt backend/pyproject.toml docker-compose.yml backend/app/storage/graphiti_memory.py backend/app/storage/__init__.py backend/app/services/graph_builder.py backend/app/services/graph_memory_updater.py backend/tests/test_graphiti_memory_adapter.py backend/uv.lock
git commit -m "feat: add graphiti memory on neo4j"
```

---

### Task 8: Local Memory Cleanup and Zep Removal

**Files:**
- Modify: `README.md`
- Modify: `.env.example`
- Modify: `backend/app/api/graph.py`
- Modify: `frontend/CHINESE_TEXT_INVENTORY.md`
- Modify: `backend/uv.lock` after dependency sync if `zep-cloud` remains

- [ ] **Step 1: Update `.env.example`**

```dotenv
# ===== LLM Provider Selection =====
LLM_DEFAULT_PROVIDER=ollama

# Local Ollama provider
OLLAMA_API_KEY=ollama
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=qwen2.5:32b

# MiniMax token-plan provider
MINIMAX_API_KEY=
MINIMAX_BASE_URL=https://api.minimax.io/anthropic
MINIMAX_MODEL=MiniMax-M2.7

# Kimi provider
KIMI_API_KEY=
KIMI_BASE_URL=https://api.kimi.com/coding/v1
KIMI_MODEL=kimi-k2.6

# Backward-compatible names used by older scripts
LLM_API_KEY=${OLLAMA_API_KEY}
LLM_BASE_URL=${OLLAMA_BASE_URL}
LLM_MODEL_NAME=${OLLAMA_MODEL}
```

- [ ] **Step 2: Rename stale progress copy**

Change `backend/app/api/graph.py` progress message from:

```python
message="Creating Zep graph...",
```

to:

```python
message="Creating local Neo4j graph memory...",
```

- [ ] **Step 3: Remove stale Zep dependency from lockfile**

Run:

```bash
cd backend
uv sync
```

Expected: `backend/uv.lock` no longer contains `zep-cloud` unless it is a transitive dependency of another required package.

- [ ] **Step 4: Update README memory section**

Add a concise statement:

```markdown
MiroFish memory is local by default. Entity memory, relationship memory, temporal episodes, search, and dynamic simulation updates use self-hosted Graphiti over Neo4j Community Edition. Zep Cloud is not required and no graph-memory content is sent to Zep.
```

- [ ] **Step 5: Verify Zep references are only historical**

Run: `rg -n "Zep|zep" README.md backend frontend .env.example`

Expected: no operational references that imply Zep is required.

- [ ] **Step 6: Commit**

```bash
git add README.md .env.example backend/app/api/graph.py frontend/CHINESE_TEXT_INVENTORY.md backend/uv.lock
git commit -m "docs: clarify local memory and remove zep leftovers"
```

---

### Task 9: End-to-End Verification

**Files:**
- No new production files unless failures require fixes.

- [ ] **Step 1: Backend tests**

Run:

```bash
cd backend
uv run pytest -v
```

Expected: all tests pass.

- [ ] **Step 2: Frontend build**

Run:

```bash
cd frontend
npm run build
```

Expected: build succeeds.

- [ ] **Step 3: Start local app**

Run:

```bash
npm run dev
```

Expected: backend on `http://localhost:5001`, frontend on Vite's printed URL.

- [ ] **Step 4: Verify provider metadata endpoint**

Run:

```bash
curl http://localhost:5001/api/settings/llm-providers
```

Expected: response includes `ollama`, `minimax`, and `kimi`, contains `configured` booleans, and does not contain API keys.

- [ ] **Step 5: Verify Kimi smoke call**

With `KIMI_API_KEY` set, run a small backend smoke script:

```bash
cd backend
uv run python - <<'PY'
from app.utils.llm_client import LLMClient
client = LLMClient(provider_name="kimi")
print(client.chat([{"role": "user", "content": "Reply with exactly: OK"}], max_tokens=16))
PY
```

Expected: output contains `OK`.

- [ ] **Step 6: Verify MiniMax smoke call**

With `MINIMAX_API_KEY` set, run:

```bash
cd backend
uv run python - <<'PY'
from app.utils.llm_client import LLMClient
client = LLMClient(provider_name="minimax")
print(client.chat([{"role": "user", "content": "Reply with exactly: OK"}], max_tokens=16))
PY
```

Expected: output contains `OK`.

- [ ] **Step 7: Verify local memory path**

Run a graph build from the UI using the provider picker, then query Neo4j or use Graphiti search through the backend.

Expected: Graphiti episodes, entities, and relationships are stored in local Neo4j; no Zep API key or Zep endpoint is required.

- [ ] **Step 8: Commit fixes from verification**

```bash
git add .
git commit -m "test: verify provider switching and local memory"
```

---

## Self-Review

- Spec coverage: MiniMax M2.7, Kimi K2.6, provider switching, and local Neo4j + Graphiti memory are each covered.
- Type consistency: provider name is consistently `llm_provider`; provider registry function is consistently `get_llm_provider`.
- Security: API keys stay server-side and are not returned by provider metadata endpoint.
- Risk: Kimi K2.6 may reject undocumented provider-specific request fields, so the initial Kimi provider sends no `extra_body` unless a current official doc requires it.
- Risk: CAMEL/OASIS may not support MiniMax's Anthropic-compatible endpoint directly. The provider layer needs an Anthropic Messages transport for MiniMax, while Kimi can still use the OpenAI SDK transport if the account is allowed to call the coding API.
