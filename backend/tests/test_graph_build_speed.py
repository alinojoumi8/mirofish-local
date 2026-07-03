import hashlib
import threading
import time

from app.config import Config
from app.services.graph_builder import GraphBuilderService
from app.storage.embedding_service import EmbeddingCache, EmbeddingProviderInfo, RETRIEVAL_DOCUMENT
from app.storage.extraction_cache import ExtractionCache
from app.storage.neo4j_storage import Neo4jStorage
from app.storage.ner_extractor import NERExtractor


def test_default_graph_chunking_is_large_enough_for_legal_documents():
    assert Config.DEFAULT_CHUNK_SIZE == 3000
    assert Config.DEFAULT_CHUNK_OVERLAP == 200


class FakeLLM:
    def __init__(self):
        self.calls = []

    def chat_json(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return {
            "chunks": [
                {
                    "chunk_index": 0,
                    "entities": [{"name": "Alpha Inc.", "type": "Party", "attributes": {}}],
                    "relations": [],
                },
                {
                    "chunk_index": 1,
                    "entities": [{"name": "Beta LLC", "type": "Party", "attributes": {}}],
                    "relations": [],
                },
            ]
        }


def test_extract_batch_uses_one_llm_call_and_returns_one_result_per_chunk():
    llm = FakeLLM()
    extractor = NERExtractor(llm_client=llm)

    results = extractor.extract_batch(
        ["Alpha Inc. filed evidence.", "Beta LLC responded."],
        {"entity_types": ["Party"], "edge_types": ["RESPONDED_TO"]},
    )

    assert len(llm.calls) == 1
    assert len(results) == 2
    assert results[0]["entities"][0]["name"] == "Alpha Inc."
    assert results[1]["entities"][0]["name"] == "Beta LLC"
    assert "chunk_index" in llm.calls[0]["messages"][1]["content"]


class BatchStorage:
    def __init__(self):
        self.batches = []

    def add_text(self, graph_id, text):
        raise AssertionError("GraphBuilderService should use storage.add_text_batch")

    def add_text_batch(
        self,
        graph_id,
        chunks,
        batch_size=3,
        progress_callback=None,
        cache_context=None,
        profile_callback=None,
    ):
        self.batches.append(
            {
                "graph_id": graph_id,
                "chunks": list(chunks),
                "batch_size": batch_size,
                "cache_context": cache_context,
            }
        )
        if progress_callback:
            progress_callback(1.0)
        if profile_callback:
            profile_callback({
                "llm_extraction_seconds": 0.1,
                "embedding_seconds": 0.2,
                "neo4j_write_seconds": 0.3,
                "cache_hits": 0,
                "cache_misses": len(chunks),
            })
        return [f"episode-{len(self.batches)}-{idx}" for idx, _ in enumerate(chunks)]


def test_graph_builder_batches_chunks_and_reports_real_progress_details():
    storage = BatchStorage()
    builder = GraphBuilderService(storage=storage)
    progress_events = []

    episode_ids = builder.add_text_batches(
        "graph-1",
        ["chunk-a", "chunk-b", "chunk-c", "chunk-d"],
        batch_size=2,
        progress_callback=lambda message, ratio, detail=None: progress_events.append(
            {"message": message, "ratio": ratio, "detail": detail}
        ),
        cache_context={"file_hash": "filehash", "ontology_hash": "ontologyhash"},
    )

    assert episode_ids == [
        "episode-1-0",
        "episode-1-1",
        "episode-2-0",
        "episode-2-1",
    ]
    assert [batch["chunks"] for batch in storage.batches] == [
        ["chunk-a", "chunk-b"],
        ["chunk-c", "chunk-d"],
    ]
    assert all(batch["cache_context"]["file_hash"] == "filehash" for batch in storage.batches)
    assert progress_events[-1]["detail"]["current_chunk"] == 4
    assert progress_events[-1]["detail"]["total_chunks"] == 4
    assert "avg_seconds_per_chunk" in progress_events[-1]["detail"]
    assert "eta_seconds" in progress_events[-1]["detail"]
    assert builder.last_build_profile["llm_extraction_seconds"] == 0.2
    assert builder.last_build_profile["embedding_seconds"] == 0.4
    assert builder.last_build_profile["neo4j_write_seconds"] == 0.6


class ConcurrentStorage:
    def __init__(self):
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def add_text_batch(
        self,
        graph_id,
        chunks,
        batch_size=3,
        progress_callback=None,
        cache_context=None,
        profile_callback=None,
    ):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.05)
        with self.lock:
            self.active -= 1
        if profile_callback:
            profile_callback({
                "llm_extraction_seconds": 0.05,
                "embedding_seconds": 0.0,
                "neo4j_write_seconds": 0.0,
                "cache_hits": 0,
                "cache_misses": len(chunks),
            })
        return [f"{chunk}-episode" for chunk in chunks]


def test_graph_builder_uses_configured_concurrency_and_preserves_episode_order(monkeypatch):
    monkeypatch.setattr(Config, "GRAPH_BUILD_LLM_CONCURRENCY", 2, raising=False)
    storage = ConcurrentStorage()
    builder = GraphBuilderService(storage=storage)

    episode_ids = builder.add_text_batches(
        "graph-1",
        ["chunk-a", "chunk-b", "chunk-c", "chunk-d"],
        batch_size=1,
    )

    assert storage.max_active == 2
    assert episode_ids == [
        "chunk-a-episode",
        "chunk-b-episode",
        "chunk-c-episode",
        "chunk-d-episode",
    ]
    assert builder.last_build_profile["llm_concurrency"] == 2


def test_graph_builder_uses_moderate_cloud_concurrency_by_default(monkeypatch):
    monkeypatch.setattr(Config, "GRAPH_BUILD_LLM_CONCURRENCY", 0, raising=False)
    monkeypatch.setattr(Config, "GRAPH_BUILD_CLOUD_LLM_CONCURRENCY", 5, raising=False)
    monkeypatch.setattr(Config, "LLM_DEFAULT_PROVIDER", "minimax", raising=False)
    monkeypatch.setattr(Config, "LLM_BASE_URL", "https://api.minimax.io/anthropic", raising=False)

    builder = GraphBuilderService(storage=BatchStorage())

    assert builder._resolve_llm_concurrency(total_batches=20) == 5


def test_graph_builder_packs_batches_by_chunk_count_and_character_budget(monkeypatch):
    monkeypatch.setattr(Config, "GRAPH_BUILD_MAX_BATCH_CHARS", 10, raising=False)
    builder = GraphBuilderService(storage=BatchStorage())

    jobs = builder._build_batch_jobs(["aaaa", "bbbb", "cccccccccccc", "dd"], batch_size=3)

    assert [job["chunks"] for job in jobs] == [
        ["aaaa", "bbbb"],
        ["cccccccccccc"],
        ["dd"],
    ]
    assert [job["first_chunk"] for job in jobs] == [1, 3, 4]
    assert [job["last_chunk"] for job in jobs] == [2, 3, 4]


class CacheNER:
    def __init__(self):
        self.calls = 0

    def extract_batch(self, chunks, ontology):
        self.calls += 1
        return [
            {
                "entities": [{"name": f"Entity {idx}", "type": "Party", "attributes": {}}],
                "relations": [],
            }
            for idx, _ in enumerate(chunks)
        ]


def test_neo4j_batch_extraction_cache_skips_llm_on_rebuild(tmp_path):
    ner = CacheNER()
    storage = Neo4jStorage.__new__(Neo4jStorage)
    storage._ner = ner
    storage._extraction_cache = ExtractionCache(tmp_path)
    storage._assert_embedding_compatible = lambda graph_id: None
    storage.get_ontology = lambda graph_id: {"entity_types": ["Party"], "edge_types": []}
    storage._write_extractions_batch = (
        lambda graph_id, chunks, extractions, profile=None:
        [hashlib.sha256(text.encode()).hexdigest() for text in chunks]
    )

    cache_context = {"file_hash": "file-123"}

    first_ids = storage.add_text_batch("graph-1", ["same chunk"], cache_context=cache_context)
    second_ids = storage.add_text_batch("graph-2", ["same chunk"], cache_context=cache_context)

    assert ner.calls == 1
    assert first_ids == second_ids


def test_extraction_cache_set_ignores_unwritable_cache_shard(tmp_path):
    cache = ExtractionCache(tmp_path)
    file_hash = "file-123"
    chunk_hash = ExtractionCache.hash_text("same chunk")
    ontology_hash = ExtractionCache.hash_ontology({"entity_types": ["Party"], "edge_types": []})
    cache_path = cache._path(file_hash, chunk_hash, ontology_hash)

    cache_path.parent.write_text("not a directory", encoding="utf-8")

    cache.set(
        file_hash,
        chunk_hash,
        ontology_hash,
        {"entities": [{"name": "Entity", "type": "Party"}], "relations": []},
    )

    assert cache.get(file_hash, chunk_hash, ontology_hash) is None


def test_embedding_cache_persists_vectors_by_provider_model_task_and_text(tmp_path):
    cache = EmbeddingCache(tmp_path)
    provider = EmbeddingProviderInfo(
        provider="ollama",
        model="nomic-embed-text",
        dimensions=768,
        base_url="http://ollama:11434",
    )
    vector = [0.1] * 768

    cache.set(provider.provider_id, "RETRIEVAL_DOCUMENT", "Alpha Inc.", vector)

    assert cache.get(provider.provider_id, "RETRIEVAL_DOCUMENT", "Alpha Inc.") == vector
    assert cache.get(provider.provider_id, "RETRIEVAL_QUERY", "Alpha Inc.") is None
    assert cache.get("other-provider", "RETRIEVAL_DOCUMENT", "Alpha Inc.") is None


def test_embedding_cache_set_ignores_unwritable_cache_shard(tmp_path):
    cache = EmbeddingCache(tmp_path)
    provider_id = "ollama:nomic-embed-text:768"
    text = "Alpha Inc."
    cache_path = cache._path(provider_id, RETRIEVAL_DOCUMENT, text)

    cache_path.parent.write_text("not a directory", encoding="utf-8")

    cache.set(provider_id, RETRIEVAL_DOCUMENT, text, [0.1] * 768)

    assert cache.get(provider_id, RETRIEVAL_DOCUMENT, text) is None


class BatchEmbedding:
    def __init__(self):
        self.calls = []

    @property
    def info(self):
        return EmbeddingProviderInfo(
            provider="fake",
            model="fake-embed",
            dimensions=768,
            base_url="memory://fake",
        )

    def embed_batch(self, texts, batch_size=32, task_type="RETRIEVAL_DOCUMENT"):
        self.calls.append(list(texts))
        return [[float(i)] * 768 for i, _text in enumerate(texts)]


class FakeSession:
    def __init__(self):
        self.queries = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute_write(self, func):
        return func(self)

    def run(self, query, **kwargs):
        self.queries.append((query, kwargs))
        rows = kwargs.get("rows", [])
        if "RETURN row.name_lower AS name_lower" in query:
            return [
                {"name_lower": row["name_lower"], "uuid": row["uuid"]}
                for row in rows
            ]
        return []


class FakeDriver:
    def __init__(self):
        self.session_obj = FakeSession()

    def session(self):
        return self.session_obj


def test_neo4j_batch_writer_embeds_once_and_bulk_writes_all_chunks():
    storage = Neo4jStorage.__new__(Neo4jStorage)
    storage._driver = FakeDriver()
    storage._embedding = BatchEmbedding()
    storage._call_with_retry = lambda func, *args, **kwargs: func(*args, **kwargs)

    ids = storage._write_extractions_batch(
        "graph-1",
        ["chunk one", "chunk two"],
        [
            {
                "entities": [{"name": "Alpha Inc.", "type": "Company", "attributes": {}}],
                "relations": [],
            },
            {
                "entities": [{"name": "Beta LLC", "type": "Company", "attributes": {}}],
                "relations": [
                    {
                        "source": "Beta LLC",
                        "target": "Alpha Inc.",
                        "type": "RELATED_TO",
                        "fact": "Beta LLC is related to Alpha Inc.",
                    }
                ],
            },
        ],
        profile={},
    )

    assert len(ids) == 2
    assert storage._embedding.calls == [[
        "Alpha Inc. (Company)",
        "Beta LLC (Company)",
        "Beta LLC is related to Alpha Inc.",
    ]]
    queries = [query for query, _kwargs in storage._driver.session_obj.queries]
    assert any("UNWIND $rows AS row" in query and "CREATE (ep:Episode" in query for query in queries)
    assert any("UNWIND $rows AS row" in query and "MERGE (n:Entity" in query for query in queries)
    assert any("UNWIND $rows AS row" in query and "CREATE (src)-[r:RELATION" in query for query in queries)
