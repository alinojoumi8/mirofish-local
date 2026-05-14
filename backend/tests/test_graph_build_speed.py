import hashlib
import threading
import time

from app.config import Config
from app.services.graph_builder import GraphBuilderService
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
    storage._write_extraction = lambda graph_id, text, extraction, profile=None: hashlib.sha256(text.encode()).hexdigest()

    cache_context = {"file_hash": "file-123"}

    first_ids = storage.add_text_batch("graph-1", ["same chunk"], cache_context=cache_context)
    second_ids = storage.add_text_batch("graph-2", ["same chunk"], cache_context=cache_context)

    assert ner.calls == 1
    assert first_ids == second_ids
