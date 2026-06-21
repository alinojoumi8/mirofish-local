from app.services.graph_tools import GraphToolsService
from app.storage import neo4j_schema
from app.storage.neo4j_storage import Neo4jStorage


class FailingLLM:
    def chat_json(self, *args, **kwargs):
        raise AssertionError("retrieval path should not call local LLM")


class FakeSearchStorage:
    def __init__(self):
        self.calls = []

    def search(self, graph_id, query, limit=10, scope="edges"):
        self.calls.append({
            "graph_id": graph_id,
            "query": query,
            "limit": limit,
            "scope": scope,
        })
        return {
            "edges": [
                {
                    "uuid": "edge-1",
                    "name": "SUPPORTS",
                    "fact": "The motion record contains direct asset-transfer evidence.",
                    "source_node_uuid": "",
                    "target_node_uuid": "",
                }
            ],
            "nodes": [],
        }


def test_insight_forge_can_use_neo4j_only_retrieval_without_llm_expansion():
    storage = FakeSearchStorage()
    service = GraphToolsService(storage=storage, llm_client=FailingLLM())

    result = service.insight_forge(
        graph_id="graph-1",
        query="asset transfer evidence",
        simulation_requirement="predict motion outcome",
        use_llm_decomposition=False,
    )

    assert result.sub_queries == ["asset transfer evidence"]
    assert result.semantic_facts == ["The motion record contains direct asset-transfer evidence."]
    assert [call["query"] for call in storage.calls] == ["asset transfer evidence"]


class FakeSession:
    def __init__(self, calls):
        self.calls = calls

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def run(self, query, **kwargs):
        self.calls.append(query)
        return []


class FakeDriver:
    def __init__(self, calls):
        self.calls = calls

    def session(self):
        return FakeSession(self.calls)


def test_neo4j_schema_setup_runs_once_per_process_for_same_connection(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "app.storage.neo4j_storage.GraphDatabase.driver",
        lambda *args, **kwargs: FakeDriver(calls),
    )
    monkeypatch.setattr("app.storage.neo4j_storage.EmbeddingService", lambda: object())
    monkeypatch.setattr("app.storage.neo4j_storage.NERExtractor", lambda: object())

    if hasattr(Neo4jStorage, "_schema_initialized_keys"):
        Neo4jStorage._schema_initialized_keys.clear()

    Neo4jStorage(uri="bolt://unit-test", user="neo4j", password="pw")
    Neo4jStorage(uri="bolt://unit-test", user="neo4j", password="pw")

    assert len(calls) == len(neo4j_schema.ALL_SCHEMA_QUERIES)
