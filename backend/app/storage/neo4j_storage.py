"""
Neo4jStorage — Neo4j Community Edition implementation of GraphStorage.

Replaces all Zep Cloud API calls with local Neo4j Cypher queries.
Includes: CRUD, NER/RE-based text ingestion, hybrid search, retry logic.
"""

import json
import time
import uuid
import logging
import threading
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Callable

from neo4j import GraphDatabase, Session as Neo4jSession
from neo4j.exceptions import (
    TransientError,
    ServiceUnavailable,
    SessionExpired,
)

from ..config import Config
from .graph_storage import GraphStorage
from .embedding_service import EmbeddingError, EmbeddingService
from .extraction_cache import ExtractionCache
from .ner_extractor import NERExtractor
from .search_service import SearchService
from . import neo4j_schema

logger = logging.getLogger('mirofish.neo4j_storage')


class Neo4jStorage(GraphStorage):
    """Neo4j CE implementation of the GraphStorage interface."""

    MAX_RETRIES = 3
    RETRY_DELAY_BASE = 1  # seconds
    _schema_init_lock = threading.Lock()
    _schema_initialized_keys = set()

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        embedding_service: Optional[EmbeddingService] = None,
        ner_extractor: Optional[NERExtractor] = None,
    ):
        self._uri = uri or Config.NEO4J_URI
        self._user = user or Config.NEO4J_USER
        self._password = password or Config.NEO4J_PASSWORD

        self._driver = GraphDatabase.driver(
            self._uri, auth=(self._user, self._password)
        )
        self._embedding = embedding_service or EmbeddingService()
        self._ner = ner_extractor or NERExtractor()
        self._extraction_cache = (
            ExtractionCache() if Config.GRAPH_EXTRACTION_CACHE_ENABLED else None
        )
        self._search = SearchService(self._embedding)

        # Initialize schema (indexes, constraints)
        self._ensure_schema()

    def close(self):
        """Close the Neo4j driver connection."""
        self._driver.close()

    def health_status(self) -> Dict[str, Any]:
        """Detailed Neo4j and vector-search readiness status."""
        status: Dict[str, Any] = {
            "healthy": False,
            "uri": self._uri,
            "edition": None,
            "version": None,
            "error": None,
            "embedding": self._embedding.health_status(),
            "vector_search_usable": False,
        }

        try:
            with self._driver.session() as session:
                record = session.run(
                    """
                    CALL dbms.components()
                    YIELD name, versions, edition
                    RETURN versions[0] AS version, edition
                    LIMIT 1
                    """
                ).single()
                if record:
                    status["version"] = record["version"]
                    status["edition"] = record["edition"]

                index_records = list(session.run(
                    """
                    SHOW INDEXES
                    YIELD name, type, state
                    WHERE name IN ['entity_embedding', 'fact_embedding']
                    RETURN name, type, state
                    """
                ))
                status["vector_indexes"] = [
                    {"name": rec["name"], "type": rec["type"], "state": rec["state"]}
                    for rec in index_records
                ]
                status["vector_search_usable"] = (
                    status["embedding"].get("healthy") is True
                    and len(index_records) == 2
                    and all(rec["state"] == "ONLINE" for rec in index_records)
                    and self._embedding.dimensions == neo4j_schema.VECTOR_DIMENSIONS
                )

            status["healthy"] = True
        except Exception as exc:
            status["error"] = str(exc)

        return status

    def _ensure_schema(self):
        """Create indexes and constraints if they don't exist."""
        schema_key = (self._uri, self._user)
        with self._schema_init_lock:
            if schema_key in self._schema_initialized_keys:
                return

            with self._driver.session() as session:
                for query in neo4j_schema.ALL_SCHEMA_QUERIES:
                    try:
                        session.run(query)
                    except Exception as e:
                        logger.warning(f"Schema query warning (may already exist): {e}")

            self._schema_initialized_keys.add(schema_key)

    # ----------------------------------------------------------------
    # Retry wrapper
    # ----------------------------------------------------------------

    def _call_with_retry(self, func, *args, **kwargs):
        """
        Execute a function with retry on Neo4j transient errors.
        Replaces 3 different retry patterns from the Zep codebase.
        """
        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except (TransientError, ServiceUnavailable, SessionExpired) as e:
                last_error = e
                wait = self.RETRY_DELAY_BASE * (2 ** attempt)
                logger.warning(
                    f"Neo4j transient error (attempt {attempt + 1}/{self.MAX_RETRIES}), "
                    f"retrying in {wait}s: {e}"
                )
                time.sleep(wait)
            except Exception:
                raise

        raise last_error  # type: ignore

    # ----------------------------------------------------------------
    # Graph lifecycle
    # ----------------------------------------------------------------

    def create_graph(self, name: str, description: str = "") -> str:
        graph_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        embedding_info = self._embedding.info

        def _create(tx):
            tx.run(
                """
                CREATE (g:Graph {
                    graph_id: $graph_id,
                    name: $name,
                    description: $description,
                    ontology_json: '{}',
                    embedding_provider: $embedding_provider,
                    embedding_model: $embedding_model,
                    embedding_dimensions: $embedding_dimensions,
                    embedding_provider_id: $embedding_provider_id,
                    created_at: $created_at
                })
                """,
                graph_id=graph_id,
                name=name,
                description=description,
                embedding_provider=embedding_info.provider,
                embedding_model=embedding_info.model,
                embedding_dimensions=embedding_info.dimensions,
                embedding_provider_id=embedding_info.provider_id,
                created_at=now,
            )

        with self._driver.session() as session:
            self._call_with_retry(session.execute_write, _create)

        logger.info(f"Created graph '{name}' with id {graph_id}")
        return graph_id

    def delete_graph(self, graph_id: str) -> None:
        def _delete(tx):
            # Delete all entities and their relationships
            tx.run(
                "MATCH (n {graph_id: $gid}) DETACH DELETE n",
                gid=graph_id,
            )
            # Delete graph node
            tx.run(
                "MATCH (g:Graph {graph_id: $gid}) DELETE g",
                gid=graph_id,
            )

        with self._driver.session() as session:
            self._call_with_retry(session.execute_write, _delete)
        logger.info(f"Deleted graph {graph_id}")

    def set_ontology(self, graph_id: str, ontology: Dict[str, Any]) -> None:
        def _set(tx):
            tx.run(
                """
                MATCH (g:Graph {graph_id: $gid})
                SET g.ontology_json = $ontology_json
                """,
                gid=graph_id,
                ontology_json=json.dumps(ontology, ensure_ascii=False),
            )

        with self._driver.session() as session:
            self._call_with_retry(session.execute_write, _set)

    def get_ontology(self, graph_id: str) -> Dict[str, Any]:
        with self._driver.session() as session:
            result = session.run(
                "MATCH (g:Graph {graph_id: $gid}) RETURN g.ontology_json AS oj",
                gid=graph_id,
            )
            record = result.single()
            if record and record["oj"]:
                return json.loads(record["oj"])
            return {}

    def get_graph_embedding_info(self, graph_id: str) -> Dict[str, Any]:
        """Return embedding metadata stored on the graph node."""
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (g:Graph {graph_id: $gid})
                RETURN g.embedding_provider AS provider,
                       g.embedding_model AS model,
                       g.embedding_dimensions AS dimensions,
                       g.embedding_provider_id AS provider_id
                """,
                gid=graph_id,
            )
            record = result.single()
            if not record:
                raise ValueError(f"Graph does not exist: {graph_id}")
            return {
                "provider": record["provider"],
                "model": record["model"],
                "dimensions": record["dimensions"],
                "provider_id": record["provider_id"],
            }

    def get_embedding_status(self, graph_id: str) -> Dict[str, Any]:
        """Return vector coverage and report-readiness for a graph."""
        graph_info = self.get_graph_embedding_info(graph_id)
        current = self._embedding.info
        compatible = self._is_graph_embedding_compatible(graph_info)

        with self._driver.session() as session:
            node_record = session.run(
                """
                MATCH (n:Entity {graph_id: $gid})
                RETURN count(n) AS total,
                       sum(CASE
                           WHEN n.embedding IS NOT NULL AND size(n.embedding) = $dims
                           THEN 1 ELSE 0 END) AS embedded
                """,
                gid=graph_id,
                dims=current.dimensions,
            ).single()
            rel_record = session.run(
                """
                MATCH ()-[r:RELATION {graph_id: $gid}]->()
                RETURN count(r) AS total,
                       sum(CASE
                           WHEN r.fact_embedding IS NOT NULL AND size(r.fact_embedding) = $dims
                           THEN 1 ELSE 0 END) AS embedded
                """,
                gid=graph_id,
                dims=current.dimensions,
            ).single()

        node_count = int(node_record["total"] or 0)
        nodes_embedded = int(node_record["embedded"] or 0)
        relationship_count = int(rel_record["total"] or 0)
        relationships_embedded = int(rel_record["embedded"] or 0)
        nodes_missing = node_count - nodes_embedded
        relationships_missing = relationship_count - relationships_embedded
        total_items = node_count + relationship_count
        embedded_items = nodes_embedded + relationships_embedded

        issues: List[str] = []
        if not compatible:
            issues.append(
                "Graph embedding provider does not match the active provider; rebuild or re-embed before reporting."
            )
        if total_items == 0:
            issues.append("Graph has no entities or relationships to report on.")
        if nodes_missing or relationships_missing:
            issues.append(
                f"Missing embeddings: {nodes_missing}/{node_count} nodes and "
                f"{relationships_missing}/{relationship_count} relationships."
            )

        return {
            "graph_id": graph_id,
            "provider": graph_info.get("provider"),
            "model": graph_info.get("model"),
            "dimensions": graph_info.get("dimensions"),
            "provider_id": graph_info.get("provider_id"),
            "current_provider_id": current.provider_id,
            "compatible": compatible,
            "node_count": node_count,
            "nodes_embedded": nodes_embedded,
            "nodes_missing": nodes_missing,
            "relationship_count": relationship_count,
            "relationships_embedded": relationships_embedded,
            "relationships_missing": relationships_missing,
            "vector_coverage": embedded_items / total_items if total_items else 0.0,
            "safe_to_report": compatible and total_items > 0 and not nodes_missing and not relationships_missing,
            "issues": issues,
        }

    def get_graph_quality(self, graph_id: str) -> Dict[str, Any]:
        """Score whether the graph is structurally rich enough for report synthesis."""
        embedding_status = self.get_embedding_status(graph_id)

        with self._driver.session() as session:
            node_record = session.run(
                """
                MATCH (n:Entity {graph_id: $gid})
                RETURN count(n) AS total,
                       sum(CASE
                           WHEN coalesce(trim(n.summary), '') = ''
                           THEN 1 ELSE 0 END) AS empty_summaries
                """,
                gid=graph_id,
            ).single()
            isolated_record = session.run(
                """
                MATCH (n:Entity {graph_id: $gid})
                WHERE NOT (n)-[:RELATION]-()
                RETURN count(n) AS isolated
                """,
                gid=graph_id,
            ).single()
            rel_record = session.run(
                """
                MATCH ()-[r:RELATION {graph_id: $gid}]->()
                RETURN count(r) AS total,
                       count(DISTINCT coalesce(r.name, '')) AS relation_type_count,
                       sum(CASE
                           WHEN coalesce(trim(r.fact), '') = ''
                           THEN 1 ELSE 0 END) AS empty_facts
                """,
                gid=graph_id,
            ).single()
            type_record = session.run(
                """
                MATCH (n:Entity {graph_id: $gid})
                UNWIND labels(n) AS lbl
                WITH lbl WHERE lbl <> 'Entity'
                RETURN count(DISTINCT lbl) AS entity_type_count
                """,
                gid=graph_id,
            ).single()
            duplicate_record = session.run(
                """
                MATCH (n:Entity {graph_id: $gid})
                WITH coalesce(n.name_lower, toLower(n.name), '') AS key, count(n) AS count
                WHERE key <> '' AND count > 1
                RETURN count(*) AS duplicate_name_groups, sum(count) AS duplicate_nodes
                """,
                gid=graph_id,
            ).single()
            episode_record = session.run(
                """
                MATCH (e:Episode {graph_id: $gid})
                RETURN count(e) AS episode_count
                """,
                gid=graph_id,
            ).single()
            try:
                uncited_fact_record = session.run(
                    """
                    MATCH ()-[r:RELATION {graph_id: $gid}]->()
                    WHERE r.episode_ids IS NULL OR size(r.episode_ids) = 0
                    RETURN count(r) AS uncited_facts
                    """,
                    gid=graph_id,
                ).single()
            except Exception:
                uncited_fact_record = {"uncited_facts": 0}

        node_count = int(node_record["total"] or 0)
        edge_count = int(rel_record["total"] or 0)
        empty_summaries = int(node_record["empty_summaries"] or 0)
        empty_facts = int(rel_record["empty_facts"] or 0)
        isolated_nodes = int(isolated_record["isolated"] or 0)
        entity_type_count = int(type_record["entity_type_count"] or 0)
        relation_type_count = int(rel_record["relation_type_count"] or 0)
        duplicate_name_groups = int(duplicate_record["duplicate_name_groups"] or 0)
        duplicate_nodes = int(duplicate_record["duplicate_nodes"] or 0)
        episode_count = int(episode_record["episode_count"] or 0)
        uncited_facts = int(uncited_fact_record["uncited_facts"] or 0)

        relation_density = edge_count / node_count if node_count else 0.0
        isolated_ratio = isolated_nodes / node_count if node_count else 0.0
        empty_summary_ratio = empty_summaries / node_count if node_count else 0.0
        empty_fact_ratio = empty_facts / edge_count if edge_count else 0.0

        deductions: List[Dict[str, Any]] = []

        def deduct(points: int, code: str, message: str):
            deductions.append({"points": points, "code": code, "message": message})

        if node_count == 0:
            deduct(40, "no_entities", "Graph has no entities.")
        elif node_count < 5:
            deduct(18, "low_entity_count", f"Only {node_count} entities were extracted.")

        if edge_count == 0:
            deduct(32, "no_relationships", "Graph has no relationships.")
        elif edge_count < 3:
            deduct(14, "low_relationship_count", f"Only {edge_count} relationships were extracted.")

        if node_count and relation_density < 0.5:
            deduct(12, "low_relation_density", f"Relation density is {relation_density:.2f} edges per entity.")

        if isolated_ratio > 0.35:
            deduct(12, "high_isolated_nodes", f"{isolated_nodes}/{node_count} entities are isolated.")

        if empty_summary_ratio > 0:
            deduct(8, "empty_entity_summaries", f"{empty_summaries}/{node_count} entities have empty summaries.")

        if empty_fact_ratio > 0:
            deduct(8, "empty_relationship_facts", f"{empty_facts}/{edge_count} relationships have empty facts.")

        if node_count >= 5 and entity_type_count < 2:
            deduct(6, "low_entity_type_diversity", "Graph uses fewer than two entity types.")

        if edge_count >= 3 and relation_type_count < 2:
            deduct(6, "low_relation_type_diversity", "Graph uses fewer than two relationship types.")

        if duplicate_name_groups:
            deduct(6, "duplicate_entity_names", f"{duplicate_nodes} entities share duplicate normalized names.")

        score = max(0, 100 - sum(item["points"] for item in deductions))
        status = "pass"
        if score < 50:
            status = "fail"
        elif score < 75:
            status = "warn"

        return {
            "graph_id": graph_id,
            "score": score,
            "status": status,
            "safe_to_report": bool(embedding_status.get("safe_to_report") and score >= 50),
            "node_count": node_count,
            "edge_count": edge_count,
            "episode_count": episode_count,
            "entity_type_count": entity_type_count,
            "relation_type_count": relation_type_count,
            "relation_density": relation_density,
            "isolated_nodes": isolated_nodes,
            "isolated_ratio": isolated_ratio,
            "empty_summaries": empty_summaries,
            "empty_facts": empty_facts,
            "duplicate_name_groups": duplicate_name_groups,
            "duplicate_nodes": duplicate_nodes,
            "duplicate_entities": duplicate_nodes,
            "orphan_facts": empty_facts,
            "uncited_facts": uncited_facts,
            "search_readiness": {
                "ready": bool(embedding_status.get("safe_to_report")),
                "vector_coverage": embedding_status.get("vector_coverage", 0.0),
                "issues": embedding_status.get("issues", []),
            },
            "embedding": embedding_status,
            "deductions": deductions,
        }

    def reembed_graph(self, graph_id: str, batch_size: int = 32) -> Dict[str, Any]:
        """Backfill entity and relationship embeddings for a graph."""
        status = self.get_embedding_status(graph_id)
        if not status.get("compatible"):
            raise EmbeddingError("; ".join(status.get("issues") or ["Graph embedding provider mismatch"]))

        targets = self._read_embedding_targets(graph_id)
        nodes = targets["nodes"]
        relationships = targets["relationships"]
        texts = [item["text"] for item in nodes] + [item["text"] for item in relationships]
        vectors = self._embedding.embed_batch(texts, batch_size=batch_size) if texts else []

        node_vectors = vectors[:len(nodes)]
        relationship_vectors = vectors[len(nodes):]
        node_updates = [
            {"uuid": item["uuid"], "embedding": vector}
            for item, vector in zip(nodes, node_vectors)
        ]
        relationship_updates = [
            {"uuid": item["uuid"], "embedding": vector}
            for item, vector in zip(relationships, relationship_vectors)
        ]

        self._write_embeddings(graph_id, node_updates, relationship_updates)
        self._update_graph_embedding_metadata(graph_id)
        after = self.get_embedding_status(graph_id)
        return {
            "graph_id": graph_id,
            "embedded_nodes": len(node_updates),
            "embedded_relationships": len(relationship_updates),
            "status": after,
        }

    def _read_embedding_targets(self, graph_id: str) -> Dict[str, List[Dict[str, str]]]:
        current = self._embedding.info
        with self._driver.session() as session:
            node_records = session.run(
                """
                MATCH (n:Entity {graph_id: $gid})
                WHERE n.embedding IS NULL OR size(n.embedding) <> $dims
                RETURN n.uuid AS uuid,
                       trim(coalesce(n.name, '') + ': ' + coalesce(n.summary, '')) AS text
                """,
                gid=graph_id,
                dims=current.dimensions,
            )
            rel_records = session.run(
                """
                MATCH ()-[r:RELATION {graph_id: $gid}]->()
                WHERE r.fact_embedding IS NULL OR size(r.fact_embedding) <> $dims
                RETURN r.uuid AS uuid, coalesce(r.fact, r.name, '') AS text
                """,
                gid=graph_id,
                dims=current.dimensions,
            )
            return {
                "nodes": [
                    {"uuid": record["uuid"], "text": record["text"] or record["uuid"]}
                    for record in node_records
                ],
                "relationships": [
                    {"uuid": record["uuid"], "text": record["text"] or record["uuid"]}
                    for record in rel_records
                ],
            }

    def _write_embeddings(
        self,
        graph_id: str,
        nodes: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
    ) -> None:
        with self._driver.session() as session:
            if nodes:
                session.run(
                    """
                    UNWIND $rows AS row
                    MATCH (n:Entity {graph_id: $gid, uuid: row.uuid})
                    SET n.embedding = row.embedding
                    """,
                    gid=graph_id,
                    rows=nodes,
                )
            if relationships:
                session.run(
                    """
                    UNWIND $rows AS row
                    MATCH ()-[r:RELATION {graph_id: $gid, uuid: row.uuid}]->()
                    SET r.fact_embedding = row.embedding
                    """,
                    gid=graph_id,
                    rows=relationships,
                )

    def _update_graph_embedding_metadata(self, graph_id: str) -> None:
        info = self._embedding.info
        with self._driver.session() as session:
            session.run(
                """
                MATCH (g:Graph {graph_id: $gid})
                SET g.embedding_provider = $provider,
                    g.embedding_model = $model,
                    g.embedding_dimensions = $dimensions,
                    g.embedding_provider_id = $provider_id
                """,
                gid=graph_id,
                provider=info.provider,
                model=info.model,
                dimensions=info.dimensions,
                provider_id=info.provider_id,
            )

    def _assert_embedding_compatible(self, graph_id: str) -> None:
        """Prevent querying or appending vectors from a different embedding space."""
        graph_info = self.get_graph_embedding_info(graph_id)
        current = self._embedding.info
        if self._is_graph_embedding_compatible(graph_info):
            return

        if not graph_info.get("provider_id"):
            raise EmbeddingError(
                "Graph has no embedding provider metadata and cannot be safely searched "
                f"with {current.provider_id}. Rebuild or re-embed the graph first."
            )
        raise EmbeddingError(
            "Graph embedding provider mismatch. "
            f"Graph was built with {graph_info['provider_id']}, current provider is {current.provider_id}. "
            "Rebuild or re-embed the graph before searching or appending text."
        )

    def _is_graph_embedding_compatible(self, graph_info: Dict[str, Any]) -> bool:
        current = self._embedding.info
        provider_id = graph_info.get("provider_id")
        if provider_id:
            return provider_id == current.provider_id
        legacy_provider_id = "ollama:nomic-embed-text:768"
        return current.provider_id == legacy_provider_id

    # ----------------------------------------------------------------
    # Add data (NER → nodes/edges)
    # ----------------------------------------------------------------

    def add_text(self, graph_id: str, text: str) -> str:
        """Process text: NER/RE → batch embed → create nodes/edges → return episode_id."""
        self._assert_embedding_compatible(graph_id)
        ontology = self.get_ontology(graph_id)

        logger.info(f"[add_text] Starting NER extraction for chunk ({len(text)} chars)...")
        extraction = self._ner.extract(text, ontology)
        logger.info(
            "[add_text] NER done: %s entities, %s relations",
            len(extraction.get("entities", [])),
            len(extraction.get("relations", [])),
        )
        return self._write_extraction(graph_id, text, extraction)

    def _write_extraction(
        self,
        graph_id: str,
        text: str,
        extraction: Dict[str, Any],
        profile: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Write a precomputed extraction to Neo4j and return the episode id."""
        episode_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        entities = extraction.get("entities", [])
        relations = extraction.get("relations", [])

        entity_summaries = [f"{e['name']} ({e['type']})" for e in entities]
        fact_texts = [r.get("fact", f"{r['source']} {r['type']} {r['target']}") for r in relations]
        all_texts_to_embed = entity_summaries + fact_texts

        all_embeddings: list = []
        embedding_start = time.perf_counter()
        if all_texts_to_embed:
            logger.info(f"[add_text] Batch-embedding {len(all_texts_to_embed)} texts...")
            try:
                all_embeddings = self._embedding.embed_batch(all_texts_to_embed)
            except Exception as e:
                raise EmbeddingError(
                    "Batch embedding failed. Graph build stopped to avoid storing empty vectors. "
                    f"Provider={self._embedding.provider_id()} Error={e}"
                ) from e
        if profile is not None:
            profile["embedding_seconds"] = profile.get("embedding_seconds", 0.0) + (
                time.perf_counter() - embedding_start
            )

        entity_embeddings = all_embeddings[:len(entities)]
        relation_embeddings = all_embeddings[len(entities):]
        logger.info("[add_text] Embedding done, writing to Neo4j...")

        write_start = time.perf_counter()
        with self._driver.session() as session:
            def _create_episode(tx):
                tx.run(
                    """
                    CREATE (ep:Episode {
                        uuid: $uuid,
                        graph_id: $graph_id,
                        data: $data,
                        processed: true,
                        created_at: $created_at
                    })
                    """,
                    uuid=episode_id,
                    graph_id=graph_id,
                    data=text,
                    created_at=now,
                )

            self._call_with_retry(session.execute_write, _create_episode)

            entity_uuid_map: Dict[str, str] = {}
            for idx, entity in enumerate(entities):
                ename = entity["name"]
                etype = entity["type"]
                attrs = entity.get("attributes", {})
                summary_text = entity_summaries[idx]
                embedding = entity_embeddings[idx] if idx < len(entity_embeddings) else []

                e_uuid = str(uuid.uuid4())
                entity_uuid_map[ename.lower()] = e_uuid

                def _merge_entity(tx, _uuid=e_uuid, _name=ename, _type=etype,
                                  _attrs=attrs, _embedding=embedding,
                                  _summary=summary_text, _now=now):
                    result = tx.run(
                        """
                        MERGE (n:Entity {graph_id: $gid, name_lower: $name_lower})
                        ON CREATE SET
                            n.uuid = $uuid,
                            n.name = $name,
                            n.summary = $summary,
                            n.attributes_json = $attrs_json,
                            n.embedding = $embedding,
                            n.created_at = $now
                        ON MATCH SET
                            n.summary = CASE WHEN n.summary = '' OR n.summary IS NULL
                                THEN $summary ELSE n.summary END,
                            n.attributes_json = $attrs_json,
                            n.embedding = $embedding
                        RETURN n.uuid AS uuid
                        """,
                        gid=graph_id,
                        name_lower=_name.lower(),
                        uuid=_uuid,
                        name=_name,
                        summary=_summary,
                        attrs_json=json.dumps(_attrs, ensure_ascii=False),
                        embedding=_embedding,
                        now=_now,
                    )
                    record = result.single()
                    return record["uuid"] if record else _uuid

                actual_uuid = self._call_with_retry(session.execute_write, _merge_entity)
                entity_uuid_map[ename.lower()] = actual_uuid

                if etype and etype != "Entity":
                    try:
                        def _add_label(tx, _name_lower=ename.lower()):
                            tx.run(
                                f"MATCH (n:Entity {{graph_id: $gid, name_lower: $nl}}) SET n:`{etype}`",
                                gid=graph_id,
                                nl=_name_lower,
                            )
                        self._call_with_retry(session.execute_write, _add_label)
                    except Exception as e:
                        logger.warning(f"Failed to add label '{etype}' to '{ename}': {e}")

            for idx, relation in enumerate(relations):
                source_name = relation["source"]
                target_name = relation["target"]
                rtype = relation["type"]
                fact = relation["fact"]

                source_uuid = entity_uuid_map.get(source_name.lower())
                target_uuid = entity_uuid_map.get(target_name.lower())

                if not source_uuid or not target_uuid:
                    logger.warning(
                        "Skipping relation %s->%s: entity not found in extraction results",
                        source_name,
                        target_name,
                    )
                    continue

                fact_embedding = relation_embeddings[idx] if idx < len(relation_embeddings) else []
                r_uuid = str(uuid.uuid4())

                def _create_relation(tx, _r_uuid=r_uuid, _source_uuid=source_uuid,
                                     _target_uuid=target_uuid, _rtype=rtype,
                                     _fact=fact, _fact_emb=fact_embedding,
                                     _episode_id=episode_id, _now=now):
                    tx.run(
                        """
                        MATCH (src:Entity {uuid: $src_uuid})
                        MATCH (tgt:Entity {uuid: $tgt_uuid})
                        CREATE (src)-[r:RELATION {
                            uuid: $uuid,
                            graph_id: $gid,
                            name: $name,
                            fact: $fact,
                            fact_embedding: $fact_embedding,
                            attributes_json: '{}',
                            episode_ids: [$episode_id],
                            created_at: $now,
                            valid_at: null,
                            invalid_at: null,
                            expired_at: null
                        }]->(tgt)
                        """,
                        src_uuid=_source_uuid,
                        tgt_uuid=_target_uuid,
                        uuid=_r_uuid,
                        gid=graph_id,
                        name=_rtype,
                        fact=_fact,
                        fact_embedding=_fact_emb,
                        episode_id=_episode_id,
                        now=_now,
                    )

                self._call_with_retry(session.execute_write, _create_relation)
        if profile is not None:
            profile["neo4j_write_seconds"] = profile.get("neo4j_write_seconds", 0.0) + (
                time.perf_counter() - write_start
            )

        logger.info(f"[add_text] Chunk done: episode={episode_id}")
        return episode_id

    def add_text_batch(
        self,
        graph_id: str,
        chunks: List[str],
        batch_size: int = 3,
        progress_callback: Optional[Callable] = None,
        cache_context: Optional[Dict[str, Any]] = None,
        profile_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        chunk_metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """Batch-add text chunks with cached batched NER extraction."""
        batch_started = time.perf_counter()
        profile: Dict[str, Any] = {
            "llm_extraction_seconds": 0.0,
            "embedding_seconds": 0.0,
            "neo4j_write_seconds": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
        }
        self._assert_embedding_compatible(graph_id)
        ontology = self.get_ontology(graph_id)
        ontology_hash = (cache_context or {}).get("ontology_hash") or ExtractionCache.hash_ontology(ontology)
        file_hash = (cache_context or {}).get("file_hash") or ExtractionCache.hash_text("\n".join(chunks))

        episode_ids = []
        total = len(chunks) or 1
        extractions: Dict[int, Dict[str, Any]] = {}
        missing: List[tuple[int, str, str]] = []

        for idx, chunk in enumerate(chunks):
            if not chunk or not chunk.strip():
                extractions[idx] = {"entities": [], "relations": []}
                continue
            chunk_hash = ExtractionCache.hash_text(chunk)
            cached = (
                self._extraction_cache.get(file_hash, chunk_hash, ontology_hash)
                if self._extraction_cache
                else None
            )
            if cached is not None:
                logger.info("[add_text_batch] NER cache hit for chunk %s/%s", idx + 1, len(chunks))
                extractions[idx] = cached
                profile["cache_hits"] += 1
            else:
                profile["cache_misses"] += 1
                missing.append((idx, chunk, chunk_hash))

        if missing:
            logger.info(
                "[add_text_batch] Batch NER for %s uncached chunks (%s cache hits)",
                len(missing),
                len(chunks) - len(missing),
            )
            miss_chunks = [chunk for _, chunk, _ in missing]
            llm_started = time.perf_counter()
            if hasattr(self._ner, "extract_batch"):
                miss_extractions = self._ner.extract_batch(miss_chunks, ontology)
            else:
                miss_extractions = [self._ner.extract(chunk, ontology) for chunk in miss_chunks]
            profile["llm_extraction_seconds"] += time.perf_counter() - llm_started

            for (idx, _chunk, chunk_hash), extraction in zip(missing, miss_extractions):
                extractions[idx] = extraction
                if self._extraction_cache:
                    self._extraction_cache.set(file_hash, chunk_hash, ontology_hash, extraction)

        ordered_extractions = [extractions.get(i, {"entities": [], "relations": []}) for i in range(len(chunks))]
        write_kwargs = {"profile": profile}
        if chunk_metadata is not None:
            write_kwargs["chunk_metadata"] = chunk_metadata
        episode_ids = self._write_extractions_batch(
            graph_id,
            chunks,
            ordered_extractions,
            **write_kwargs,
        )

        if progress_callback:
            progress_callback(1.0)
        logger.info("Processed batch of %s chunks", total)

        profile["total_seconds"] = time.perf_counter() - batch_started
        for key, value in list(profile.items()):
            if isinstance(value, float):
                profile[key] = round(value, 4)
        if profile_callback:
            profile_callback(profile)

        return episode_ids

    def _write_extractions_batch(
        self,
        graph_id: str,
        chunks: List[str],
        extractions: List[Dict[str, Any]],
        profile: Optional[Dict[str, Any]] = None,
        chunk_metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """Embed and write a full graph-build batch with bulk Cypher queries."""
        now = datetime.now(timezone.utc).isoformat()
        episode_rows = []
        entity_rows_by_key: Dict[str, Dict[str, Any]] = {}
        relation_specs = []
        embed_texts: List[str] = []

        metadata_rows = chunk_metadata or [{} for _ in chunks]
        for chunk_index, (chunk, extraction) in enumerate(zip(chunks, extractions)):
            metadata = metadata_rows[chunk_index] if chunk_index < len(metadata_rows) else {}
            episode_id = str(uuid.uuid4())
            episode_rows.append({
                "uuid": episode_id,
                "graph_id": graph_id,
                "data": chunk,
                "created_at": now,
                "case_id": metadata.get("case_id"),
                "case_version_id": metadata.get("version_id") or metadata.get("case_version_id"),
                "document_id": metadata.get("document_id"),
                "filename": metadata.get("filename"),
                "document_type": metadata.get("document_type"),
                "chunk_index": metadata.get("chunk_index"),
                "page": metadata.get("page"),
                "paragraph": metadata.get("paragraph"),
            })

            local_entity_keys: Dict[str, str] = {}
            for entity in extraction.get("entities", []):
                name = str(entity.get("name", "")).strip()
                etype = str(entity.get("type", "Entity") or "Entity").strip() or "Entity"
                if not name:
                    continue
                name_lower = name.lower()
                if name_lower not in entity_rows_by_key:
                    summary = f"{name} ({etype})"
                    entity_rows_by_key[name_lower] = {
                        "uuid": str(uuid.uuid4()),
                        "graph_id": graph_id,
                        "name_lower": name_lower,
                        "name": name,
                        "type": etype,
                        "summary": summary,
                        "attributes_json": json.dumps(entity.get("attributes", {}) or {}, ensure_ascii=False),
                        "embedding_index": len(embed_texts),
                        "embedding": [],
                        "created_at": now,
                    }
                    embed_texts.append(summary)
                local_entity_keys[name_lower] = name_lower

            for relation in extraction.get("relations", []):
                source_key = str(relation.get("source", "")).strip().lower()
                target_key = str(relation.get("target", "")).strip().lower()
                if not source_key or not target_key:
                    continue
                fact = str(
                    relation.get("fact")
                    or f"{relation.get('source', '')} {relation.get('type', '')} {relation.get('target', '')}"
                ).strip()
                relation_specs.append({
                    "uuid": str(uuid.uuid4()),
                    "graph_id": graph_id,
                    "source_key": source_key,
                    "target_key": target_key,
                    "name": str(relation.get("type", "RELATED_TO") or "RELATED_TO"),
                    "fact": fact,
                    "embedding_index": len(embed_texts),
                    "embedding": [],
                    "episode_id": episode_id,
                    "created_at": now,
                    "case_id": metadata.get("case_id"),
                    "case_version_id": metadata.get("version_id") or metadata.get("case_version_id"),
                    "document_id": metadata.get("document_id"),
                    "filename": metadata.get("filename"),
                    "document_type": metadata.get("document_type"),
                    "chunk_index": metadata.get("chunk_index"),
                    "page": metadata.get("page"),
                    "paragraph": metadata.get("paragraph"),
                })
                embed_texts.append(fact)

        embedding_start = time.perf_counter()
        embeddings: List[List[float]] = []
        if embed_texts:
            logger.info("[add_text_batch] Batch-embedding %s texts across %s chunks...", len(embed_texts), len(chunks))
            try:
                embeddings = self._embedding.embed_batch(embed_texts)
            except Exception as e:
                raise EmbeddingError(
                    "Batch embedding failed. Graph build stopped to avoid storing empty vectors. "
                    f"Provider={self._embedding.provider_id()} Error={e}"
                ) from e
        if profile is not None:
            profile["embedding_seconds"] = profile.get("embedding_seconds", 0.0) + (
                time.perf_counter() - embedding_start
            )

        for row in entity_rows_by_key.values():
            idx = row.pop("embedding_index")
            row["embedding"] = embeddings[idx] if idx < len(embeddings) else []
        for row in relation_specs:
            idx = row.pop("embedding_index")
            row["fact_embedding"] = embeddings[idx] if idx < len(embeddings) else []

        write_start = time.perf_counter()
        entity_rows = list(entity_rows_by_key.values())
        with self._driver.session() as session:
            if episode_rows:
                def _create_episodes(tx):
                    tx.run(
                        """
                        UNWIND $rows AS row
                        CREATE (ep:Episode {
                            uuid: row.uuid,
                            graph_id: row.graph_id,
                            data: row.data,
                            processed: true,
                            created_at: row.created_at,
                            case_id: row.case_id,
                            case_version_id: row.case_version_id,
                            document_id: row.document_id,
                            filename: row.filename,
                            document_type: row.document_type,
                            chunk_index: row.chunk_index,
                            page: row.page,
                            paragraph: row.paragraph
                        })
                        """,
                        rows=episode_rows,
                    )
                self._call_with_retry(session.execute_write, _create_episodes)

            entity_uuid_by_key: Dict[str, str] = {}
            if entity_rows:
                def _merge_entities(tx):
                    result = tx.run(
                        """
                        UNWIND $rows AS row
                        MERGE (n:Entity {graph_id: row.graph_id, name_lower: row.name_lower})
                        ON CREATE SET
                            n.uuid = row.uuid,
                            n.name = row.name,
                            n.summary = row.summary,
                            n.attributes_json = row.attributes_json,
                            n.embedding = row.embedding,
                            n.created_at = row.created_at
                        ON MATCH SET
                            n.summary = CASE WHEN n.summary = '' OR n.summary IS NULL
                                THEN row.summary ELSE n.summary END,
                            n.attributes_json = row.attributes_json,
                            n.embedding = row.embedding
                        RETURN row.name_lower AS name_lower, n.uuid AS uuid
                        """,
                        rows=entity_rows,
                    )
                    return {record["name_lower"]: record["uuid"] for record in result}
                entity_uuid_by_key = self._call_with_retry(session.execute_write, _merge_entities)

                label_groups: Dict[str, List[str]] = {}
                for row in entity_rows:
                    etype = row.get("type")
                    if etype and etype != "Entity":
                        label_groups.setdefault(etype, []).append(row["name_lower"])
                for etype, name_lowers in label_groups.items():
                    safe_label = str(etype).replace("`", "")
                    if not safe_label:
                        continue

                    def _add_labels(tx, _label=safe_label, _name_lowers=name_lowers):
                        tx.run(
                            f"""
                            UNWIND $name_lowers AS name_lower
                            MATCH (n:Entity {{graph_id: $gid, name_lower: name_lower}})
                            SET n:`{_label}`
                            """,
                            gid=graph_id,
                            name_lowers=_name_lowers,
                        )
                    self._call_with_retry(session.execute_write, _add_labels)

            relation_rows = []
            for relation in relation_specs:
                source_uuid = entity_uuid_by_key.get(relation["source_key"])
                target_uuid = entity_uuid_by_key.get(relation["target_key"])
                if not source_uuid or not target_uuid:
                    logger.warning(
                        "Skipping relation %s->%s: entity not found in extraction results",
                        relation["source_key"],
                        relation["target_key"],
                    )
                    continue
                relation_rows.append({
                    **relation,
                    "source_uuid": source_uuid,
                    "target_uuid": target_uuid,
                })

            if relation_rows:
                def _create_relations(tx):
                    tx.run(
                        """
                        UNWIND $rows AS row
                        MATCH (src:Entity {uuid: row.source_uuid})
                        MATCH (tgt:Entity {uuid: row.target_uuid})
                        CREATE (src)-[r:RELATION {
                            uuid: row.uuid,
                            graph_id: row.graph_id,
                            name: row.name,
                            fact: row.fact,
                            fact_embedding: row.fact_embedding,
                            attributes_json: '{}',
                            episode_ids: [row.episode_id],
                            case_id: row.case_id,
                            case_version_id: row.case_version_id,
                            document_id: row.document_id,
                            filename: row.filename,
                            document_type: row.document_type,
                            chunk_index: row.chunk_index,
                            page: row.page,
                            paragraph: row.paragraph,
                            created_at: row.created_at,
                            valid_at: null,
                            invalid_at: null,
                            expired_at: null
                        }]->(tgt)
                        """,
                        rows=relation_rows,
                    )
                self._call_with_retry(session.execute_write, _create_relations)

        if profile is not None:
            profile["neo4j_write_seconds"] = profile.get("neo4j_write_seconds", 0.0) + (
                time.perf_counter() - write_start
            )

        return [row["uuid"] for row in episode_rows]

    def add_agent_memories(self, graph_id: str, memories: List[Dict[str, Any]]) -> List[str]:
        """Write forecast/simulation memories as searchable Neo4j Entity nodes."""
        clean_memories = [
            memory for memory in memories
            if memory.get("text") and str(memory.get("text")).strip()
        ]
        if not clean_memories:
            return []

        texts = [str(memory["text"])[:2000] for memory in clean_memories]
        embeddings = self._embedding.embed_batch(texts)
        now = datetime.now(timezone.utc).isoformat()
        rows = []
        for memory, text, embedding in zip(clean_memories, texts, embeddings):
            memory_uuid = str(uuid.uuid4())
            attrs = {
                "agent_id": memory.get("agent_id"),
                "agent_name": memory.get("agent_name"),
                "platform": memory.get("platform"),
                "action_type": memory.get("action_type"),
                "round_num": memory.get("round_num"),
                "importance": memory.get("importance", 0.25),
                "recency": memory.get("recency") or memory.get("timestamp") or now,
                "source": memory.get("source", "simulation"),
                "visibility": memory.get("visibility", "public"),
            }
            rows.append({
                "uuid": memory_uuid,
                "name": f"Memory: {attrs.get('agent_name') or attrs.get('agent_id')} {attrs.get('action_type')}",
                "summary": text,
                "embedding": embedding,
                "attrs_json": json.dumps(attrs, ensure_ascii=False),
                "importance": attrs["importance"],
                "recency": attrs["recency"],
                "source": attrs["source"],
                "visibility": attrs["visibility"],
                "created_at": now,
            })

        def _write(tx):
            tx.run(
                """
                UNWIND $rows AS row
                CREATE (m:Entity:AgentMemory {
                    uuid: row.uuid,
                    graph_id: $gid,
                    name: row.name,
                    summary: row.summary,
                    embedding: row.embedding,
                    attributes_json: row.attrs_json,
                    importance: row.importance,
                    recency: row.recency,
                    source: row.source,
                    visibility: row.visibility,
                    created_at: row.created_at
                })
                """,
                gid=graph_id,
                rows=rows,
            )

        with self._driver.session() as session:
            self._call_with_retry(session.execute_write, _write)

        logger.info("[agent_memory] Wrote %s memory records to graph %s", len(rows), graph_id)
        return [row["uuid"] for row in rows]

    def wait_for_processing(
        self,
        episode_ids: List[str],
        progress_callback: Optional[Callable] = None,
        timeout: int = 600,
    ) -> None:
        """No-op — processing is synchronous in Neo4j."""
        if progress_callback:
            progress_callback(1.0)

    # ----------------------------------------------------------------
    # Read nodes
    # ----------------------------------------------------------------

    def get_all_nodes(self, graph_id: str, limit: int = 2000) -> List[Dict[str, Any]]:
        def _read(tx):
            result = tx.run(
                """
                MATCH (n:Entity {graph_id: $gid})
                RETURN n, labels(n) AS labels
                ORDER BY n.created_at DESC
                LIMIT $limit
                """,
                gid=graph_id,
                limit=limit,
            )
            return [self._node_to_dict(record["n"], record["labels"]) for record in result]

        with self._driver.session() as session:
            return self._call_with_retry(session.execute_read, _read)

    def get_node(self, uuid: str) -> Optional[Dict[str, Any]]:
        def _read(tx):
            result = tx.run(
                "MATCH (n:Entity {uuid: $uuid}) RETURN n, labels(n) AS labels",
                uuid=uuid,
            )
            record = result.single()
            if record:
                return self._node_to_dict(record["n"], record["labels"])
            return None

        with self._driver.session() as session:
            return self._call_with_retry(session.execute_read, _read)

    def get_node_edges(self, node_uuid: str) -> List[Dict[str, Any]]:
        """O(1) Cypher — NOT full scan + filter like the old Zep code."""
        def _read(tx):
            result = tx.run(
                """
                MATCH (n:Entity {uuid: $uuid})-[r:RELATION]-(m:Entity)
                RETURN r, startNode(r).uuid AS src_uuid, endNode(r).uuid AS tgt_uuid
                """,
                uuid=node_uuid,
            )
            return [
                self._edge_to_dict(record["r"], record["src_uuid"], record["tgt_uuid"])
                for record in result
            ]

        with self._driver.session() as session:
            return self._call_with_retry(session.execute_read, _read)

    def get_nodes_by_label(self, graph_id: str, label: str) -> List[Dict[str, Any]]:
        def _read(tx):
            # Dynamic label in query (safe — label comes from ontology, not user input)
            query = f"""
                MATCH (n:Entity:`{label}` {{graph_id: $gid}})
                RETURN n, labels(n) AS labels
            """
            result = tx.run(query, gid=graph_id)
            return [self._node_to_dict(record["n"], record["labels"]) for record in result]

        with self._driver.session() as session:
            return self._call_with_retry(session.execute_read, _read)

    # ----------------------------------------------------------------
    # Read edges
    # ----------------------------------------------------------------

    def get_all_edges(self, graph_id: str) -> List[Dict[str, Any]]:
        def _read(tx):
            result = tx.run(
                """
                MATCH (src:Entity)-[r:RELATION {graph_id: $gid}]->(tgt:Entity)
                RETURN r, src.uuid AS src_uuid, tgt.uuid AS tgt_uuid
                ORDER BY r.created_at DESC
                """,
                gid=graph_id,
            )
            return [
                self._edge_to_dict(record["r"], record["src_uuid"], record["tgt_uuid"])
                for record in result
            ]

        with self._driver.session() as session:
            return self._call_with_retry(session.execute_read, _read)

    # ----------------------------------------------------------------
    # Search
    # ----------------------------------------------------------------

    def search(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
        scope: str = "edges",
    ):
        """
        Hybrid search — returns results matching the scope.

        Returns a dict with 'edges' and/or 'nodes' lists
        (callers like zep_tools will wrap into SearchResult).
        """
        self._assert_embedding_compatible(graph_id)
        result = {"edges": [], "nodes": [], "query": query}

        with self._driver.session() as session:
            if scope in ("edges", "both"):
                result["edges"] = self._search.search_edges(
                    session, graph_id, query, limit
                )

            if scope in ("nodes", "both"):
                result["nodes"] = self._search.search_nodes(
                    session, graph_id, query, limit
                )

        return result

    # ----------------------------------------------------------------
    # Graph info
    # ----------------------------------------------------------------

    def get_graph_info(self, graph_id: str) -> Dict[str, Any]:
        def _read(tx):
            # Count nodes
            node_result = tx.run(
                "MATCH (n:Entity {graph_id: $gid}) RETURN count(n) AS cnt",
                gid=graph_id,
            )
            node_count = node_result.single()["cnt"]

            # Count edges
            edge_result = tx.run(
                "MATCH ()-[r:RELATION {graph_id: $gid}]->() RETURN count(r) AS cnt",
                gid=graph_id,
            )
            edge_count = edge_result.single()["cnt"]

            # Distinct entity types
            label_result = tx.run(
                """
                MATCH (n:Entity {graph_id: $gid})
                UNWIND labels(n) AS lbl
                WITH lbl WHERE lbl <> 'Entity'
                RETURN DISTINCT lbl
                """,
                gid=graph_id,
            )
            entity_types = [record["lbl"] for record in label_result]

            return {
                "graph_id": graph_id,
                "node_count": node_count,
                "edge_count": edge_count,
                "entity_types": entity_types,
            }

        with self._driver.session() as session:
            return self._call_with_retry(session.execute_read, _read)

    def get_graph_data(self, graph_id: str) -> Dict[str, Any]:
        """
        Full graph dump with enriched edge format (for frontend).
        Includes derived fields: fact_type, source_node_name, target_node_name.
        """
        def _read(tx):
            # Get all nodes
            node_result = tx.run(
                """
                MATCH (n:Entity {graph_id: $gid})
                RETURN n, labels(n) AS labels
                """,
                gid=graph_id,
            )
            nodes = []
            node_map: Dict[str, str] = {}  # uuid -> name
            for record in node_result:
                nd = self._node_to_dict(record["n"], record["labels"])
                nodes.append(nd)
                node_map[nd["uuid"]] = nd["name"]

            # Get all edges with source/target node names (JOIN)
            edge_result = tx.run(
                """
                MATCH (src:Entity)-[r:RELATION {graph_id: $gid}]->(tgt:Entity)
                RETURN r, src.uuid AS src_uuid, tgt.uuid AS tgt_uuid,
                       src.name AS src_name, tgt.name AS tgt_name
                """,
                gid=graph_id,
            )
            edges = []
            for record in edge_result:
                ed = self._edge_to_dict(record["r"], record["src_uuid"], record["tgt_uuid"])
                # Enriched fields for frontend
                ed["fact_type"] = ed["name"]
                ed["source_node_name"] = record["src_name"] or ""
                ed["target_node_name"] = record["tgt_name"] or ""
                # Legacy alias
                ed["episodes"] = ed.get("episode_ids", [])
                edges.append(ed)

            return {
                "graph_id": graph_id,
                "nodes": nodes,
                "edges": edges,
                "node_count": len(nodes),
                "edge_count": len(edges),
            }

        with self._driver.session() as session:
            return self._call_with_retry(session.execute_read, _read)

    # ----------------------------------------------------------------
    # Dict conversion helpers
    # ----------------------------------------------------------------

    @staticmethod
    def _node_to_dict(node, labels: List[str]) -> Dict[str, Any]:
        """Convert Neo4j node to the standard node dict format."""
        props = dict(node)
        attrs_json = props.pop("attributes_json", "{}")
        try:
            attributes = json.loads(attrs_json) if attrs_json else {}
        except (json.JSONDecodeError, TypeError):
            attributes = {}

        # Remove internal fields from dict
        props.pop("embedding", None)
        props.pop("name_lower", None)

        return {
            "uuid": props.get("uuid", ""),
            "name": props.get("name", ""),
            "labels": [l for l in labels if l != "Entity"] if labels else [],
            "summary": props.get("summary", ""),
            "attributes": attributes,
            "created_at": props.get("created_at"),
        }

    @staticmethod
    def _edge_to_dict(rel, source_uuid: str, target_uuid: str) -> Dict[str, Any]:
        """Convert Neo4j relationship to the standard edge dict format."""
        props = dict(rel)
        attrs_json = props.pop("attributes_json", "{}")
        try:
            attributes = json.loads(attrs_json) if attrs_json else {}
        except (json.JSONDecodeError, TypeError):
            attributes = {}

        # Remove internal fields
        props.pop("fact_embedding", None)

        episode_ids = props.get("episode_ids", [])
        if episode_ids and not isinstance(episode_ids, list):
            episode_ids = [str(episode_ids)]

        return {
            "uuid": props.get("uuid", ""),
            "name": props.get("name", ""),
            "fact": props.get("fact", ""),
            "source_node_uuid": source_uuid,
            "target_node_uuid": target_uuid,
            "attributes": attributes,
            "provenance": {
                "case_id": props.get("case_id"),
                "version_id": props.get("case_version_id"),
                "document_id": props.get("document_id"),
                "filename": props.get("filename"),
                "document_type": props.get("document_type"),
                "chunk_index": props.get("chunk_index"),
                "page": props.get("page"),
                "paragraph": props.get("paragraph"),
            },
            "created_at": props.get("created_at"),
            "valid_at": props.get("valid_at"),
            "invalid_at": props.get("invalid_at"),
            "expired_at": props.get("expired_at"),
            "episode_ids": episode_ids,
        }
