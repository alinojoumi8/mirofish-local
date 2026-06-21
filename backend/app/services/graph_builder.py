"""
Graph building service.
Uses GraphStorage (Neo4j) to replace Zep Cloud API.
"""

import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass

from ..config import Config
from ..models.task import TaskManager, TaskStatus
from ..models.case import normalize_graph_build_settings
from ..storage import GraphStorage
from ..storage.extraction_cache import ExtractionCache
from .text_processor import TextProcessor

logger = logging.getLogger('mirofish.graph_builder')


@dataclass
class GraphInfo:
    """Graph information"""
    graph_id: str
    node_count: int
    edge_count: int
    entity_types: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "entity_types": self.entity_types,
        }


class GraphBuilderService:
    """
    Graph building service
    Build knowledge graph through GraphStorage interface
    """

    def __init__(self, storage: GraphStorage):
        self.storage = storage
        self.task_manager = TaskManager()
        self.last_build_profile: Dict[str, Any] = {}

    @staticmethod
    def resolve_build_settings(raw: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Normalize graph build presets and explicit overrides."""
        return normalize_graph_build_settings(raw)

    @staticmethod
    def split_documents_with_provenance(
        documents: List[Dict[str, Any]],
        chunk_size: int = Config.DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = Config.DEFAULT_CHUNK_OVERLAP,
    ) -> List[Dict[str, Any]]:
        """
        Split documents without flattening boundaries.

        Returns rows shaped as {"text": chunk_text, "metadata": provenance}.
        """
        rows: List[Dict[str, Any]] = []
        for document in documents or []:
            text = document.get("text") or document.get("content") or ""
            if not text.strip():
                continue
            chunks = TextProcessor.split_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
            for idx, chunk in enumerate(chunks):
                metadata = {
                    "case_id": document.get("case_id"),
                    "version_id": document.get("version_id") or document.get("case_version_id"),
                    "document_id": document.get("document_id"),
                    "filename": document.get("filename"),
                    "document_type": document.get("document_type") or "other",
                    "chunk_index": idx,
                    "page": document.get("page"),
                    "paragraph": document.get("paragraph"),
                }
                rows.append({
                    "text": chunk,
                    "metadata": {k: v for k, v in metadata.items() if v is not None},
                })
        return rows

    def build_graph_async(
        self,
        text: str,
        ontology: Dict[str, Any],
        graph_name: str = "MiroFish Graph",
        chunk_size: int = Config.DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = Config.DEFAULT_CHUNK_OVERLAP,
        batch_size: int = Config.GRAPH_BUILD_BATCH_SIZE,
    ) -> str:
        """
        Build graph asynchronously

        Args:
            text: Input text to process
            ontology: Ontology definition (from ontology generator output)
            graph_name: Name for the graph
            chunk_size: Text chunk size
            chunk_overlap: Chunk overlap size
            batch_size: Number of chunks to send per batch

        Returns:
            Task ID
        """
        # Create task
        task_id = self.task_manager.create_task(
            task_type="graph_build",
            metadata={
                "graph_name": graph_name,
                "chunk_size": chunk_size,
                "text_length": len(text),
            }
        )

        # Execute build in background thread
        thread = threading.Thread(
            target=self._build_graph_worker,
            args=(task_id, text, ontology, graph_name, chunk_size, chunk_overlap, batch_size)
        )
        thread.daemon = True
        thread.start()

        return task_id

    def _build_graph_worker(
        self,
        task_id: str,
        text: str,
        ontology: Dict[str, Any],
        graph_name: str,
        chunk_size: int,
        chunk_overlap: int,
        batch_size: int
    ):
        """Graph build worker thread"""
        try:
            self.task_manager.update_task(
                task_id,
                status=TaskStatus.PROCESSING,
                progress=5,
                message="Starting graph building..."
            )

            # 1. Create graph
            graph_id = self.create_graph(graph_name)
            self.task_manager.update_task(
                task_id,
                progress=10,
                message=f"Graph created: {graph_id}"
            )

            # 2. Set ontology
            self.set_ontology(graph_id, ontology)
            self.task_manager.update_task(
                task_id,
                progress=15,
                message="Ontology set"
            )

            # 3. Text chunking
            chunks = TextProcessor.split_text(text, chunk_size, chunk_overlap)
            total_chunks = len(chunks)
            self.task_manager.update_task(
                task_id,
                progress=20,
                message=f"Text split into {total_chunks} chunks"
            )

            cache_context = {
                "file_hash": ExtractionCache.hash_text(text),
                "ontology_hash": ExtractionCache.hash_ontology(ontology),
            }

            # 4. Send data in batches (NER + embedding + Neo4j insert — synchronous)
            episode_uuids = self.add_text_batches(
                graph_id, chunks, batch_size,
                lambda msg, prog, detail=None: self.task_manager.update_task(
                    task_id,
                    progress=20 + int(prog * 60),  # 20-80%
                    message=msg,
                    progress_detail=detail or {},
                ),
                cache_context=cache_context,
            )

            # 5. Wait for processing (no-op for Neo4j — already synchronous)
            self.storage.wait_for_processing(episode_uuids)

            self.task_manager.update_task(
                task_id,
                progress=85,
                message="Data processing completed, getting graph information..."
            )

            # 6. Get graph information
            graph_info = self._get_graph_info(graph_id)

            # Completed
            self.task_manager.complete_task(task_id, {
                "graph_id": graph_id,
                "graph_info": graph_info.to_dict(),
                "chunks_processed": total_chunks,
                "profile": self.last_build_profile,
            })

        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.task_manager.fail_task(task_id, error_msg)

    def create_graph(self, name: str) -> str:
        """Create graph"""
        return self.storage.create_graph(
            name=name,
            description="MiroFish Social Simulation Graph"
        )

    def set_ontology(self, graph_id: str, ontology: Dict[str, Any]):
        """
        SetGraphOntology

        Simply stores ontology as JSON in the Graph node.
        No more dynamic Pydantic class creation (was Zep-specific).
        The NER extractor reads this ontology to guide extraction.
        """
        self.storage.set_ontology(graph_id, ontology)

    def add_text_batches(
        self,
        graph_id: str,
        chunks: List[str],
        batch_size: int = 3,
        progress_callback: Optional[Callable] = None,
        cache_context: Optional[Dict[str, Any]] = None,
        llm_concurrency_override: Optional[int] = None,
        chunk_metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """Add text in batches to graph, return uuid list of all episodes"""
        total_chunks = len(chunks)
        if total_chunks == 0:
            self.last_build_profile = self._new_build_profile(
                total_chunks=0,
                total_batches=0,
                batch_size=max(1, int(batch_size)),
                llm_concurrency=0,
            )
            return []
        batch_size = max(1, int(batch_size))
        batch_jobs = self._build_batch_jobs(chunks, batch_size)
        if chunk_metadata is not None:
            for job in batch_jobs:
                start = max(0, int(job["first_chunk"]) - 1)
                end = int(job["last_chunk"])
                job["chunk_metadata"] = chunk_metadata[start:end]
        total_batches = len(batch_jobs)
        llm_concurrency = self._resolve_llm_concurrency(total_batches, override=llm_concurrency_override)
        started_at = time.perf_counter()
        profile = self._new_build_profile(
            total_chunks=total_chunks,
            total_batches=total_batches,
            batch_size=batch_size,
            llm_concurrency=llm_concurrency,
        )

        logger.info(
            "[graph_build] Starting: %s chunks, %s batches (batch_size=%s, concurrency=%s)",
            total_chunks,
            total_batches,
            batch_size,
            llm_concurrency,
        )

        processed_chunks = 0
        results_by_batch: Dict[int, List[str]] = {}

        with ThreadPoolExecutor(max_workers=llm_concurrency) as executor:
            futures = {
                executor.submit(
                    self._add_batch_with_retry,
                    graph_id,
                    job["chunks"],
                    job["batch_num"],
                    total_batches,
                    cache_context,
                    job.get("chunk_metadata"),
                ): job
                for job in batch_jobs
            }

            for future in as_completed(futures):
                job = futures[future]
                batch_num = job["batch_num"]
                t0 = time.perf_counter()
                try:
                    batch_episode_ids, batch_profile, batch_elapsed = future.result()
                except Exception as e:
                    elapsed = time.perf_counter() - t0
                    profile["failed_batches"] += 1
                    logger.error(
                        "[graph_build] Batch %s/%s FAILED after %.1fs: %s",
                        batch_num,
                        total_batches,
                        elapsed,
                        e,
                    )
                    self._emit_progress(
                        progress_callback,
                        f"Batch {batch_num}/{total_batches} failed: {str(e)}",
                        processed_chunks / total_chunks,
                        {
                            "current_chunk": processed_chunks,
                            "total_chunks": total_chunks,
                            "batch_number": batch_num,
                            "total_batches": total_batches,
                            "llm_concurrency": llm_concurrency,
                            "eta_seconds": None,
                        },
                    )
                    self.last_build_profile = profile
                    raise

                results_by_batch[job["batch_index"]] = batch_episode_ids
                self._merge_profile(profile, batch_profile)
                profile["batch_wall_seconds"] += batch_elapsed
                processed_chunks += len(job["chunks"])
                total_elapsed = time.perf_counter() - started_at
                avg_seconds = total_elapsed / processed_chunks if processed_chunks else 0.0
                eta_seconds = max(total_chunks - processed_chunks, 0) * avg_seconds
                progress = processed_chunks / total_chunks
                detail = {
                    "current_chunk": processed_chunks,
                    "total_chunks": total_chunks,
                    "batch_number": batch_num,
                    "total_batches": total_batches,
                    "batch_size": len(job["chunks"]),
                    "llm_concurrency": llm_concurrency,
                    "batch_seconds": round(batch_elapsed, 2),
                    "elapsed_seconds": round(total_elapsed, 2),
                    "avg_seconds_per_chunk": round(avg_seconds, 2),
                    "eta_seconds": round(eta_seconds, 2),
                    "profile": self._rounded_profile(profile, total_elapsed),
                }
                message = (
                    f"Processed chunks {processed_chunks}/{total_chunks} "
                    f"(batch {batch_num}/{total_batches}); "
                    f"avg {avg_seconds:.1f}s/chunk; ETA {self._format_eta(eta_seconds)}"
                )
                logger.info("[graph_build] %s", message)
                self._emit_progress(progress_callback, message, progress, detail)

        total_elapsed = time.perf_counter() - started_at
        self.last_build_profile = self._rounded_profile(profile, total_elapsed)

        episode_uuids: List[str] = []
        for batch_index in range(total_batches):
            episode_uuids.extend(results_by_batch.get(batch_index, []))

        logger.info(f"[graph_build] All {total_chunks} chunks processed successfully")
        return episode_uuids

    def _add_batch_with_retry(
        self,
        graph_id: str,
        batch_chunks: List[str],
        batch_num: int,
        total_batches: int,
        cache_context: Optional[Dict[str, Any]],
        chunk_metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> tuple[List[str], Dict[str, Any], float]:
        """Run one storage batch with retry/backoff and return ids, profile, elapsed seconds."""
        first_chunk = batch_num
        last_error: Optional[Exception] = None
        attempts = max(0, int(Config.GRAPH_BUILD_BATCH_RETRIES)) + 1
        for attempt in range(attempts):
            batch_profile: Dict[str, Any] = {}
            t0 = time.perf_counter()
            try:
                logger.info(
                    "[graph_build] Batch %s/%s attempt %s/%s (%s chunks)",
                    batch_num,
                    total_batches,
                    attempt + 1,
                    attempts,
                    len(batch_chunks),
                )
                kwargs = {
                    "batch_size": len(batch_chunks),
                    "cache_context": cache_context,
                    "profile_callback": batch_profile.update,
                }
                if chunk_metadata is not None:
                    kwargs["chunk_metadata"] = chunk_metadata
                batch_episode_ids = self.storage.add_text_batch(graph_id, batch_chunks, **kwargs)
                return batch_episode_ids, batch_profile, time.perf_counter() - t0
            except Exception as e:
                last_error = e
                elapsed = time.perf_counter() - t0
                if attempt >= attempts - 1:
                    logger.error(
                        "[graph_build] Batch %s/%s failed permanently after %.1fs: %s",
                        batch_num,
                        total_batches,
                        elapsed,
                        e,
                    )
                    break
                wait_seconds = max(0.0, Config.GRAPH_BUILD_BATCH_RETRY_BASE_SECONDS * (2 ** attempt))
                logger.warning(
                    "[graph_build] Batch %s/%s failed after %.1fs; retrying in %.1fs: %s",
                    batch_num,
                    total_batches,
                    elapsed,
                    wait_seconds,
                    e,
                )
                time.sleep(wait_seconds)
        raise last_error or RuntimeError(f"Batch {first_chunk} failed")

    def _resolve_llm_concurrency(self, total_batches: int, override: Optional[int] = None) -> int:
        if total_batches <= 0:
            return 0
        if override is not None:
            return max(1, min(int(override), total_batches))
        configured = int(getattr(Config, "GRAPH_BUILD_LLM_CONCURRENCY", 0) or 0)
        if configured > 0:
            return max(1, min(configured, total_batches))
        provider = (getattr(Config, "LLM_DEFAULT_PROVIDER", "") or "").lower()
        base_url = (getattr(Config, "LLM_BASE_URL", "") or "").lower()
        is_local = (
            provider == "ollama"
            or "localhost" in base_url
            or "127.0.0.1" in base_url
            or "ollama" in base_url
            or ":11434" in base_url
        )
        fallback = (
            getattr(Config, "GRAPH_BUILD_LOCAL_LLM_CONCURRENCY", 1)
            if is_local
            else getattr(Config, "GRAPH_BUILD_CLOUD_LLM_CONCURRENCY", 3)
        )
        return max(1, min(int(fallback), total_batches))

    def _build_batch_jobs(self, chunks: List[str], batch_size: int) -> List[Dict[str, Any]]:
        """Pack chunks by max chunk count and prompt character budget."""
        batch_size = max(1, int(batch_size))
        max_chars = max(1, int(getattr(Config, "GRAPH_BUILD_MAX_BATCH_CHARS", 18000)))
        jobs: List[Dict[str, Any]] = []
        current_chunks: List[str] = []
        current_chars = 0
        first_chunk_index = 0

        def flush() -> None:
            nonlocal current_chunks, current_chars, first_chunk_index
            if not current_chunks:
                return
            batch_num = len(jobs) + 1
            last_chunk_index = first_chunk_index + len(current_chunks) - 1
            jobs.append({
                "batch_index": batch_num - 1,
                "batch_num": batch_num,
                "first_chunk": first_chunk_index + 1,
                "last_chunk": last_chunk_index + 1,
                "chunks": current_chunks,
            })
            current_chunks = []
            current_chars = 0
            first_chunk_index = last_chunk_index + 1

        for idx, chunk in enumerate(chunks):
            chunk_chars = len(chunk or "")
            if not current_chunks:
                first_chunk_index = idx
            would_exceed_count = len(current_chunks) >= batch_size
            would_exceed_chars = current_chunks and (current_chars + chunk_chars > max_chars)
            if would_exceed_count or would_exceed_chars:
                flush()
                first_chunk_index = idx
            current_chunks.append(chunk)
            current_chars += chunk_chars

        flush()
        return jobs

    def _new_build_profile(
        self,
        total_chunks: int,
        total_batches: int,
        batch_size: int,
        llm_concurrency: int,
    ) -> Dict[str, Any]:
        return {
            "total_chunks": total_chunks,
            "total_batches": total_batches,
            "batch_size": batch_size,
            "llm_concurrency": llm_concurrency,
            "llm_extraction_seconds": 0.0,
            "embedding_seconds": 0.0,
            "neo4j_write_seconds": 0.0,
            "batch_wall_seconds": 0.0,
            "total_wall_seconds": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "failed_batches": 0,
        }

    def _merge_profile(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        for key in (
            "llm_extraction_seconds",
            "embedding_seconds",
            "neo4j_write_seconds",
            "cache_hits",
            "cache_misses",
            "failed_batches",
        ):
            value = source.get(key, 0) if source else 0
            target[key] = target.get(key, 0) + value

    def _rounded_profile(self, profile: Dict[str, Any], total_elapsed: float) -> Dict[str, Any]:
        result = dict(profile)
        result["total_wall_seconds"] = total_elapsed
        for key, value in list(result.items()):
            if isinstance(value, float):
                result[key] = round(value, 4)
        return result

    def _emit_progress(
        self,
        progress_callback: Optional[Callable],
        message: str,
        progress: float,
        detail: Dict[str, Any],
    ) -> None:
        if not progress_callback:
            return
        try:
            progress_callback(message, progress, detail)
        except TypeError:
            progress_callback(message, progress)

    def _format_eta(self, seconds: float) -> str:
        if seconds <= 0:
            return "0s"
        minutes, sec = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours}h {minutes}m"
        if minutes:
            return f"{minutes}m {sec}s"
        return f"{sec}s"

    def _get_graph_info(self, graph_id: str) -> GraphInfo:
        """Get graph information"""
        info = self.storage.get_graph_info(graph_id)
        return GraphInfo(
            graph_id=info["graph_id"],
            node_count=info["node_count"],
            edge_count=info["edge_count"],
            entity_types=info.get("entity_types", []),
        )

    def get_graph_data(self, graph_id: str) -> Dict[str, Any]:
        """Get complete graph data (including details)"""
        return self.storage.get_graph_data(graph_id)

    def delete_graph(self, graph_id: str):
        """Delete graph"""
        self.storage.delete_graph(graph_id)
