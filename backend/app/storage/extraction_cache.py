"""Persistent cache for expensive graph NER extraction results."""

import hashlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from ..config import Config


logger = logging.getLogger('mirofish.extraction_cache')


class ExtractionCache:
    """Disk-backed cache keyed by file, chunk, and ontology hashes."""

    VERSION = "ner-extraction-v1"

    def __init__(self, cache_dir: Optional[str | os.PathLike[str]] = None):
        self.cache_dir = Path(cache_dir or Config.GRAPH_EXTRACTION_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @staticmethod
    def hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @classmethod
    def hash_ontology(cls, ontology: Dict[str, Any]) -> str:
        payload = json.dumps(ontology or {}, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return cls.hash_text(payload)

    def get(self, file_hash: str, chunk_hash: str, ontology_hash: str) -> Optional[Dict[str, Any]]:
        path = self._path(file_hash, chunk_hash, ontology_hash)
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
            or metadata.get("file_hash") != file_hash
            or metadata.get("chunk_hash") != chunk_hash
            or metadata.get("ontology_hash") != ontology_hash
        ):
            return None
        extraction = payload.get("extraction")
        return extraction if isinstance(extraction, dict) else None

    def set(
        self,
        file_hash: str,
        chunk_hash: str,
        ontology_hash: str,
        extraction: Dict[str, Any],
    ) -> None:
        path = self._path(file_hash, chunk_hash, ontology_hash)
        payload = {
            "metadata": {
                "version": self.VERSION,
                "file_hash": file_hash,
                "chunk_hash": chunk_hash,
                "ontology_hash": ontology_hash,
            },
            "extraction": extraction,
        }
        tmp_path = path.with_suffix(".tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                with tmp_path.open("w", encoding="utf-8") as handle:
                    json.dump(payload, handle, ensure_ascii=False)
                os.replace(tmp_path, path)
        except OSError as exc:
            logger.warning("Skipping NER extraction cache write: path=%s error=%s", path, exc)
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _path(self, file_hash: str, chunk_hash: str, ontology_hash: str) -> Path:
        cache_key = self.hash_text("|".join([self.VERSION, file_hash, chunk_hash, ontology_hash]))
        return self.cache_dir / cache_key[:2] / f"{cache_key}.json"
