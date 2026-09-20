"""Transactional SQLite control-plane storage.

Neo4j remains the knowledge-graph store and the filesystem remains the artifact
store. This database owns small, frequently updated metadata that must survive
process restarts without partial JSON writes.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional


logger = logging.getLogger("mirofish.control_db")

SCHEMA_VERSION = 1
_TABLE_IDS = {
    "projects": "project_id",
    "cases": "case_id",
    "tasks": "task_id",
    "simulation_runs": "simulation_id",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ControlDatabase:
    """Repository for durable application metadata using short SQLite transactions."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._migration_lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    @contextmanager
    def _transaction(self):
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._transaction() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    name TEXT PRIMARY KEY,
                    version INTEGER NOT NULL,
                    applied_at TEXT NOT NULL,
                    details_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            for table in _TABLE_IDS:
                connection.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {table} (
                        record_id TEXT PRIMARY KEY,
                        payload_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{table}_updated_at ON {table}(updated_at DESC)"
                )
            connection.execute(
                """
                INSERT INTO schema_migrations(name, version, applied_at, details_json)
                VALUES('schema', ?, ?, '{}')
                ON CONFLICT(name) DO UPDATE SET version=excluded.version
                """,
                (SCHEMA_VERSION, _utc_now()),
            )

    @staticmethod
    def _timestamps(payload: Dict[str, Any]) -> tuple[str, str]:
        now = _utc_now()
        created_at = str(payload.get("created_at") or now)
        updated_at = str(payload.get("updated_at") or created_at)
        return created_at, updated_at

    def _put(self, table: str, payload: Dict[str, Any], *, replace: bool = True) -> bool:
        id_field = _TABLE_IDS[table]
        record_id = str(payload.get(id_field) or "")
        if not record_id:
            raise ValueError(f"{id_field} is required for {table}")
        created_at, updated_at = self._timestamps(payload)
        conflict = (
            "DO UPDATE SET payload_json=excluded.payload_json, updated_at=excluded.updated_at"
            if replace
            else "DO NOTHING"
        )
        with self._transaction() as connection:
            cursor = connection.execute(
                f"""
                INSERT INTO {table}(record_id, payload_json, created_at, updated_at)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(record_id) {conflict}
                """,
                (record_id, json.dumps(payload, ensure_ascii=False), created_at, updated_at),
            )
            return cursor.rowcount > 0

    def _get(self, table: str, record_id: str) -> Optional[Dict[str, Any]]:
        with self._transaction() as connection:
            row = connection.execute(
                f"SELECT payload_json FROM {table} WHERE record_id = ?",
                (record_id,),
            ).fetchone()
        return json.loads(row["payload_json"]) if row else None

    def _list(self, table: str, limit: int = 50) -> list[Dict[str, Any]]:
        safe_limit = max(0, int(limit))
        with self._transaction() as connection:
            rows = connection.execute(
                f"SELECT payload_json FROM {table} ORDER BY updated_at DESC LIMIT ?",
                (safe_limit,),
            ).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    def _delete(self, table: str, record_id: str) -> bool:
        with self._transaction() as connection:
            cursor = connection.execute(
                f"DELETE FROM {table} WHERE record_id = ?",
                (record_id,),
            )
            return cursor.rowcount > 0

    def _mutate(
        self,
        table: str,
        record_id: str,
        mutator: Callable[[Dict[str, Any]], Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Serialize a read-modify-write cycle with a SQLite write lock."""
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                f"SELECT payload_json, created_at FROM {table} WHERE record_id = ?",
                (record_id,),
            ).fetchone()
            if row is None:
                connection.rollback()
                return None
            current = json.loads(row["payload_json"])
            updated = mutator(dict(current))
            if not isinstance(updated, dict):
                raise TypeError("metadata mutator must return a dictionary")
            id_field = _TABLE_IDS[table]
            if str(updated.get(id_field) or "") != record_id:
                raise ValueError(f"metadata mutator cannot change {id_field}")
            updated["updated_at"] = _utc_now()
            connection.execute(
                f"UPDATE {table} SET payload_json = ?, updated_at = ? WHERE record_id = ?",
                (json.dumps(updated, ensure_ascii=False), updated["updated_at"], record_id),
            )
            connection.commit()
            return updated
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def put_project(self, payload: Dict[str, Any]) -> bool:
        return self._put("projects", payload)

    def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        return self._get("projects", project_id)

    def list_projects(self, limit: int = 50) -> list[Dict[str, Any]]:
        return self._list("projects", limit)

    def delete_project(self, project_id: str) -> bool:
        return self._delete("projects", project_id)

    def put_case(self, payload: Dict[str, Any]) -> bool:
        return self._put("cases", payload)

    def get_case(self, case_id: str) -> Optional[Dict[str, Any]]:
        return self._get("cases", case_id)

    def list_cases(self, limit: int = 50) -> list[Dict[str, Any]]:
        return self._list("cases", limit)

    def delete_case(self, case_id: str) -> bool:
        return self._delete("cases", case_id)

    def mutate_case(
        self,
        case_id: str,
        mutator: Callable[[Dict[str, Any]], Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        return self._mutate("cases", case_id, mutator)

    def put_task(self, payload: Dict[str, Any]) -> bool:
        return self._put("tasks", payload)

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        return self._get("tasks", task_id)

    def list_tasks(self, limit: int = 1000) -> list[Dict[str, Any]]:
        return self._list("tasks", limit)

    def delete_task(self, task_id: str) -> bool:
        return self._delete("tasks", task_id)

    def put_simulation_run(self, payload: Dict[str, Any]) -> bool:
        return self._put("simulation_runs", payload)

    def get_simulation_run(self, simulation_id: str) -> Optional[Dict[str, Any]]:
        return self._get("simulation_runs", simulation_id)

    def delete_simulation_run(self, simulation_id: str) -> bool:
        return self._delete("simulation_runs", simulation_id)

    def list_simulation_runs(self, limit: int = 1000) -> list[Dict[str, Any]]:
        return self._list("simulation_runs", limit)

    def migrate_legacy(
        self,
        projects_dir: str | Path,
        cases_dir: str | Path,
        simulations_dir: str | Path,
    ) -> Dict[str, int]:
        """Import legacy JSON once without deleting or rewriting source files."""
        migration_name = "legacy_json_v1"
        counts = {"projects": 0, "cases": 0, "simulation_runs": 0}
        with self._migration_lock:
            with self._transaction() as connection:
                applied = connection.execute(
                    "SELECT 1 FROM schema_migrations WHERE name = ?",
                    (migration_name,),
                ).fetchone()
                if applied:
                    return counts

            candidates = (
                [("projects", path) for path in Path(projects_dir).glob("*/project.json")]
                + [("cases", path) for path in Path(cases_dir).glob("*/case.json")]
                + [("simulation_runs", path) for path in Path(simulations_dir).glob("*/run_state.json")]
            )
            loaded: list[tuple[str, Dict[str, Any]]] = []
            for table, path in candidates:
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise RuntimeError(f"Cannot import legacy metadata {path}: {exc}") from exc
                if not isinstance(payload, dict):
                    raise RuntimeError(f"Legacy metadata is not an object: {path}")
                loaded.append((table, payload))

            with self._transaction() as connection:
                for table, payload in loaded:
                    id_field = _TABLE_IDS[table]
                    record_id = str(payload.get(id_field) or "")
                    if not record_id:
                        raise RuntimeError(f"Legacy {table} record is missing {id_field}")
                    created_at, updated_at = self._timestamps(payload)
                    cursor = connection.execute(
                        f"""
                        INSERT INTO {table}(record_id, payload_json, created_at, updated_at)
                        VALUES(?, ?, ?, ?)
                        ON CONFLICT(record_id) DO NOTHING
                        """,
                        (record_id, json.dumps(payload, ensure_ascii=False), created_at, updated_at),
                    )
                    counts[table] += int(cursor.rowcount > 0)
                connection.execute(
                    """
                    INSERT INTO schema_migrations(name, version, applied_at, details_json)
                    VALUES(?, ?, ?, ?)
                    """,
                    (migration_name, SCHEMA_VERSION, _utc_now(), json.dumps(counts)),
                )
        return counts

    def migration_state(self) -> Dict[str, Any]:
        with self._transaction() as connection:
            rows = connection.execute(
                "SELECT name, version, applied_at, details_json FROM schema_migrations ORDER BY applied_at"
            ).fetchall()
        return {
            row["name"]: {
                "version": row["version"],
                "applied_at": row["applied_at"],
                "details": json.loads(row["details_json"]),
            }
            for row in rows
        }

    def health_status(self) -> Dict[str, Any]:
        try:
            with self._transaction() as connection:
                connection.execute("SELECT 1").fetchone()
                journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
                schema = connection.execute(
                    "SELECT version FROM schema_migrations WHERE name = 'schema'"
                ).fetchone()
            return {
                "healthy": True,
                "path": str(self.path),
                "schema_version": int(schema[0]) if schema else 0,
                "journal_mode": str(journal_mode).lower(),
            }
        except sqlite3.Error as exc:
            logger.error("Control database health check failed: %s", exc)
            return {
                "healthy": False,
                "path": str(self.path),
                "schema_version": 0,
                "error": str(exc),
            }
