"""Deterministic economic kernel for MiroFish simulations.

The social simulators remain responsible for agent conversation and behaviour.
This module owns the canonical economic identity map, validated state
transitions, and the double-entry ledger shared by every social platform.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional

from ..config import Config


SCHEMA_VERSION = 1
SUPPORTED_ACTIONS = {
    "post_job",
    "apply_job",
    "hire",
    "complete_job",
    "transfer",
    "buy_goods",
    "do_nothing",
}

DEFAULT_ECONOMY_SETTINGS: Dict[str, Any] = {
    "enabled": False,
    "initial_balance_cents": 1_000_000,
    "rounds_per_tick": None,
    "max_decisions_per_tick": 50,
    "currency": "USD",
}


class EconomyError(ValueError):
    """A user or agent requested an invalid economic transition."""


class LedgerInvariantError(RuntimeError):
    """The double-entry ledger invariant was violated."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _integer_cents(value: Any, field_name: str, *, allow_zero: bool) -> int:
    if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
        qualifier = "non-negative" if allow_zero else "positive"
        raise EconomyError(f"{field_name} must be a {qualifier} integer")
    try:
        cents = int(value)
    except (TypeError, ValueError) as exc:
        qualifier = "non-negative" if allow_zero else "positive"
        raise EconomyError(f"{field_name} must be a {qualifier} integer") from exc
    if cents < 0 or (cents == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise EconomyError(f"{field_name} must be a {qualifier} integer")
    return cents


def normalize_economy_settings(
    raw: Optional[Dict[str, Any]],
    *,
    minutes_per_round: int = 60,
) -> Dict[str, Any]:
    """Validate and fill the version-one economy settings."""
    settings = dict(DEFAULT_ECONOMY_SETTINGS)
    if isinstance(raw, dict):
        settings.update(raw)

    enabled = settings.get("enabled", False)
    if isinstance(enabled, str):
        normalized_enabled = enabled.strip().lower()
        if normalized_enabled in {"true", "1", "yes", "on"}:
            enabled = True
        elif normalized_enabled in {"false", "0", "no", "off", ""}:
            enabled = False
        else:
            raise EconomyError("enabled must be a boolean")
    settings["enabled"] = bool(enabled)
    try:
        settings["initial_balance_cents"] = _integer_cents(
            settings["initial_balance_cents"], "initial_balance_cents", allow_zero=True
        )
        settings["max_decisions_per_tick"] = int(settings["max_decisions_per_tick"])
        minutes = max(1, int(minutes_per_round))
        configured_rounds = settings.get("rounds_per_tick")
        settings["rounds_per_tick"] = (
            max(1, int(configured_rounds))
            if configured_rounds not in (None, "")
            else max(1, math.ceil(1440 / minutes))
        )
    except (TypeError, ValueError) as exc:
        raise EconomyError(f"Invalid economy settings: {exc}") from exc

    if not 1 <= settings["max_decisions_per_tick"] <= 500:
        raise EconomyError("max_decisions_per_tick must be between 1 and 500")

    currency = str(settings.get("currency") or "USD").strip().upper()
    if not 3 <= len(currency) <= 8 or not currency.isalnum():
        raise EconomyError("currency must be a 3-8 character alphanumeric code")
    settings["currency"] = currency
    settings["schema_version"] = SCHEMA_VERSION
    return settings


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _stable_id(prefix: str, *parts: Any) -> str:
    digest = hashlib.sha256(_json(parts).encode("utf-8")).hexdigest()[:24]
    return f"{prefix}_{digest}"


def _positive_cents(value: Any, field_name: str = "amount_cents") -> int:
    return _integer_cents(value, field_name, allow_zero=False)


def _economic_agent_id(profile: Dict[str, Any]) -> str:
    source = (
        profile.get("entity_uuid")
        or profile.get("source_entity_uuid")
        or profile.get("uuid")
        or f"oasis:{profile.get('agent_id', profile.get('user_id'))}"
    )
    digest = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:20]
    return f"econ_{digest}"


class EconomyStore:
    """SQLite-backed economic state and settlement engine."""

    def __init__(self, db_path: str, *, readonly: bool = False):
        self.db_path = os.path.abspath(db_path)
        self.readonly = bool(readonly)
        self._lock = threading.RLock()
        if not self.readonly:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            self.initialize_schema()

    @classmethod
    def for_simulation(
        cls, simulation_id: str, *, readonly: bool = False
    ) -> "EconomyStore":
        safe_id = str(simulation_id).strip()
        if not safe_id or safe_id in {".", ".."} or any(
            token in safe_id for token in ("/", "\\", "\x00")
        ):
            raise EconomyError("Invalid simulation id")
        return cls(
            os.path.join(Config.OASIS_SIMULATION_DATA_DIR, safe_id, "economy.db"),
            readonly=readonly,
        )

    def _connect(self) -> sqlite3.Connection:
        target = self.db_path
        kwargs: Dict[str, Any] = {}
        if self.readonly:
            target = f"{Path(self.db_path).as_uri()}?mode=ro"
            kwargs["uri"] = True
        conn = sqlite3.connect(target, timeout=30, isolation_level=None, **kwargs)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        if not self.readonly:
            conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 30000")
        return conn

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Yield a transactional connection and always release its OS handle."""
        conn = self._connect()
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def initialize_schema(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS economy_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS economic_agents (
            economic_agent_id TEXT PRIMARY KEY,
            oasis_agent_id INTEGER NOT NULL UNIQUE,
            source_entity_uuid TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL,
            role TEXT NOT NULL,
            profile_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS accounts (
            account_id TEXT PRIMARY KEY,
            owner_agent_id TEXT UNIQUE,
            kind TEXT NOT NULL,
            currency TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(owner_agent_id) REFERENCES economic_agents(economic_agent_id)
        );
        CREATE TABLE IF NOT EXISTS transactions (
            transaction_id TEXT PRIMARY KEY,
            tick INTEGER NOT NULL,
            action_type TEXT NOT NULL,
            actor_id TEXT,
            counterparty_id TEXT,
            amount_cents INTEGER NOT NULL DEFAULT 0,
            idempotency_key TEXT NOT NULL UNIQUE,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS ledger_entries (
            entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT NOT NULL,
            account_id TEXT NOT NULL,
            amount_cents INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(transaction_id) REFERENCES transactions(transaction_id),
            FOREIGN KEY(account_id) REFERENCES accounts(account_id)
        );
        CREATE INDEX IF NOT EXISTS idx_ledger_account ON ledger_entries(account_id);
        CREATE INDEX IF NOT EXISTS idx_transactions_tick ON transactions(tick);
        CREATE TABLE IF NOT EXISTS economic_intents (
            intent_id TEXT PRIMARY KEY,
            tick INTEGER NOT NULL,
            actor_id TEXT NOT NULL,
            action_type TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            status TEXT NOT NULL,
            rejection_reason TEXT,
            idempotency_key TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(actor_id) REFERENCES economic_agents(economic_agent_id)
        );
        CREATE INDEX IF NOT EXISTS idx_intents_tick ON economic_intents(tick);
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            employer_id TEXT NOT NULL,
            worker_id TEXT,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            wage_cents INTEGER NOT NULL,
            status TEXT NOT NULL,
            created_tick INTEGER NOT NULL,
            updated_tick INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(employer_id) REFERENCES economic_agents(economic_agent_id),
            FOREIGN KEY(worker_id) REFERENCES economic_agents(economic_agent_id)
        );
        CREATE TABLE IF NOT EXISTS job_applications (
            job_id TEXT NOT NULL,
            applicant_id TEXT NOT NULL,
            status TEXT NOT NULL,
            applied_tick INTEGER NOT NULL,
            updated_tick INTEGER NOT NULL,
            PRIMARY KEY(job_id, applicant_id),
            FOREIGN KEY(job_id) REFERENCES jobs(job_id),
            FOREIGN KEY(applicant_id) REFERENCES economic_agents(economic_agent_id)
        );
        CREATE TABLE IF NOT EXISTS trades (
            trade_id TEXT PRIMARY KEY,
            transaction_id TEXT NOT NULL UNIQUE,
            tick INTEGER NOT NULL,
            buyer_id TEXT NOT NULL,
            seller_id TEXT NOT NULL,
            amount_cents INTEGER NOT NULL,
            description TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(transaction_id) REFERENCES transactions(transaction_id),
            FOREIGN KEY(buyer_id) REFERENCES economic_agents(economic_agent_id),
            FOREIGN KEY(seller_id) REFERENCES economic_agents(economic_agent_id)
        );
        CREATE INDEX IF NOT EXISTS idx_trades_tick ON trades(tick);
        CREATE TABLE IF NOT EXISTS economy_events (
            event_id TEXT PRIMARY KEY,
            tick INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            actor_id TEXT,
            counterparty_id TEXT,
            amount_cents INTEGER NOT NULL DEFAULT 0,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_economy_events_tick ON economy_events(tick);
        CREATE TABLE IF NOT EXISTS economy_ticks (
            tick INTEGER PRIMARY KEY,
            round_num INTEGER NOT NULL,
            status TEXT NOT NULL,
            error TEXT,
            started_at TEXT NOT NULL,
            completed_at TEXT
        );
        CREATE TABLE IF NOT EXISTS economy_metrics (
            tick INTEGER PRIMARY KEY,
            agents_count INTEGER NOT NULL,
            total_supply_cents INTEGER NOT NULL,
            employed_count INTEGER NOT NULL,
            open_jobs INTEGER NOT NULL,
            completed_jobs INTEGER NOT NULL,
            trade_volume_cents INTEGER NOT NULL,
            gini REAL NOT NULL,
            created_at TEXT NOT NULL
        );
        """
        with self._lock, self._connection() as conn:
            conn.executescript(schema)
            conn.execute(
                "INSERT OR REPLACE INTO economy_meta(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

    def configure(self, settings: Dict[str, Any]) -> None:
        with self._lock, self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT OR REPLACE INTO economy_meta(key, value) VALUES('settings', ?)",
                (_json(settings),),
            )
            conn.execute(
                "INSERT OR IGNORE INTO economy_meta(key, value) VALUES('status', 'ready')"
            )
            conn.execute("COMMIT")

    def register_agents(
        self,
        profiles: Iterable[Dict[str, Any]],
        *,
        initial_balance_cents: int,
        currency: str,
    ) -> List[Dict[str, Any]]:
        """Create the cross-platform identity map and balanced opening entries."""
        initial_balance_cents = _integer_cents(
            initial_balance_cents, "initial_balance_cents", allow_zero=True
        )
        registered: List[Dict[str, Any]] = []
        with self._lock, self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                now = utc_now()
                conn.execute(
                    """INSERT OR IGNORE INTO accounts(
                        account_id, owner_agent_id, kind, currency, created_at
                    ) VALUES('system:issuance', NULL, 'issuance', ?, ?)""",
                    (currency, now),
                )
                for index, raw in enumerate(profiles):
                    profile = dict(raw or {})
                    oasis_id = profile.get("agent_id", profile.get("user_id", index))
                    try:
                        oasis_id = int(oasis_id)
                    except (TypeError, ValueError):
                        oasis_id = index
                    profile["agent_id"] = oasis_id
                    economic_id = _economic_agent_id(profile)
                    source_uuid = str(
                        profile.get("entity_uuid")
                        or profile.get("source_entity_uuid")
                        or profile.get("uuid")
                        or f"oasis:{oasis_id}"
                    )
                    name = str(
                        profile.get("entity_name")
                        or profile.get("name")
                        or profile.get("user_name")
                        or f"Agent {oasis_id}"
                    )
                    role = str(
                        profile.get("entity_type")
                        or profile.get("role")
                        or profile.get("stance")
                        or "participant"
                    )
                    conn.execute(
                        """INSERT OR IGNORE INTO economic_agents(
                            economic_agent_id, oasis_agent_id, source_entity_uuid,
                            name, role, profile_json, created_at
                        ) VALUES(?, ?, ?, ?, ?, ?, ?)""",
                        (economic_id, oasis_id, source_uuid, name, role, _json(profile), now),
                    )
                    account_id = f"agent:{economic_id}"
                    conn.execute(
                        """INSERT OR IGNORE INTO accounts(
                            account_id, owner_agent_id, kind, currency, created_at
                        ) VALUES(?, ?, 'wallet', ?, ?)""",
                        (account_id, economic_id, currency, now),
                    )
                    opening_key = f"genesis:{economic_id}"
                    exists = conn.execute(
                        "SELECT transaction_id FROM transactions WHERE idempotency_key = ?",
                        (opening_key,),
                    ).fetchone()
                    if not exists and initial_balance_cents:
                        tx_id = _stable_id("tx", opening_key)
                        conn.execute(
                            """INSERT INTO transactions(
                                transaction_id, tick, action_type, actor_id,
                                counterparty_id, amount_cents, idempotency_key,
                                payload_json, created_at
                            ) VALUES(?, 0, 'genesis', NULL, ?, ?, ?, '{}', ?)""",
                            (tx_id, economic_id, initial_balance_cents, opening_key, now),
                        )
                        conn.executemany(
                            """INSERT INTO ledger_entries(
                                transaction_id, account_id, amount_cents, created_at
                            ) VALUES(?, ?, ?, ?)""",
                            [
                                (tx_id, "system:issuance", -initial_balance_cents, now),
                                (tx_id, account_id, initial_balance_cents, now),
                            ],
                        )
                    registered.append(
                        {
                            "economic_agent_id": economic_id,
                            "oasis_agent_id": oasis_id,
                            "source_entity_uuid": source_uuid,
                            "name": name,
                            "role": role,
                        }
                    )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        self.assert_invariants()
        return registered

    @staticmethod
    def _account_id(agent_id: str) -> str:
        return f"agent:{agent_id}"

    def _agent_exists(self, conn: sqlite3.Connection, agent_id: str) -> bool:
        return conn.execute(
            "SELECT 1 FROM economic_agents WHERE economic_agent_id = ?", (agent_id,)
        ).fetchone() is not None

    def _balance(self, conn: sqlite3.Connection, agent_id: str) -> int:
        row = conn.execute(
            "SELECT COALESCE(SUM(amount_cents), 0) AS balance FROM ledger_entries WHERE account_id = ?",
            (self._account_id(agent_id),),
        ).fetchone()
        return int(row["balance"] if row else 0)

    def resolve_agent_id(self, identifier: Any) -> Optional[str]:
        if identifier is None:
            return None
        with self._connection() as conn:
            return self._resolve_agent_id(conn, identifier)

    @staticmethod
    def _resolve_agent_id(conn: sqlite3.Connection, identifier: Any) -> Optional[str]:
        if identifier is None:
            return None
        if str(identifier).startswith("econ_"):
            row = conn.execute(
                "SELECT economic_agent_id FROM economic_agents WHERE economic_agent_id = ?",
                (str(identifier),),
            ).fetchone()
        else:
            try:
                oasis_id = int(identifier)
            except (TypeError, ValueError):
                oasis_id = -1
            row = conn.execute(
                """SELECT economic_agent_id FROM economic_agents
                   WHERE oasis_agent_id = ? OR source_entity_uuid = ?""",
                (oasis_id, str(identifier)),
            ).fetchone()
        return str(row["economic_agent_id"]) if row else None

    def _record_event(
        self,
        conn: sqlite3.Connection,
        *,
        tick: int,
        event_type: str,
        actor_id: Optional[str],
        counterparty_id: Optional[str] = None,
        amount_cents: int = 0,
        payload: Optional[Dict[str, Any]] = None,
    ) -> str:
        event_id = _stable_id(
            "evt", tick, event_type, actor_id, counterparty_id, amount_cents, payload or {}
        )
        conn.execute(
            """INSERT INTO economy_events(
                event_id, tick, event_type, actor_id, counterparty_id,
                amount_cents, payload_json, created_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event_id,
                tick,
                event_type,
                actor_id,
                counterparty_id,
                int(amount_cents),
                _json(payload or {}),
                utc_now(),
            ),
        )
        return event_id

    def _settle(
        self,
        conn: sqlite3.Connection,
        *,
        tick: int,
        action_type: str,
        actor_id: str,
        counterparty_id: str,
        amount_cents: int,
        idempotency_key: str,
        payload: Dict[str, Any],
    ) -> str:
        amount = _positive_cents(amount_cents)
        if actor_id == counterparty_id:
            raise EconomyError("actor and counterparty must be different")
        if not self._agent_exists(conn, counterparty_id):
            raise EconomyError("counterparty does not exist")
        if self._balance(conn, actor_id) < amount:
            raise EconomyError("insufficient funds")

        tx_id = _stable_id("tx", idempotency_key)
        now = utc_now()
        conn.execute(
            """INSERT INTO transactions(
                transaction_id, tick, action_type, actor_id, counterparty_id,
                amount_cents, idempotency_key, payload_json, created_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                tx_id,
                tick,
                action_type,
                actor_id,
                counterparty_id,
                amount,
                idempotency_key,
                _json(payload),
                now,
            ),
        )
        conn.executemany(
            """INSERT INTO ledger_entries(
                transaction_id, account_id, amount_cents, created_at
            ) VALUES(?, ?, ?, ?)""",
            [
                (tx_id, self._account_id(actor_id), -amount, now),
                (tx_id, self._account_id(counterparty_id), amount, now),
            ],
        )
        return tx_id

    def _apply_do_nothing(
        self, conn, tick, actor_id, payload, idempotency_key, now
    ) -> Dict[str, Any]:
        self._record_event(
            conn, tick=tick, event_type="do_nothing", actor_id=actor_id, payload=payload
        )
        return {}

    def _apply_post_job(
        self, conn, tick, actor_id, payload, idempotency_key, now
    ) -> Dict[str, Any]:
        title = str(payload.get("title") or "").strip()
        description = str(payload.get("description") or "").strip()
        wage = _positive_cents(payload.get("wage_cents", 0), "wage_cents")
        if not title:
            raise EconomyError("post_job requires title")
        job_id = str(payload.get("job_id") or _stable_id("job", idempotency_key))
        if conn.execute("SELECT 1 FROM jobs WHERE job_id=?", (job_id,)).fetchone():
            raise EconomyError("job_id already exists")
        conn.execute(
            """INSERT INTO jobs(
                job_id, employer_id, worker_id, title, description,
                wage_cents, status, created_tick, updated_tick,
                created_at, updated_at
            ) VALUES(?, ?, NULL, ?, ?, ?, 'open', ?, ?, ?, ?)""",
            (job_id, actor_id, title, description, wage, tick, tick, now, now),
        )
        self._record_event(
            conn,
            tick=tick,
            event_type="post_job",
            actor_id=actor_id,
            amount_cents=wage,
            payload={**payload, "job_id": job_id},
        )
        return {"job_id": job_id}

    def _apply_job_application(
        self, conn, tick, actor_id, payload, idempotency_key, now
    ) -> Dict[str, Any]:
        job_id = str(payload.get("job_id") or "")
        job = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if not job or job["status"] != "open":
            raise EconomyError("job is not open")
        if job["employer_id"] == actor_id:
            raise EconomyError("employer cannot apply to own job")
        if conn.execute(
            "SELECT 1 FROM job_applications WHERE job_id=? AND applicant_id=?",
            (job_id, actor_id),
        ).fetchone():
            raise EconomyError("agent has already applied to this job")
        conn.execute(
            """INSERT INTO job_applications(
                job_id, applicant_id, status, applied_tick, updated_tick
            ) VALUES(?, ?, 'pending', ?, ?)""",
            (job_id, actor_id, tick, tick),
        )
        self._record_event(
            conn,
            tick=tick,
            event_type="apply_job",
            actor_id=actor_id,
            counterparty_id=job["employer_id"],
            payload={"job_id": job_id},
        )
        return {"job_id": job_id}

    def _apply_hire(
        self, conn, tick, actor_id, payload, idempotency_key, now
    ) -> Dict[str, Any]:
        job_id = str(payload.get("job_id") or "")
        worker_id = self._resolve_agent_id(conn, payload.get("worker_id"))
        job = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if not job or job["status"] != "open":
            raise EconomyError("job is not open")
        if job["employer_id"] != actor_id:
            raise EconomyError("only the employer can hire")
        application = conn.execute(
            """SELECT 1 FROM job_applications
               WHERE job_id = ? AND applicant_id = ? AND status = 'pending'""",
            (job_id, worker_id),
        ).fetchone()
        if not application:
            raise EconomyError("worker has no pending application")
        conn.execute(
            "UPDATE jobs SET worker_id=?, status='hired', updated_tick=?, updated_at=? WHERE job_id=?",
            (worker_id, tick, now, job_id),
        )
        conn.execute(
            "UPDATE job_applications SET status=CASE WHEN applicant_id=? THEN 'accepted' ELSE 'rejected' END, updated_tick=? WHERE job_id=?",
            (worker_id, tick, job_id),
        )
        self._record_event(
            conn,
            tick=tick,
            event_type="hire",
            actor_id=actor_id,
            counterparty_id=worker_id,
            amount_cents=job["wage_cents"],
            payload={"job_id": job_id},
        )
        return {"job_id": job_id, "worker_id": worker_id}

    def _apply_complete_job(
        self, conn, tick, actor_id, payload, idempotency_key, now
    ) -> Dict[str, Any]:
        job_id = str(payload.get("job_id") or "")
        job = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if not job or job["status"] != "hired":
            raise EconomyError("job is not in hired state")
        if actor_id not in {job["employer_id"], job["worker_id"]}:
            raise EconomyError("only the employer or worker can complete this job")
        tx_id = self._settle(
            conn,
            tick=tick,
            action_type="complete_job",
            actor_id=job["employer_id"],
            counterparty_id=job["worker_id"],
            amount_cents=job["wage_cents"],
            idempotency_key=f"{idempotency_key}:settlement",
            payload={"job_id": job_id},
        )
        conn.execute(
            "UPDATE jobs SET status='completed', updated_tick=?, updated_at=? WHERE job_id=?",
            (tick, now, job_id),
        )
        self._record_event(
            conn,
            tick=tick,
            event_type="complete_job",
            actor_id=job["employer_id"],
            counterparty_id=job["worker_id"],
            amount_cents=job["wage_cents"],
            payload={"job_id": job_id, "transaction_id": tx_id},
        )
        return {"job_id": job_id, "transaction_id": tx_id}

    def _apply_payment(
        self,
        conn,
        tick: int,
        actor_id: str,
        payload: Dict[str, Any],
        idempotency_key: str,
        action: str,
    ) -> Dict[str, Any]:
        counterparty_key = "seller_id" if action == "buy_goods" else "counterparty_id"
        counterparty_id = self._resolve_agent_id(conn, payload.get(counterparty_key))
        amount = _positive_cents(payload.get("amount_cents", 0))
        if not counterparty_id:
            raise EconomyError("counterparty does not exist")
        tx_id = self._settle(
            conn,
            tick=tick,
            action_type=action,
            actor_id=actor_id,
            counterparty_id=counterparty_id,
            amount_cents=amount,
            idempotency_key=f"{idempotency_key}:settlement",
            payload=payload,
        )
        if action == "buy_goods":
            conn.execute(
                """INSERT INTO trades(
                    trade_id, transaction_id, tick, buyer_id, seller_id,
                    amount_cents, description, payload_json, created_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    _stable_id("trade", tx_id),
                    tx_id,
                    tick,
                    actor_id,
                    counterparty_id,
                    amount,
                    str(payload.get("description") or payload.get("item") or "goods"),
                    _json(payload),
                    utc_now(),
                ),
            )
        self._record_event(
            conn,
            tick=tick,
            event_type=action,
            actor_id=actor_id,
            counterparty_id=counterparty_id,
            amount_cents=amount,
            payload={**payload, "transaction_id": tx_id},
        )
        return {"transaction_id": tx_id}

    def _apply_transfer(self, conn, tick, actor_id, payload, idempotency_key, now):
        return self._apply_payment(
            conn, tick, actor_id, payload, idempotency_key, "transfer"
        )

    def _apply_buy_goods(self, conn, tick, actor_id, payload, idempotency_key, now):
        return self._apply_payment(
            conn, tick, actor_id, payload, idempotency_key, "buy_goods"
        )

    def _apply_intent_action(
        self,
        conn: sqlite3.Connection,
        *,
        tick: int,
        actor_id: str,
        action: str,
        payload: Dict[str, Any],
        idempotency_key: str,
        now: str,
    ) -> Dict[str, Any]:
        handlers = {
            "do_nothing": self._apply_do_nothing,
            "post_job": self._apply_post_job,
            "apply_job": self._apply_job_application,
            "hire": self._apply_hire,
            "complete_job": self._apply_complete_job,
            "transfer": self._apply_transfer,
            "buy_goods": self._apply_buy_goods,
        }
        handler = handlers.get(action)
        if not handler:
            raise EconomyError(f"unsupported action: {action}")
        return handler(conn, tick, actor_id, payload, idempotency_key, now)

    def execute_intent(
        self,
        *,
        tick: int,
        actor_id: str,
        intent: Dict[str, Any],
        idempotency_key: str,
    ) -> Dict[str, Any]:
        """Validate and atomically apply one economic intent."""
        action = str((intent or {}).get("action") or "").strip().lower()
        payload = dict((intent or {}).get("payload") or {})
        if action not in SUPPORTED_ACTIONS:
            action = action or "invalid"

        with self._lock, self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                existing = conn.execute(
                    """SELECT intent_id, action_type, status, rejection_reason
                       FROM economic_intents WHERE idempotency_key = ?""",
                    (idempotency_key,),
                ).fetchone()
                if existing:
                    conn.execute("COMMIT")
                    return {**dict(existing), "idempotent": True}

                if not self._agent_exists(conn, actor_id):
                    raise EconomyError("actor does not exist")
                intent_id = _stable_id("intent", idempotency_key)
                now = utc_now()
                conn.execute(
                    """INSERT INTO economic_intents(
                        intent_id, tick, actor_id, action_type, payload_json,
                        status, rejection_reason, idempotency_key, created_at, updated_at
                    ) VALUES(?, ?, ?, ?, ?, 'pending', NULL, ?, ?, ?)""",
                    (intent_id, int(tick), actor_id, action, _json(payload), idempotency_key, now, now),
                )

                result: Dict[str, Any] = {
                    "intent_id": intent_id,
                    "action_type": action,
                    "status": "applied",
                    "idempotent": False,
                }
                result.update(
                    self._apply_intent_action(
                        conn,
                        tick=tick,
                        actor_id=actor_id,
                        action=action,
                        payload=payload,
                        idempotency_key=idempotency_key,
                        now=now,
                    )
                )

                conn.execute(
                    "UPDATE economic_intents SET status='applied', updated_at=? WHERE intent_id=?",
                    (utc_now(), intent_id),
                )
                conn.execute("COMMIT")
                return result
            except EconomyError as exc:
                if "intent_id" not in locals():
                    conn.execute("ROLLBACK")
                    raise
                conn.execute(
                    "UPDATE economic_intents SET status='rejected', rejection_reason=?, updated_at=? WHERE intent_id=?",
                    (str(exc), utc_now(), intent_id),
                )
                self._record_event(
                    conn, tick=tick, event_type="intent_rejected", actor_id=actor_id,
                    payload={"action": action, "reason": str(exc), **payload},
                )
                conn.execute("COMMIT")
                return {
                    "intent_id": intent_id,
                    "action_type": action,
                    "status": "rejected",
                    "rejection_reason": str(exc),
                    "idempotent": False,
                }
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def reject_intent(
        self,
        *,
        tick: int,
        actor_id: str,
        idempotency_key: str,
        reason: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Persist provider and parse failures without mutating economic state."""
        with self._lock, self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT * FROM economic_intents WHERE idempotency_key=?", (idempotency_key,)
            ).fetchone()
            if existing:
                conn.execute("COMMIT")
                return {"intent_id": existing["intent_id"], "status": existing["status"], "idempotent": True}
            intent_id = _stable_id("intent", idempotency_key)
            now = utc_now()
            conn.execute(
                """INSERT INTO economic_intents(
                    intent_id, tick, actor_id, action_type, payload_json, status,
                    rejection_reason, idempotency_key, created_at, updated_at
                ) VALUES(?, ?, ?, 'provider_error', ?, 'rejected', ?, ?, ?, ?)""",
                (intent_id, tick, actor_id, _json(payload or {}), reason, idempotency_key, now, now),
            )
            self._record_event(
                conn, tick=tick, event_type="intent_rejected", actor_id=actor_id,
                payload={"reason": reason, **(payload or {})},
            )
            conn.execute("COMMIT")
            return {"intent_id": intent_id, "status": "rejected", "rejection_reason": reason}

    def claim_tick(self, tick: int, round_num: int) -> bool:
        with self._lock, self._connection() as conn:
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    "INSERT INTO economy_ticks(tick, round_num, status, started_at) VALUES(?, ?, 'running', ?)",
                    (tick, round_num, utc_now()),
                )
                conn.execute("COMMIT")
                return True
            except sqlite3.IntegrityError:
                conn.execute("ROLLBACK")
                return False

    def finish_tick(self, tick: int, *, error: Optional[str] = None) -> None:
        status = "failed" if error else "completed"
        with self._lock, self._connection() as conn:
            conn.execute(
                "UPDATE economy_ticks SET status=?, error=?, completed_at=? WHERE tick=?",
                (status, error, utc_now(), tick),
            )

    def recover_interrupted_ticks(self) -> List[int]:
        """Fail closed any tick left running by an interrupted process."""
        reason = "simulation process interrupted before tick completion"
        recovered: List[int] = []
        with self._lock, self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                "SELECT tick, round_num FROM economy_ticks WHERE status='running' ORDER BY tick"
            ).fetchall()
            completed_at = utc_now()
            for row in rows:
                tick = int(row["tick"])
                recovered.append(tick)
                conn.execute(
                    """UPDATE economy_ticks
                       SET status='failed', error=?, completed_at=?
                       WHERE tick=? AND status='running'""",
                    (reason, completed_at, tick),
                )
                self._record_event(
                    conn,
                    tick=tick,
                    event_type="tick_interrupted",
                    actor_id=None,
                    payload={"round_num": int(row["round_num"]), "reason": reason},
                )
            conn.execute("COMMIT")
        return recovered

    def assert_invariants(self) -> None:
        with self._connection() as conn:
            imbalance = conn.execute(
                "SELECT COALESCE(SUM(amount_cents), 0) AS total FROM ledger_entries"
            ).fetchone()["total"]
            negative = conn.execute(
                """SELECT a.owner_agent_id, COALESCE(SUM(l.amount_cents), 0) AS balance
                   FROM accounts a LEFT JOIN ledger_entries l ON l.account_id=a.account_id
                   WHERE a.kind='wallet' GROUP BY a.account_id HAVING balance < 0 LIMIT 1"""
            ).fetchone()
            unbalanced_transaction = conn.execute(
                """SELECT transaction_id, SUM(amount_cents) AS balance
                   FROM ledger_entries GROUP BY transaction_id
                   HAVING balance != 0 LIMIT 1"""
            ).fetchone()
        if int(imbalance) != 0:
            raise LedgerInvariantError(f"ledger entries are unbalanced by {imbalance} cents")
        if unbalanced_transaction:
            raise LedgerInvariantError(
                f"transaction {unbalanced_transaction['transaction_id']} is unbalanced by "
                f"{unbalanced_transaction['balance']} cents"
            )
        if negative:
            raise LedgerInvariantError(
                f"agent {negative['owner_agent_id']} has negative balance {negative['balance']}"
            )

    @staticmethod
    def _gini(values: List[int]) -> float:
        if not values or sum(values) == 0:
            return 0.0
        ordered = sorted(max(0, int(v)) for v in values)
        n = len(ordered)
        weighted = sum((index + 1) * value for index, value in enumerate(ordered))
        return round((2 * weighted) / (n * sum(ordered)) - (n + 1) / n, 6)

    def _metrics_snapshot(self, conn: sqlite3.Connection, tick: int) -> Dict[str, Any]:
        balances = [row["balance_cents"] for row in self.list_agents(conn=conn)]
        employed = conn.execute(
            "SELECT COUNT(*) AS n FROM jobs WHERE status='hired'"
        ).fetchone()["n"]
        open_jobs = conn.execute(
            "SELECT COUNT(*) AS n FROM jobs WHERE status='open'"
        ).fetchone()["n"]
        completed = conn.execute(
            "SELECT COUNT(*) AS n FROM jobs WHERE status='completed'"
        ).fetchone()["n"]
        volume = conn.execute(
            """SELECT COALESCE(SUM(amount_cents), 0) AS n FROM transactions
               WHERE tick=? AND action_type != 'genesis'""",
            (tick,),
        ).fetchone()["n"]
        return {
            "tick": tick,
            "agents_count": len(balances),
            "total_supply_cents": sum(balances),
            "employed_count": int(employed),
            "open_jobs": int(open_jobs),
            "completed_jobs": int(completed),
            "trade_volume_cents": int(volume),
            "gini": self._gini(balances),
            "created_at": utc_now(),
        }

    def record_metrics(self, tick: int) -> Dict[str, Any]:
        with self._lock, self._connection() as conn:
            metrics = self._metrics_snapshot(conn, tick)
            conn.execute(
                """INSERT OR REPLACE INTO economy_metrics(
                    tick, agents_count, total_supply_cents, employed_count,
                    open_jobs, completed_jobs, trade_volume_cents, gini, created_at
                ) VALUES(:tick, :agents_count, :total_supply_cents, :employed_count,
                         :open_jobs, :completed_jobs, :trade_volume_cents, :gini, :created_at)""",
                metrics,
            )
            return metrics

    def list_agents(
        self,
        *,
        limit: int = 200,
        offset: int = 0,
        conn: Optional[sqlite3.Connection] = None,
    ) -> List[Dict[str, Any]]:
        owns_conn = conn is None
        active = conn or self._connect()
        try:
            rows = active.execute(
                """SELECT e.economic_agent_id, e.oasis_agent_id, e.source_entity_uuid,
                          e.name, e.role, COALESCE(SUM(l.amount_cents), 0) AS balance_cents
                   FROM economic_agents e
                   JOIN accounts a ON a.owner_agent_id=e.economic_agent_id
                   LEFT JOIN ledger_entries l ON l.account_id=a.account_id
                   GROUP BY e.economic_agent_id
                   ORDER BY e.oasis_agent_id LIMIT ? OFFSET ?""",
                (min(max(1, int(limit)), 1000), max(0, int(offset))),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            if owns_conn:
                active.close()

    def list_jobs(self, *, limit: int = 200, offset: int = 0) -> List[Dict[str, Any]]:
        with self._connection() as conn:
            rows = conn.execute(
                """SELECT j.*, employer.name AS employer_name, worker.name AS worker_name
                   FROM jobs j
                   JOIN economic_agents employer ON employer.economic_agent_id=j.employer_id
                   LEFT JOIN economic_agents worker ON worker.economic_agent_id=j.worker_id
                   ORDER BY j.updated_tick DESC, j.created_at DESC LIMIT ? OFFSET ?""",
                (min(max(1, int(limit)), 1000), max(0, int(offset))),
            ).fetchall()
            return [dict(row) for row in rows]

    def list_events(self, *, limit: int = 200, offset: int = 0) -> List[Dict[str, Any]]:
        with self._connection() as conn:
            rows = conn.execute(
                """SELECT ev.*, actor.name AS actor_name, counterparty.name AS counterparty_name
                   FROM economy_events ev
                   LEFT JOIN economic_agents actor ON actor.economic_agent_id=ev.actor_id
                   LEFT JOIN economic_agents counterparty ON counterparty.economic_agent_id=ev.counterparty_id
                   ORDER BY ev.tick DESC, ev.created_at DESC LIMIT ? OFFSET ?""",
                (min(max(1, int(limit)), 1000), max(0, int(offset))),
            ).fetchall()
            result = []
            for row in rows:
                item = dict(row)
                item["payload"] = json.loads(item.pop("payload_json") or "{}")
                result.append(item)
            return result

    def list_ledger(self, *, limit: int = 200, offset: int = 0) -> List[Dict[str, Any]]:
        with self._connection() as conn:
            rows = conn.execute(
                """SELECT t.transaction_id, t.tick, t.action_type, t.actor_id,
                          t.counterparty_id, t.amount_cents, t.payload_json, t.created_at,
                          COALESCE(SUM(l.amount_cents), 0) AS entry_balance_cents,
                          COUNT(l.entry_id) AS entry_count
                   FROM transactions t LEFT JOIN ledger_entries l ON l.transaction_id=t.transaction_id
                   GROUP BY t.transaction_id
                   ORDER BY t.tick DESC, t.created_at DESC LIMIT ? OFFSET ?""",
                (min(max(1, int(limit)), 1000), max(0, int(offset))),
            ).fetchall()
            result = []
            for row in rows:
                item = dict(row)
                item["payload"] = json.loads(item.pop("payload_json") or "{}")
                result.append(item)
            return result

    def list_trades(self, *, limit: int = 200, offset: int = 0) -> List[Dict[str, Any]]:
        with self._connection() as conn:
            rows = conn.execute(
                """SELECT t.*, buyer.name AS buyer_name, seller.name AS seller_name
                   FROM trades t
                   JOIN economic_agents buyer ON buyer.economic_agent_id=t.buyer_id
                   JOIN economic_agents seller ON seller.economic_agent_id=t.seller_id
                   ORDER BY t.tick DESC, t.created_at DESC LIMIT ? OFFSET ?""",
                (min(max(1, int(limit)), 1000), max(0, int(offset))),
            ).fetchall()
            result = []
            for row in rows:
                item = dict(row)
                item["payload"] = json.loads(item.pop("payload_json") or "{}")
                result.append(item)
            return result

    def summary(self) -> Dict[str, Any]:
        with self._connection() as conn:
            meta = {row["key"]: row["value"] for row in conn.execute("SELECT key, value FROM economy_meta")}
            latest = conn.execute(
                "SELECT * FROM economy_metrics ORDER BY tick DESC LIMIT 1"
            ).fetchone()
            ticks = conn.execute(
                """SELECT COUNT(*) AS total,
                          SUM(status='completed') AS completed,
                          SUM(status='failed') AS failed,
                          SUM(status='running') AS running
                   FROM economy_ticks"""
            ).fetchone()
            latest_tick = conn.execute(
                """SELECT tick, round_num, status, error, started_at, completed_at
                   FROM economy_ticks ORDER BY tick DESC LIMIT 1"""
            ).fetchone()
            rejected = conn.execute(
                "SELECT COUNT(*) AS n FROM economic_intents WHERE status='rejected'"
            ).fetchone()["n"]
            settings = json.loads(meta.get("settings", "{}"))
            return {
                "enabled": bool(settings.get("enabled", False)),
                "status": meta.get("status", "ready"),
                "schema_version": int(meta.get("schema_version", SCHEMA_VERSION)),
                "settings": settings,
                "latest_metrics": dict(latest) if latest else self._metrics_snapshot(conn, 0),
                "ticks": {
                    "total": int(ticks["total"] or 0),
                    "completed": int(ticks["completed"] or 0),
                    "failed": int(ticks["failed"] or 0),
                    "running": int(ticks["running"] or 0),
                },
                "latest_tick": dict(latest_tick) if latest_tick else None,
                "rejected_intents": int(rejected),
            }

    def recent_context_for_agent(self, agent_id: str, *, limit: int = 8) -> Dict[str, Any]:
        with self._connection() as conn:
            jobs = [dict(row) for row in conn.execute(
                """SELECT job_id, title, wage_cents, status, employer_id, worker_id
                   FROM jobs WHERE employer_id=? OR worker_id=? ORDER BY updated_tick DESC LIMIT ?""",
                (agent_id, agent_id, limit),
            ).fetchall()]
            events = [dict(row) for row in conn.execute(
                """SELECT tick, event_type, actor_id, counterparty_id, amount_cents
                   FROM economy_events WHERE actor_id=? OR counterparty_id=?
                   ORDER BY tick DESC, created_at DESC LIMIT ?""",
                (agent_id, agent_id, limit),
            ).fetchall()]
            return {"balance_cents": self._balance(conn, agent_id), "jobs": jobs, "events": events}


DecisionProvider = Callable[[List[Dict[str, Any]]], Any]


class EconomyRuntime:
    """Coordinates one exact-once economic tick across social environments."""

    def __init__(
        self,
        simulation_id: str,
        config: Dict[str, Any],
        *,
        decision_provider: Optional[DecisionProvider] = None,
    ):
        self.simulation_id = simulation_id
        self.config = config
        minutes = config.get("time_config", {}).get("minutes_per_round", 60)
        self.settings = normalize_economy_settings(config.get("economy"), minutes_per_round=minutes)
        self.enabled = bool(self.settings["enabled"])
        self.halted = False
        self._lock = threading.RLock()
        self._decision_provider = decision_provider
        self.store: Optional[EconomyStore] = None
        if self.enabled:
            self.store = EconomyStore.for_simulation(simulation_id)
            self.store.recover_interrupted_ticks()
            self.store.configure(self.settings)
            profiles = config.get("agent_configs") or []
            self.store.register_agents(
                profiles,
                initial_balance_cents=self.settings["initial_balance_cents"],
                currency=self.settings["currency"],
            )

    def _default_decisions(self, contexts: List[Dict[str, Any]]) -> Any:
        from ..utils.llm_client import LLMClient

        client = LLMClient(
            base_url=self.config.get("llm_base_url"),
            model=self.config.get("llm_model"),
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You choose one bounded economic action per simulated agent. "
                    "Return JSON only as {\"decisions\":[{\"economic_agent_id\":str,"
                    "\"action\":str,\"payload\":object}]}. Allowed actions: "
                    + ", ".join(sorted(SUPPORTED_ACTIONS))
                    + ". Use integer cents. Choose do_nothing when no valid opportunity exists."
                ),
            },
            {
                "role": "user",
                "content": _json({"agents": contexts}),
            },
        ]
        return client.chat_json(
            messages,
            temperature=0.0,
            max_tokens=4096,
            disable_reasoning=True,
        )

    @staticmethod
    def _normalize_decisions(raw: Any) -> Dict[str, Dict[str, Any]]:
        decisions = raw.get("decisions") if isinstance(raw, dict) else raw
        if not isinstance(decisions, list):
            raise EconomyError("decision provider must return a decisions list")
        result: Dict[str, Dict[str, Any]] = {}
        for item in decisions:
            if not isinstance(item, dict):
                continue
            agent_id = str(item.get("economic_agent_id") or item.get("agent_id") or "")
            if agent_id:
                result[agent_id] = {
                    "action": item.get("action"),
                    "payload": item.get("payload") if isinstance(item.get("payload"), dict) else {},
                }
        return result

    def _load_social_context(self) -> Dict[str, Any]:
        path = os.path.join(
            Config.OASIS_SIMULATION_DATA_DIR,
            self.simulation_id,
            "agent_short_memory.json",
        )
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _select_agents_for_tick(self, tick: int) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        all_agents = self.store.list_agents(limit=1000)
        decision_limit = self.settings["max_decisions_per_tick"]
        if len(all_agents) <= decision_limit:
            return all_agents, all_agents
        start = ((tick - 1) * decision_limit) % len(all_agents)
        selected = (all_agents + all_agents)[start:start + decision_limit]
        return all_agents, selected

    def _build_decision_contexts(
        self,
        all_agents: List[Dict[str, Any]],
        selected_agents: List[Dict[str, Any]],
        social_context: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        merged_social_context = self._load_social_context()
        if social_context:
            merged_social_context.update(social_context)
        open_jobs = [
            job for job in self.store.list_jobs(limit=100) if job.get("status") == "open"
        ][:25]
        market_agents = [
            {
                "economic_agent_id": item["economic_agent_id"],
                "name": item["name"],
                "role": item["role"],
            }
            for item in all_agents
        ]
        market_index = {
            item["economic_agent_id"]: index for index, item in enumerate(market_agents)
        }
        contexts = []
        for agent in selected_agents:
            economic_id = agent["economic_agent_id"]
            position = market_index[economic_id]
            nearby_market = (market_agents + market_agents)[position + 1:position + 13]
            contexts.append(
                {
                    **agent,
                    **self.store.recent_context_for_agent(economic_id),
                    "social_activity": merged_social_context.get(str(agent["oasis_agent_id"]), []),
                    "available_counterparties": nearby_market,
                    "open_jobs": open_jobs,
                }
            )
        return contexts

    def _request_decisions(
        self,
        tick: int,
        agents: List[Dict[str, Any]],
        contexts: List[Dict[str, Any]],
    ) -> tuple[Dict[str, Dict[str, Any]], Optional[str]]:
        provider = self._decision_provider or self._default_decisions
        try:
            return self._normalize_decisions(provider(contexts)), None
        except Exception as exc:
            return {}, f"decision provider error: {exc}"

    def _apply_decisions(
        self,
        tick: int,
        agents: List[Dict[str, Any]],
        decisions: Dict[str, Dict[str, Any]],
        provider_error: Optional[str],
    ) -> List[Dict[str, Any]]:
        results = []
        for agent in agents:
            economic_id = agent["economic_agent_id"]
            key = f"tick:{tick}:agent:{economic_id}"
            if economic_id in decisions:
                results.append(
                    self.store.execute_intent(
                        tick=tick,
                        actor_id=economic_id,
                        intent=decisions[economic_id],
                        idempotency_key=key,
                    )
                )
            else:
                results.append(
                    self.store.reject_intent(
                        tick=tick,
                        actor_id=economic_id,
                        idempotency_key=key,
                        reason=provider_error or "decision provider omitted agent",
                    )
                )
        return results

    def _halt_economy(self, tick: int, round_num: int, error: str) -> Dict[str, Any]:
        self.halted = True
        self.store.finish_tick(tick, error=error)
        with self.store._connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO economy_meta(key, value) VALUES('status', 'halted')"
            )
        return {"tick": tick, "round_num": round_num, "halted": True, "error": error}

    def maybe_run_tick(
        self,
        round_num: int,
        *,
        social_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.enabled or self.halted or not self.store:
            return None
        rounds_per_tick = self.settings["rounds_per_tick"]
        if round_num <= 0 or round_num % rounds_per_tick != 0:
            return None
        tick = round_num // rounds_per_tick
        with self._lock:
            if not self.store.claim_tick(tick, round_num):
                return None
            try:
                all_agents, agents = self._select_agents_for_tick(tick)
                contexts = self._build_decision_contexts(all_agents, agents, social_context)
                decisions, provider_error = self._request_decisions(tick, agents, contexts)
                results = self._apply_decisions(tick, agents, decisions, provider_error)
                self.store.assert_invariants()
                metrics = self.store.record_metrics(tick)
                self.store.finish_tick(tick)
                return {"tick": tick, "round_num": round_num, "results": results, "metrics": metrics}
            except LedgerInvariantError as exc:
                return self._halt_economy(tick, round_num, str(exc))
            except Exception as exc:
                self.store.finish_tick(tick, error=str(exc))
                return {"tick": tick, "round_num": round_num, "error": str(exc)}


class EconomyTickCoordinator:
    """Synchronize social platforms before executing each economic tick once."""

    def __init__(self, runtime: EconomyRuntime, platforms: Iterable[str]):
        self.runtime = runtime
        self.platforms = {str(platform).strip().lower() for platform in platforms}
        if not self.platforms or "" in self.platforms:
            raise EconomyError("EconomyTickCoordinator requires at least one platform")
        self._lock = asyncio.Lock()
        self._rounds: Dict[int, Dict[str, Any]] = {}
        self._completed_rounds: set[int] = set()

    async def after_round(
        self, platform: str, round_num: int
    ) -> Optional[Dict[str, Any]]:
        """Wait at a tick boundary until every enabled platform has persisted."""
        platform_name = str(platform).strip().lower()
        if platform_name not in self.platforms:
            raise EconomyError(f"Unknown economy platform: {platform}")
        if not self.runtime.enabled or self.runtime.halted:
            return None
        rounds_per_tick = int(self.runtime.settings["rounds_per_tick"])
        if round_num <= 0 or round_num % rounds_per_tick != 0:
            return None

        async with self._lock:
            if round_num in self._completed_rounds:
                return None
            state = self._rounds.get(round_num)
            if state is None:
                state = {"arrivals": set(), "event": asyncio.Event()}
                self._rounds[round_num] = state
            if platform_name in state["arrivals"]:
                return None
            state["arrivals"].add(platform_name)
            should_settle = state["arrivals"] == self.platforms
            event = state["event"]

        if not should_settle:
            await event.wait()
            return None

        try:
            outcome = await asyncio.to_thread(self.runtime.maybe_run_tick, round_num)
        finally:
            async with self._lock:
                self._completed_rounds.add(round_num)
                self._rounds.pop(round_num, None)
                event.set()
        return outcome


def economy_evidence(simulation_id: str, *, limit: int = 25) -> Dict[str, Any]:
    """Read-only evidence bundle used by APIs and the report agent."""
    safe_id = str(simulation_id).strip()
    if not safe_id or safe_id in {".", ".."} or any(token in safe_id for token in ("/", "\\", "\x00")):
        raise EconomyError("Invalid simulation id")
    db_path = os.path.join(Config.OASIS_SIMULATION_DATA_DIR, safe_id, "economy.db")
    if not os.path.isfile(db_path):
        raise EconomyError("Economic twin evidence is not available for this simulation")
    store = EconomyStore(db_path, readonly=True)
    return {
        "summary": store.summary(),
        "agents": store.list_agents(limit=limit),
        "jobs": store.list_jobs(limit=limit),
        "trades": store.list_trades(limit=limit),
        "events": store.list_events(limit=limit),
        "ledger": store.list_ledger(limit=limit),
    }
