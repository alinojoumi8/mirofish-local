import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest
from flask import Flask

from app.api import simulation_bp
from app.config import Config
from app.services.economy import (
    EconomyRuntime,
    EconomyStore,
    normalize_economy_settings,
)
from app.services.simulation_runner import SimulationRunner


PROFILES = [
    {
        "agent_id": 1,
        "entity_uuid": "entity-employer",
        "entity_name": "Employer",
        "entity_type": "company",
    },
    {
        "agent_id": 2,
        "entity_uuid": "entity-worker",
        "entity_name": "Worker",
        "entity_type": "person",
    },
]


def make_store(tmp_path):
    store = EconomyStore(str(tmp_path / "economy.db"))
    store.configure(normalize_economy_settings({"enabled": True}, minutes_per_round=60))
    agents = store.register_agents(PROFILES, initial_balance_cents=100_000, currency="USD")
    return store, agents


def test_economy_settings_default_to_opt_out_and_one_simulated_day():
    settings = normalize_economy_settings(None, minutes_per_round=30)

    assert settings["enabled"] is False
    assert settings["rounds_per_tick"] == 48
    assert settings["initial_balance_cents"] == 1_000_000
    assert normalize_economy_settings({"enabled": "false"})["enabled"] is False
    with pytest.raises(ValueError, match="initial_balance_cents"):
        normalize_economy_settings({"initial_balance_cents": -1})


def test_identity_map_and_genesis_are_idempotent(tmp_path):
    store, agents = make_store(tmp_path)

    repeated = store.register_agents(PROFILES, initial_balance_cents=100_000, currency="USD")

    assert repeated == agents
    assert len(store.list_agents()) == 2
    assert sum(agent["balance_cents"] for agent in store.list_agents()) == 200_000
    store.assert_invariants()


def test_transfer_is_balanced_idempotent_and_rejects_insufficient_funds(tmp_path):
    store, agents = make_store(tmp_path)
    sender, receiver = [agent["economic_agent_id"] for agent in agents]

    first = store.execute_intent(
        tick=1,
        actor_id=sender,
        intent={
            "action": "transfer",
            "payload": {"counterparty_id": receiver, "amount_cents": 12_500},
        },
        idempotency_key="transfer-1",
    )
    repeated = store.execute_intent(
        tick=1,
        actor_id=sender,
        intent={
            "action": "transfer",
            "payload": {"counterparty_id": receiver, "amount_cents": 12_500},
        },
        idempotency_key="transfer-1",
    )
    rejected = store.execute_intent(
        tick=1,
        actor_id=sender,
        intent={
            "action": "transfer",
            "payload": {"counterparty_id": receiver, "amount_cents": 999_999},
        },
        idempotency_key="transfer-too-large",
    )

    balances = {agent["economic_agent_id"]: agent["balance_cents"] for agent in store.list_agents()}
    assert first["status"] == "applied"
    assert repeated["idempotent"] is True
    assert rejected["status"] == "rejected"
    assert rejected["rejection_reason"] == "insufficient funds"
    assert balances[sender] == 87_500
    assert balances[receiver] == 112_500
    assert all(item["entry_balance_cents"] == 0 for item in store.list_ledger())
    store.assert_invariants()


def test_job_state_machine_settles_wage_once(tmp_path):
    store, agents = make_store(tmp_path)
    employer, worker = [agent["economic_agent_id"] for agent in agents]

    posted = store.execute_intent(
        tick=1,
        actor_id=employer,
        intent={
            "action": "post_job",
            "payload": {"title": "Research", "description": "Analyze signals", "wage_cents": 25_000},
        },
        idempotency_key="job-post",
    )
    job_id = posted["job_id"]
    applied = store.execute_intent(
        tick=1,
        actor_id=worker,
        intent={"action": "apply_job", "payload": {"job_id": job_id}},
        idempotency_key="job-apply",
    )
    hired = store.execute_intent(
        tick=2,
        actor_id=employer,
        intent={"action": "hire", "payload": {"job_id": job_id, "worker_id": worker}},
        idempotency_key="job-hire",
    )
    completed = store.execute_intent(
        tick=3,
        actor_id=worker,
        intent={"action": "complete_job", "payload": {"job_id": job_id}},
        idempotency_key="job-complete",
    )
    repeated = store.execute_intent(
        tick=3,
        actor_id=worker,
        intent={"action": "complete_job", "payload": {"job_id": job_id}},
        idempotency_key="job-complete",
    )

    jobs = store.list_jobs()
    balances = {agent["economic_agent_id"]: agent["balance_cents"] for agent in store.list_agents()}
    assert applied["status"] == hired["status"] == completed["status"] == "applied"
    assert repeated["idempotent"] is True
    assert jobs[0]["status"] == "completed"
    assert balances[employer] == 75_000
    assert balances[worker] == 125_000
    store.assert_invariants()


def test_parallel_runtime_claims_each_tick_exactly_once(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "OASIS_SIMULATION_DATA_DIR", str(tmp_path))
    calls = []

    def provider(contexts):
        calls.append(contexts)
        return {
            "decisions": [
                {
                    "economic_agent_id": item["economic_agent_id"],
                    "action": "do_nothing",
                    "payload": {},
                }
                for item in contexts
            ]
        }

    config = {
        "simulation_id": "sim-exact-once",
        "time_config": {"minutes_per_round": 60},
        "agent_configs": PROFILES,
        "economy": {"enabled": True, "rounds_per_tick": 1},
    }
    runtime = EconomyRuntime(
        "sim-exact-once",
        config,
        decision_provider=provider,
    )
    second_runtime = EconomyRuntime(
        "sim-exact-once",
        config,
        decision_provider=provider,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda item: item.maybe_run_tick(1), [runtime, second_runtime]))

    assert sorted(outcome is None for outcome in outcomes) == [False, True]
    assert next(outcome for outcome in outcomes if outcome is not None)["tick"] == 1
    assert len(calls) == 1
    assert runtime.store.summary()["ticks"] == {"total": 1, "completed": 1, "failed": 0}


def test_provider_parse_failure_rejects_intents_without_halting_social_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "OASIS_SIMULATION_DATA_DIR", str(tmp_path))

    def broken_provider(_contexts):
        raise ValueError("not json")

    runtime = EconomyRuntime(
        "sim-provider-error",
        {
            "simulation_id": "sim-provider-error",
            "time_config": {"minutes_per_round": 60},
            "agent_configs": PROFILES,
            "economy": {"enabled": True, "rounds_per_tick": 1},
        },
        decision_provider=broken_provider,
    )

    outcome = runtime.maybe_run_tick(1)

    assert outcome["metrics"]["total_supply_cents"] == 2_000_000
    assert runtime.halted is False
    assert runtime.store.summary()["rejected_intents"] == 2
    runtime.store.assert_invariants()


def test_disabled_runtime_creates_no_economic_database(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "OASIS_SIMULATION_DATA_DIR", str(tmp_path))
    runtime = EconomyRuntime(
        "sim-disabled",
        {"time_config": {"minutes_per_round": 60}, "agent_configs": PROFILES},
    )

    assert runtime.enabled is False
    assert runtime.maybe_run_tick(24) is None
    assert not (tmp_path / "sim-disabled" / "economy.db").exists()


def test_ledger_database_reports_zero_total_directly(tmp_path):
    store, agents = make_store(tmp_path)
    sender, receiver = [agent["economic_agent_id"] for agent in agents]
    store.execute_intent(
        tick=1,
        actor_id=sender,
        intent={"action": "buy_goods", "payload": {"seller_id": receiver, "amount_cents": 500}},
        idempotency_key="purchase",
    )

    with sqlite3.connect(store.db_path) as conn:
        assert conn.execute("SELECT SUM(amount_cents) FROM ledger_entries").fetchone()[0] == 0
    assert store.list_trades()[0]["amount_cents"] == 500


def test_economy_read_apis_expose_summary_and_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "OASIS_SIMULATION_DATA_DIR", str(tmp_path))
    simulation_dir = tmp_path / "sim-api"
    store = EconomyStore(str(simulation_dir / "economy.db"))
    settings = normalize_economy_settings({"enabled": True}, minutes_per_round=60)
    store.configure(settings)
    store.register_agents(PROFILES, initial_balance_cents=100_000, currency="USD")
    store.record_metrics(0)

    app = Flask(__name__)
    app.register_blueprint(simulation_bp, url_prefix="/api/simulation")
    client = app.test_client()

    summary = client.get("/api/simulation/sim-api/economy/summary")
    agents = client.get("/api/simulation/sim-api/economy/agents")
    jobs = client.get("/api/simulation/sim-api/economy/jobs")
    events = client.get("/api/simulation/sim-api/economy/events")
    ledger = client.get("/api/simulation/sim-api/economy/ledger")
    missing = client.get("/api/simulation/no-economy/economy/summary")

    assert summary.status_code == 200
    assert summary.get_json()["data"]["latest_metrics"]["agents_count"] == 2
    assert agents.get_json()["data"]["count"] == 2
    assert jobs.get_json()["data"]["count"] == 0
    assert events.status_code == 200
    assert ledger.get_json()["data"]["count"] == 2
    assert missing.status_code == 404


def test_force_restart_cleanup_removes_economy_database_sidecars(tmp_path, monkeypatch):
    monkeypatch.setattr(SimulationRunner, "RUN_STATE_DIR", str(tmp_path))
    simulation_dir = tmp_path / "sim-cleanup"
    simulation_dir.mkdir()
    for filename in ("economy.db", "economy.db-wal", "economy.db-shm"):
        (simulation_dir / filename).write_bytes(b"runtime")

    result = SimulationRunner.cleanup_simulation_logs("sim-cleanup")

    assert result["success"] is True
    assert all(not (simulation_dir / filename).exists() for filename in (
        "economy.db", "economy.db-wal", "economy.db-shm"
    ))
