"""Contract tests for the ensemble simulation endpoints.

These endpoints are what the Step 3 UI calls to run a multi-seed ensemble and poll
its progress, so the request/response shape (task_id on start, task dict / not_started
on status) is a contract the frontend depends on.
"""

import pytest

import app.api.simulation as simulation_api
import app.services.ensemble_runner as ensemble_runner_module
from app import create_app
from app.models.task import TaskManager


class FakeNeo4jStorage:
    def health_status(self):
        return {
            "healthy": True,
            "embedding": {"healthy": True},
            "vector_search_usable": True,
        }


class EnsembleTestConfig:
    TESTING = True
    DEBUG = False
    JSON_AS_ASCII = False


@pytest.fixture
def client(monkeypatch):
    import app.storage as storage_module
    from app.services.simulation_runner import SimulationRunner

    monkeypatch.setattr(storage_module, "Neo4jStorage", FakeNeo4jStorage)
    monkeypatch.setattr(SimulationRunner, "register_cleanup", classmethod(lambda cls: None))
    app = create_app(EnsembleTestConfig)
    app.config["PROPAGATE_EXCEPTIONS"] = False
    return app.test_client()


def test_run_ensemble_requires_simulation_id(client):
    resp = client.post("/api/simulation/run-ensemble", json={})
    assert resp.status_code == 400
    body = resp.get_json()
    assert body["success"] is False


def test_run_ensemble_rejects_unprepared_simulation(client, monkeypatch):
    monkeypatch.setattr(simulation_api, "_check_simulation_prepared", lambda sid: (False, {}))
    resp = client.post("/api/simulation/run-ensemble", json={"simulation_id": "sim_x"})
    assert resp.status_code == 400
    assert resp.get_json()["success"] is False


def test_run_ensemble_returns_task_id_for_prepared_simulation(client, monkeypatch):
    monkeypatch.setattr(simulation_api, "_check_simulation_prepared", lambda sid: (True, {}))

    calls = {}

    def fake_run_ensemble(simulation_id, **kwargs):
        calls["simulation_id"] = simulation_id
        calls["runs"] = kwargs.get("runs")
        # Exercise the progress callback so a running task also has a shape.
        cb = kwargs.get("progress_callback")
        if cb:
            cb(1, 3, {"total_actions": 4})
        return {"runs": 3, "completed_runs": 3, "mean_net": 0.1, "std_net": 0.02, "total_actions": 12}

    monkeypatch.setattr(ensemble_runner_module.EnsembleRunner, "run_ensemble", staticmethod(fake_run_ensemble))

    resp = client.post("/api/simulation/run-ensemble", json={"simulation_id": "sim_ok", "runs": 3})
    assert resp.status_code == 200
    data = resp.get_json()["data"]
    assert data["simulation_id"] == "sim_ok"
    assert data["task_id"]  # frontend polls on this

    # The background task should reference the task we can query by id.
    status = client.post("/api/simulation/run-ensemble/status", json={"task_id": data["task_id"]})
    assert status.status_code == 200
    task = status.get_json()["data"]
    assert task["task_id"] == data["task_id"]
    assert task["status"] in {"pending", "processing", "completed"}


def test_ensemble_status_reports_not_started_without_signal(client):
    resp = client.post(
        "/api/simulation/run-ensemble/status",
        json={"simulation_id": "sim_that_never_ran"},
    )
    assert resp.status_code == 200
    assert resp.get_json()["data"]["status"] == "not_started"


def test_ensemble_status_returns_completed_task_result(client):
    manager = TaskManager()
    task_id = manager.create_task("ensemble_run", metadata={"simulation_id": "sim_done"})
    manager.complete_task(task_id, {"simulation_id": "sim_done", "ensemble": {"runs": 3, "mean_net": 0.2}})

    resp = client.post("/api/simulation/run-ensemble/status", json={"task_id": task_id})
    assert resp.status_code == 200
    task = resp.get_json()["data"]
    assert task["status"] == "completed"
    assert task["result"]["ensemble"]["runs"] == 3


def test_ensemble_status_requires_identifier(client):
    resp = client.post("/api/simulation/run-ensemble/status", json={})
    assert resp.status_code == 400
    assert resp.get_json()["success"] is False


def test_ensemble_forwards_economy_and_round_cap(client, monkeypatch):
    import threading
    monkeypatch.setattr(simulation_api, "_check_simulation_prepared", lambda sid: (True, {}))
    calls = {}
    finished = threading.Event()
    def run(simulation_id, **kwargs):
        calls.update(kwargs)
        finished.set()
        return {"completed_runs": 1}
    monkeypatch.setattr(ensemble_runner_module.EnsembleRunner, "run_ensemble", staticmethod(run))
    response = client.post("/api/simulation/run-ensemble", json={
        "simulation_id": "sim_options", "runs": 2, "max_rounds": 17,
        "economy": {"enabled": True}, "enable_graph_memory_update": False,
    })
    assert response.status_code == 200
    assert finished.wait(2)
    assert calls["max_rounds"] == 17
    assert calls["economy"] == {"enabled": True}
