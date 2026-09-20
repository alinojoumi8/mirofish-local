import json

from app.services.simulation_runner import RunnerStatus, SimulationRunState, SimulationRunner
from app.storage.control_db import ControlDatabase


def _reset_runner():
    SimulationRunner.configure_repository(None)
    SimulationRunner._run_states.clear()


def test_run_state_is_restored_from_sqlite_and_json_is_only_a_compatibility_mirror(tmp_path, monkeypatch):
    database = ControlDatabase(tmp_path / "control.db")
    run_dir = tmp_path / "simulations"
    monkeypatch.setattr(SimulationRunner, "RUN_STATE_DIR", str(run_dir))
    SimulationRunner.configure_repository(database)
    SimulationRunner._run_states.clear()
    try:
        state = SimulationRunState(
            simulation_id="sim_1",
            runner_status=RunnerStatus.RUNNING,
            current_round=3,
            process_pid=123,
        )
        SimulationRunner._save_run_state(state)

        mirror = run_dir / "sim_1" / "run_state.json"
        assert json.loads(mirror.read_text(encoding="utf-8"))["current_round"] == 3
        database.put_simulation_run({**database.get_simulation_run("sim_1"), "current_round": 7})
        SimulationRunner._run_states.clear()

        restored = SimulationRunner.get_run_state("sim_1")

        assert restored.current_round == 7
    finally:
        _reset_runner()


def test_reconciliation_marks_missing_recorded_process_as_interrupted(tmp_path, monkeypatch):
    database = ControlDatabase(tmp_path / "control.db")
    monkeypatch.setattr(SimulationRunner, "RUN_STATE_DIR", str(tmp_path / "simulations"))
    SimulationRunner.configure_repository(database)
    SimulationRunner._run_states.clear()
    database.put_simulation_run({
        "simulation_id": "sim_orphan",
        "runner_status": "running",
        "process_pid": 999999,
        "updated_at": "2026-01-01T00:00:00",
    })

    def missing_process(pid, signal_number):
        del pid, signal_number
        raise ProcessLookupError("gone")

    monkeypatch.setattr("os.kill", missing_process)
    try:
        result = SimulationRunner.reconcile_persisted_runs()

        restored = database.get_simulation_run("sim_orphan")
        assert result == {"checked": 1, "alive": 0, "interrupted": 1}
        assert restored["runner_status"] == "stopped"
        assert restored["process_pid"] is None
        assert "interrupted during backend restart" in restored["error"].lower()
    finally:
        _reset_runner()


def test_reconciliation_keeps_live_recorded_process_running(tmp_path, monkeypatch):
    database = ControlDatabase(tmp_path / "control.db")
    monkeypatch.setattr(SimulationRunner, "RUN_STATE_DIR", str(tmp_path / "simulations"))
    SimulationRunner.configure_repository(database)
    SimulationRunner._run_states.clear()
    database.put_simulation_run({
        "simulation_id": "sim_live",
        "runner_status": "running",
        "process_pid": 123,
        "updated_at": "2026-01-01T00:00:00",
    })
    monkeypatch.setattr("os.kill", lambda pid, signal_number: None)
    try:
        result = SimulationRunner.reconcile_persisted_runs()

        assert result == {"checked": 1, "alive": 1, "interrupted": 0}
        assert database.get_simulation_run("sim_live")["runner_status"] == "running"
    finally:
        _reset_runner()
