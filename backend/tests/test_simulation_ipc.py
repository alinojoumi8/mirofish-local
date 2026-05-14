import json

from app.services.simulation_ipc import SimulationIPCClient


def test_check_env_alive_rejects_stale_alive_status_when_process_missing(tmp_path, monkeypatch):
    (tmp_path / "env_status.json").write_text(
        json.dumps({"status": "alive"}),
        encoding="utf-8",
    )
    (tmp_path / "run_state.json").write_text(
        json.dumps({"process_pid": 999999}),
        encoding="utf-8",
    )

    def fake_kill(_pid, _signal):
        raise OSError("no process")

    monkeypatch.setattr("os.kill", fake_kill)

    assert SimulationIPCClient(str(tmp_path)).check_env_alive() is False


def test_check_env_alive_accepts_live_recorded_process(tmp_path, monkeypatch):
    (tmp_path / "env_status.json").write_text(
        json.dumps({"status": "alive"}),
        encoding="utf-8",
    )
    (tmp_path / "run_state.json").write_text(
        json.dumps({"process_pid": 123}),
        encoding="utf-8",
    )

    monkeypatch.setattr("os.kill", lambda _pid, _signal: None)

    assert SimulationIPCClient(str(tmp_path)).check_env_alive() is True
