import json
import logging

from app.services.simulation_ipc import SimulationIPCClient


class CapturingHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


def capture_logger(name):
    target_logger = logging.getLogger(name)
    handler = CapturingHandler()
    previous_level = target_logger.level
    target_logger.setLevel(logging.DEBUG)
    target_logger.addHandler(handler)
    return target_logger, handler, previous_level


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

    target_logger, handler, previous_level = capture_logger("mirofish.simulation_ipc")
    try:
        assert SimulationIPCClient(str(tmp_path)).check_env_alive() is False
    finally:
        target_logger.removeHandler(handler)
        target_logger.setLevel(previous_level)

    assert any(
        record.levelno == logging.WARNING
        and "IPC environment marked alive but process is not running" in record.getMessage()
        and "pid=999999" in record.getMessage()
        for record in handler.records
    )


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
