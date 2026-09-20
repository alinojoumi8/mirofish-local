import logging

import pytest

from flask import jsonify

from app import create_app
from app.models.case import CaseManager
from app.models.task import TaskManager, TaskStatus


class CapturingHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def capture_named_logger():
    captures = []

    def _capture(name, level=logging.DEBUG):
        target_logger = logging.getLogger(name)
        handler = CapturingHandler()
        previous_level = target_logger.level
        target_logger.setLevel(level)
        target_logger.addHandler(handler)
        captures.append((target_logger, handler, previous_level))
        return handler.records

    yield _capture

    for target_logger, handler, previous_level in captures:
        target_logger.removeHandler(handler)
        target_logger.setLevel(previous_level)


class FakeNeo4jStorage:
    def health_status(self):
        return {
            "healthy": True,
            "embedding": {"healthy": True},
            "vector_search_usable": True,
        }


class LoggingTestConfig:
    TESTING = True
    DEBUG = False
    JSON_AS_ASCII = False
    EXPOSE_INTERNAL_ERRORS = False
    CORS_ORIGINS = ["http://127.0.0.1:3000", "http://localhost:3000"]


@pytest.fixture
def app_client(monkeypatch):
    import app.storage as storage_module
    from app.services.simulation_runner import SimulationRunner

    monkeypatch.setattr(storage_module, "Neo4jStorage", FakeNeo4jStorage)
    monkeypatch.setattr(SimulationRunner, "register_cleanup", classmethod(lambda cls: None))
    app = create_app(LoggingTestConfig)
    app.config["PROPAGATE_EXCEPTIONS"] = False

    @app.route("/boom")
    def boom():
        raise RuntimeError("exploded")

    @app.route("/leaky-error")
    def leaky_error():
        return jsonify({
            "success": False,
            "error": "database exploded",
            "traceback": "secret stack and filesystem path",
        }), 500

    @app.route("/api/test-health")
    def api_test_health():
        return jsonify({"success": True})

    return app.test_client()


def test_request_logging_records_method_path_status_and_duration(app_client, capture_named_logger):
    records = capture_named_logger("mirofish.request")

    response = app_client.get("/health")

    assert response.status_code == 200
    assert any(
        record.levelno == logging.DEBUG
        and "HTTP GET /health -> 200" in record.getMessage()
        and "duration_ms=" in record.getMessage()
        for record in records
    )


def test_request_logging_warns_for_client_errors(app_client, capture_named_logger):
    records = capture_named_logger("mirofish.request")

    response = app_client.get("/missing")

    assert response.status_code == 404
    assert any(
        record.levelno == logging.WARNING
        and "HTTP GET /missing -> 404" in record.getMessage()
        and "duration_ms=" in record.getMessage()
        for record in records
    )


def test_request_logging_errors_for_unhandled_exceptions(app_client, capture_named_logger):
    records = capture_named_logger("mirofish.request")

    response = app_client.get("/boom")

    assert response.status_code == 500
    assert any(
        record.levelno == logging.ERROR
        and "Unhandled exception during GET /boom" in record.getMessage()
        and record.exc_info
        for record in records
    )
    assert any(
        record.levelno == logging.ERROR
        and "HTTP GET /boom -> 500" in record.getMessage()
        and "duration_ms=" in record.getMessage()
        for record in records
    )


def test_internal_traceback_is_removed_from_json_error_response(app_client):
    response = app_client.get("/leaky-error")

    assert response.status_code == 500
    payload = response.get_json()
    assert payload["success"] is False
    assert payload["error"] == "database exploded"
    assert payload["error_code"] == "internal_error"
    assert payload["request_id"]
    assert "traceback" not in payload


def test_cors_rejects_unconfigured_origins(app_client):
    response = app_client.get("/api/test-health", headers={"Origin": "https://attacker.example"})

    assert "Access-Control-Allow-Origin" not in response.headers


def test_cors_allows_configured_local_origin(app_client):
    response = app_client.get("/api/test-health", headers={"Origin": "http://127.0.0.1:3000"})

    assert response.headers["Access-Control-Allow-Origin"] == "http://127.0.0.1:3000"


def test_status_includes_control_plane_in_overall_readiness(app_client):
    response = app_client.get("/api/status")

    assert response.status_code == 200
    data = response.get_json()["data"]
    assert data["neo4j"]["healthy"] is True
    assert data["embedding"]["healthy"] is True
    assert data["control_db"]["healthy"] is False
    assert data["healthy"] is False


def test_task_manager_logs_lifecycle_transitions(capture_named_logger):
    records = capture_named_logger("mirofish.task")
    manager = TaskManager()
    manager._tasks.clear()

    task_id = manager.create_task(
        "graph_build",
        metadata={"project_id": "proj_1", "graph_id": "graph_1"},
    )
    manager.update_task(
        task_id,
        status=TaskStatus.PROCESSING,
        progress=25,
        message="Extracting graph entities",
    )
    manager.complete_task(task_id, {"graph_id": "graph_1"})

    messages = [record.getMessage() for record in records]
    assert any(
        f"Task created: task_id={task_id} task_type=graph_build" in message
        and "metadata_keys=graph_id,project_id" in message
        for message in messages
    )
    assert any(
        f"Task status changed: task_id={task_id} pending -> processing" in message
        and "progress=25" in message
        and "message=Extracting graph entities" in message
        for message in messages
    )
    assert any(
        f"Task completed: task_id={task_id}" in message
        and "result_keys=graph_id" in message
        for message in messages
    )


def test_task_manager_warns_when_updating_missing_task(capture_named_logger):
    records = capture_named_logger("mirofish.task")
    manager = TaskManager()
    manager._tasks.clear()

    manager.update_task(
        "missing-task",
        status=TaskStatus.FAILED,
        error="worker crashed",
    )

    assert any(
        record.levelno == logging.WARNING
        and "Task update skipped: task_id=missing-task not found" in record.getMessage()
        and "requested_status=failed" in record.getMessage()
        for record in records
    )


def test_task_manager_logs_task_failures(capture_named_logger):
    records = capture_named_logger("mirofish.task")
    manager = TaskManager()
    manager._tasks.clear()

    task_id = manager.create_task("simulation_prepare")
    manager.fail_task(task_id, "profile generation crashed")

    assert any(
        record.levelno == logging.ERROR
        and f"Task failed: task_id={task_id}" in record.getMessage()
        and "error=profile generation crashed" in record.getMessage()
        for record in records
    )


def test_case_snapshot_warns_when_case_missing(capture_named_logger, tmp_path, monkeypatch):
    monkeypatch.setattr(CaseManager, "CASES_DIR", str(tmp_path / "cases"))
    records = capture_named_logger("mirofish.case")

    result = CaseManager.record_prediction_snapshot("case_missing", "ver_x", {"Win": 0.5})

    assert result is None
    assert any(
        record.levelno == logging.WARNING
        and "record_prediction_snapshot skipped: case not found" in record.getMessage()
        and "case_id=case_missing" in record.getMessage()
        for record in records
    )


def test_case_snapshot_warns_when_version_missing(capture_named_logger, tmp_path, monkeypatch):
    monkeypatch.setattr(CaseManager, "CASES_DIR", str(tmp_path / "cases"))
    records = capture_named_logger("mirofish.case")
    case = CaseManager.create_case(name="Doe v Roe")

    result = CaseManager.record_prediction_snapshot(case.case_id, "ver_missing", {"Win": 0.5})

    assert result is None
    assert any(
        record.levelno == logging.WARNING
        and "record_prediction_snapshot skipped: version not found" in record.getMessage()
        and "ver_missing" in record.getMessage()
        for record in records
    )


def test_case_snapshot_logs_success(capture_named_logger, tmp_path, monkeypatch):
    monkeypatch.setattr(CaseManager, "CASES_DIR", str(tmp_path / "cases"))
    records = capture_named_logger("mirofish.case")
    case = CaseManager.create_case(name="Doe v Roe")
    version = CaseManager.create_version(case.case_id, project_id="proj_1")

    result = CaseManager.record_prediction_snapshot(
        case.case_id, version.version_id, {"Win": 0.5, "_meta": {"report_id": "rep_1"}}, report_id="rep_1",
    )

    assert result is not None
    assert any(
        record.levelno == logging.INFO
        and "Recorded prediction snapshot" in record.getMessage()
        and "outcomes=1" in record.getMessage()  # _meta excluded from the count
        for record in records
    )


def test_update_version_warns_when_case_missing(capture_named_logger, tmp_path, monkeypatch):
    monkeypatch.setattr(CaseManager, "CASES_DIR", str(tmp_path / "cases"))
    records = capture_named_logger("mirofish.case")

    result = CaseManager.update_version("case_missing", "ver_x", status="built")

    assert result is None
    assert any(
        record.levelno == logging.WARNING
        and "update_version skipped: case not found" in record.getMessage()
        and "fields=status" in record.getMessage()
        for record in records
    )


def test_update_version_warns_when_version_missing(capture_named_logger, tmp_path, monkeypatch):
    monkeypatch.setattr(CaseManager, "CASES_DIR", str(tmp_path / "cases"))
    records = capture_named_logger("mirofish.case")
    case = CaseManager.create_case(name="Doe v Roe")

    result = CaseManager.update_version(case.case_id, "ver_missing", graph_id="g1")

    assert result is None
    assert any(
        record.levelno == logging.WARNING
        and "update_version skipped: version not found" in record.getMessage()
        for record in records
    )


def test_update_version_warns_on_unknown_field(capture_named_logger, tmp_path, monkeypatch):
    monkeypatch.setattr(CaseManager, "CASES_DIR", str(tmp_path / "cases"))
    records = capture_named_logger("mirofish.case")
    case = CaseManager.create_case(name="Doe v Roe")
    version = CaseManager.create_version(case.case_id, project_id="proj_1")

    CaseManager.update_version(case.case_id, version.version_id, not_a_real_field="x")

    assert any(
        record.levelno == logging.WARNING
        and "update_version ignoring unknown field" in record.getMessage()
        and "field=not_a_real_field" in record.getMessage()
        for record in records
    )


def test_numeric_forecast_warns_when_market_enrichment_fails(capture_named_logger, monkeypatch):
    import app.utils.market_data as market_data
    from app.services.forecasting import ForecastSynthesizer

    records = capture_named_logger("mirofish.forecast")
    monkeypatch.setattr(market_data, "is_enabled", lambda: True)

    def _boom(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(market_data, "resolve_target", _boom)

    synth = ForecastSynthesizer("sim_test", simulation_config={"forecast_mode": "market_economy"})
    target = {
        "kind": "price",
        "symbol": "^GSPC",
        "bands": [
            {"label": "Down", "lo": None, "hi": 0.0},
            {"label": "Up", "lo": 0.0, "hi": None},
        ],
    }

    result = synth._numeric_forecast(target, 0.1)

    assert result is not None  # degrades gracefully rather than crashing
    assert any(
        record.levelno == logging.WARNING
        and "Market-data enrichment failed" in record.getMessage()
        and "^GSPC" in record.getMessage()
        for record in records
    )


def test_file_parser_warns_when_document_extraction_fails(capture_named_logger, tmp_path):
    from app.utils.file_parser import FileParser

    records = capture_named_logger("mirofish.file_parser")
    missing = str(tmp_path / "does_not_exist.pdf")

    merged = FileParser.extract_from_multiple([missing])

    # Degrades gracefully: a placeholder still appears in the merged text...
    assert "extraction failed" in merged
    # ...but the failure is now also visible in the logs.
    assert any(
        record.levelno == logging.WARNING
        and "Failed to extract text from document" in record.getMessage()
        and "does_not_exist.pdf" in record.getMessage()
        for record in records
    )
