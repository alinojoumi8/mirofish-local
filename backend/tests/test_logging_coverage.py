import logging

import pytest

from app import create_app
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
