import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.models.case import CaseManager
from app.models.project import ProjectManager
from app.models.task import TaskManager, TaskStatus
from app.storage.control_db import ControlDatabase


def test_legacy_json_migration_is_non_destructive_and_idempotent(tmp_path):
    uploads = tmp_path / "uploads"
    projects = uploads / "projects"
    cases = uploads / "cases"
    simulations = uploads / "simulations"
    project_file = projects / "proj_1" / "project.json"
    case_file = cases / "case_1" / "case.json"
    run_file = simulations / "sim_1" / "run_state.json"
    for path, payload in (
        (project_file, {"project_id": "proj_1", "name": "Legacy project", "created_at": "2026-01-01", "updated_at": "2026-01-01", "status": "created"}),
        (case_file, {"case_id": "case_1", "name": "Legacy case", "created_at": "2026-01-01", "updated_at": "2026-01-01", "simulation_requirement": "", "versions": []}),
        (run_file, {"simulation_id": "sim_1", "runner_status": "completed", "updated_at": "2026-01-01"}),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    database = ControlDatabase(tmp_path / "control.db")
    first = database.migrate_legacy(projects, cases, simulations)

    assert first == {"projects": 1, "cases": 1, "simulation_runs": 1}
    assert database.get_project("proj_1")["name"] == "Legacy project"
    assert database.get_case("case_1")["name"] == "Legacy case"
    assert database.get_simulation_run("sim_1")["runner_status"] == "completed"
    assert project_file.exists() and case_file.exists() and run_file.exists()

    database.put_project({**database.get_project("proj_1"), "name": "SQLite project"})
    second = database.migrate_legacy(projects, cases, simulations)

    assert second == {"projects": 0, "cases": 0, "simulation_runs": 0}
    assert database.get_project("proj_1")["name"] == "SQLite project"


def test_task_manager_restores_task_from_sqlite_after_memory_is_cleared(tmp_path):
    database = ControlDatabase(tmp_path / "control.db")
    manager = TaskManager()
    TaskManager.configure_repository(database)
    manager._tasks.clear()
    try:
        task_id = manager.create_task("graph_build", {"project_id": "proj_1"})
        manager.update_task(
            task_id,
            status=TaskStatus.PROCESSING,
            progress=42,
            message="Extracting",
        )
        manager._tasks.clear()

        restored = manager.get_task(task_id)

        assert restored is not None
        assert restored.status is TaskStatus.PROCESSING
        assert restored.progress == 42
        assert restored.metadata == {"project_id": "proj_1"}
    finally:
        TaskManager.configure_repository(None)
        manager._tasks.clear()


def test_project_and_case_managers_use_sqlite_when_configured(tmp_path, monkeypatch):
    database = ControlDatabase(tmp_path / "control.db")
    projects_dir = tmp_path / "projects"
    cases_dir = tmp_path / "cases"
    monkeypatch.setattr(ProjectManager, "PROJECTS_DIR", str(projects_dir))
    monkeypatch.setattr(CaseManager, "CASES_DIR", str(cases_dir))
    ProjectManager.configure_repository(database)
    CaseManager.configure_repository(database)
    try:
        project = ProjectManager.create_project("Transactional project")
        case = CaseManager.create_case("Transactional case")

        assert ProjectManager.get_project(project.project_id).name == "Transactional project"
        assert CaseManager.get_case(case.case_id).name == "Transactional case"
        assert database.get_project(project.project_id)["name"] == "Transactional project"
        assert database.get_case(case.case_id)["name"] == "Transactional case"
        assert not (projects_dir / project.project_id / "project.json").exists()
        assert not (cases_dir / case.case_id / "case.json").exists()
    finally:
        ProjectManager.configure_repository(None)
        CaseManager.configure_repository(None)


def test_control_database_health_reports_schema_and_wal_mode(tmp_path):
    database = ControlDatabase(tmp_path / "control.db")

    health = database.health_status()

    assert health["healthy"] is True
    assert health["schema_version"] == 1
    assert health["journal_mode"] == "wal"
    assert Path(health["path"]) == tmp_path / "control.db"


def test_transactional_case_mutation_does_not_lose_concurrent_updates(tmp_path):
    database = ControlDatabase(tmp_path / "control.db")
    database.put_case({
        "case_id": "case_1",
        "name": "Concurrent case",
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
        "revision_count": 0,
    })

    def increment(_):
        database.mutate_case(
            "case_1",
            lambda payload: {**payload, "revision_count": payload["revision_count"] + 1},
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(increment, range(40)))

    assert database.get_case("case_1")["revision_count"] == 40


def test_failed_legacy_import_can_be_retried_after_source_is_repaired(tmp_path):
    projects = tmp_path / "projects"
    project_file = projects / "proj_broken" / "project.json"
    project_file.parent.mkdir(parents=True)
    project_file.write_text("{not-json", encoding="utf-8")
    database = ControlDatabase(tmp_path / "control.db")

    try:
        database.migrate_legacy(projects, tmp_path / "cases", tmp_path / "simulations")
    except RuntimeError as exc:
        assert "Cannot import legacy metadata" in str(exc)
    else:
        raise AssertionError("invalid legacy JSON should fail the migration")

    project_file.write_text(json.dumps({
        "project_id": "proj_broken",
        "name": "Repaired",
        "created_at": "2026-01-01",
        "updated_at": "2026-01-01",
    }), encoding="utf-8")

    result = database.migrate_legacy(projects, tmp_path / "cases", tmp_path / "simulations")

    assert result["projects"] == 1
    assert database.get_project("proj_broken")["name"] == "Repaired"
