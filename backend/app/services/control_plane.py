"""Application startup wiring for durable metadata and simulation recovery."""

from __future__ import annotations

from typing import Any, Dict

from ..models.case import CaseManager
from ..models.project import ProjectManager
from ..models.task import TaskManager
from ..storage import ControlDatabase
from .simulation_runner import SimulationRunner


def initialize_control_plane(app, logger, *, log_startup: bool) -> None:
    control_db_path = app.config.get("CONTROL_DB_PATH")
    if not control_db_path:
        app.extensions["control_db"] = None
        app.extensions["control_db_migration"] = {"disabled": True}
        _configure_repositories(None)
        return

    try:
        control_db = ControlDatabase(control_db_path)
        migration_counts = control_db.migrate_legacy(
            ProjectManager.PROJECTS_DIR,
            CaseManager.CASES_DIR,
            app.config["OASIS_SIMULATION_DATA_DIR"],
        )
        app.extensions["control_db"] = control_db
        app.extensions["control_db_migration"] = migration_counts
        _configure_repositories(control_db)
        if log_startup:
            logger.info(
                "Control database ready: path=%s imported=%s",
                control_db_path,
                migration_counts,
            )
    except Exception as exc:
        logger.error("Control database initialization failed: %s", exc)
        raise RuntimeError("Control database initialization failed") from exc


def initialize_simulation_runtime(app, logger, *, log_startup: bool) -> Dict[str, int]:
    SimulationRunner.configure_repository(app.extensions.get("control_db"))
    reconciliation = SimulationRunner.reconcile_persisted_runs()
    app.extensions["simulation_reconciliation"] = reconciliation
    if log_startup and reconciliation.get("checked"):
        logger.info("Simulation restart reconciliation: %s", reconciliation)
    SimulationRunner.register_cleanup()
    return reconciliation


def _configure_repositories(repository: Any) -> None:
    ProjectManager.configure_repository(repository)
    CaseManager.configure_repository(repository)
    TaskManager.configure_repository(repository)
