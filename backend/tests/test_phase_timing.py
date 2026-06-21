from app.services.report_agent import Report, ReportManager, ReportStatus
from app.services.simulation_manager import SimulationState
from app.services.simulation_runner import RunnerStatus, SimulationRunState
from app.utils.timing import PhaseTimer


def test_phase_timer_records_ordered_phases_and_totals():
    timer = PhaseTimer("graph_build")

    with timer.time("chunking", label="Chunk documents"):
        pass
    with timer.time("extraction", label="Extract graph facts", metadata={"chunks": 2}):
        pass

    summary = timer.complete()

    assert summary["name"] == "graph_build"
    assert summary["status"] == "completed"
    assert summary["total_seconds"] >= 0
    assert [phase["name"] for phase in summary["phases"]] == ["chunking", "extraction"]
    assert summary["phase_totals"]["chunking"] >= 0
    assert summary["phase_totals"]["extraction"] >= 0
    assert summary["phases"][1]["metadata"]["chunks"] == 2


def test_report_serializes_top_level_timing():
    report = Report(
        report_id="report_timing",
        simulation_id="sim_1",
        graph_id="graph_1",
        simulation_requirement="test",
        status=ReportStatus.COMPLETED,
        timing={"name": "report_generation", "total_seconds": 12.3, "phases": []},
    )

    data = report.to_dict()
    restored = ReportManager._dict_to_report("report_timing", data)

    assert data["timing"]["total_seconds"] == 12.3
    assert restored.timing["name"] == "report_generation"


def test_simulation_state_serializes_prepare_timing():
    state = SimulationState(
        simulation_id="sim_1",
        project_id="proj_1",
        graph_id="graph_1",
        timings={
            "simulation_prepare": {
                "name": "simulation_prepare",
                "total_seconds": 4.2,
                "phases": [],
            }
        },
    )

    data = state.to_dict()

    assert data["timings"]["simulation_prepare"]["total_seconds"] == 4.2


def test_run_state_includes_live_wall_clock_timing(monkeypatch):
    state = SimulationRunState(
        simulation_id="sim_1",
        runner_status=RunnerStatus.RUNNING,
        total_rounds=10,
        total_simulation_hours=10,
        started_at="2026-05-19T01:00:00",
    )

    data = state.to_dict()

    assert data["timing"]["run"]["started_at"] == "2026-05-19T01:00:00"
    assert data["timing"]["run"]["duration_seconds"] >= 0
