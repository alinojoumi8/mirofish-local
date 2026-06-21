import json
from pathlib import Path

from app.models.case import CaseManager, DocumentType, normalize_prediction_settings
from app.models.project import Project, ProjectManager, ProjectStatus
from app.services.graph_builder import GraphBuilderService
from app.services.simulation_runner import SimulationRunner


def test_case_manager_creates_versions_and_preserves_document_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(CaseManager, "CASES_DIR", str(tmp_path / "cases"))

    case = CaseManager.create_case(
        name="Smith v Jones",
        simulation_requirement="Predict the injunction motion outcome.",
    )
    version = CaseManager.create_version(
        case.case_id,
        project_id="proj_1",
        graph_id="graph_1",
        documents=[
            {
                "filename": "statement-of-claim.pdf",
                "size": 123,
                "text": "Plaintiff pleads breach of contract.",
            },
            {
                "filename": "reply_factum.txt",
                "size": 456,
                "text": "The defendant says the motion should fail.",
            },
        ],
        build_settings={"build_preset": "fast_scan"},
    )

    loaded = CaseManager.get_case(case.case_id)

    assert loaded is not None
    assert loaded.case_id == case.case_id
    assert loaded.versions[0].version_id == version.version_id
    assert loaded.versions[0].version_number == 1
    assert loaded.versions[0].documents[0].document_type == DocumentType.PLEADING
    assert loaded.versions[0].documents[1].document_type == DocumentType.FACTUM
    assert loaded.versions[0].documents[0].provenance["case_id"] == case.case_id
    assert loaded.versions[0].documents[0].provenance["version_id"] == version.version_id
    assert loaded.versions[0].documents[0].hash


def test_project_serializes_case_and_prediction_metadata():
    project = Project(
        project_id="proj_1",
        name="Smith v Jones",
        status=ProjectStatus.ONTOLOGY_GENERATED,
        created_at="2026-05-18T12:00:00",
        updated_at="2026-05-18T12:00:00",
        case_id="case_1",
        case_version_id="casever_1",
        case_version_number=2,
        graph_build_settings={"build_preset": "deep_evidence", "chunk_size": 1200},
        prediction_settings={"preset": "quick_legal_read", "agent_count": 24},
        documents=[{"document_id": "doc_1", "filename": "motion.pdf"}],
    )

    round_tripped = Project.from_dict(project.to_dict())

    assert round_tripped.case_id == "case_1"
    assert round_tripped.case_version_id == "casever_1"
    assert round_tripped.case_version_number == 2
    assert round_tripped.graph_build_settings["build_preset"] == "deep_evidence"
    assert round_tripped.prediction_settings["agent_count"] == 24
    assert round_tripped.documents[0]["filename"] == "motion.pdf"


def test_graph_build_presets_and_document_chunking_attach_provenance():
    settings = GraphBuilderService.resolve_build_settings(
        {
            "build_preset": "fast_scan",
            "chunk_size": 9999,
            "chunk_overlap": 50,
            "batch_size": 4,
            "llm_concurrency": 3,
            "incremental": True,
        }
    )
    chunks = GraphBuilderService.split_documents_with_provenance(
        [
            {
                "document_id": "doc_1",
                "filename": "affidavit.pdf",
                "document_type": "affidavit",
                "text": "Alpha evidence. " * 80,
                "case_id": "case_1",
                "version_id": "casever_1",
            },
            {
                "document_id": "doc_2",
                "filename": "order.pdf",
                "document_type": "order",
                "text": "The court ordered production. " * 40,
                "case_id": "case_1",
                "version_id": "casever_1",
            },
        ],
        chunk_size=120,
        chunk_overlap=10,
    )

    assert settings["build_preset"] == "fast_scan"
    assert settings["chunk_size"] == 9999
    assert settings["batch_size"] == 4
    assert settings["llm_concurrency"] == 3
    assert settings["incremental"] is True
    assert {chunk["metadata"]["document_id"] for chunk in chunks} == {"doc_1", "doc_2"}
    assert chunks[0]["metadata"]["case_id"] == "case_1"
    assert chunks[0]["metadata"]["version_id"] == "casever_1"
    assert chunks[0]["metadata"]["filename"] == "affidavit.pdf"
    assert chunks[0]["metadata"]["chunk_index"] == 0


def test_prediction_settings_normalize_presets_and_explicit_controls():
    settings = normalize_prediction_settings(
        {
            "preset": "deep_adversarial_run",
            "agent_count": 80,
            "entity_types_include": ["Judge", "Party"],
            "profile_parallelism": 8,
            "simulation_duration": 96,
            "minutes_per_round": 30,
            "active_agents_per_round": 12,
            "ensemble_runs": 7,
            "scenario_pack": "baseline,plaintiff-favorable",
            "memory_mode": "full",
            "report_mode": "litigation_case",
            "seed": 1234,
        }
    )

    assert settings["preset"] == "deep_adversarial_run"
    assert settings["agent_count"] == 80
    assert settings["entity_types_include"] == ["Judge", "Party"]
    assert settings["profile_parallelism"] == 8
    assert settings["simulation_duration"] == 96
    assert settings["minutes_per_round"] == 30
    assert settings["active_agents_per_round"] == 12
    assert settings["ensemble_runs"] == 7
    assert settings["scenario_pack"] == "baseline,plaintiff-favorable"
    assert settings["memory_mode"] == "full"
    assert settings["report_mode"] == "litigation_case"
    assert settings["seed"] == 1234


def test_case_compare_prediction_versions_uses_latest_snapshots(tmp_path, monkeypatch):
    monkeypatch.setattr(CaseManager, "CASES_DIR", str(tmp_path / "cases"))
    case = CaseManager.create_case("Smith v Jones", "Will the motion succeed?")
    v1 = CaseManager.create_version(case.case_id, "proj_1", prediction_snapshot={"overall_probability": 0.42})
    v2 = CaseManager.create_version(case.case_id, "proj_2", prediction_snapshot={"overall_probability": 0.61})

    comparison = CaseManager.compare_prediction_versions(case.case_id, v1.version_id, v2.version_id)

    assert comparison["case_id"] == case.case_id
    assert comparison["from_version_id"] == v1.version_id
    assert comparison["to_version_id"] == v2.version_id
    assert comparison["changed_predictions"]["overall_probability"]["delta"] == 0.19


def test_simulation_diagnostics_summarize_agent_topics_and_quality(tmp_path, monkeypatch):
    monkeypatch.setattr(SimulationRunner, "RUN_STATE_DIR", str(tmp_path))
    sim_dir = Path(tmp_path) / "sim_1" / "reddit"
    sim_dir.mkdir(parents=True)
    actions = [
        {
            "round": 1,
            "timestamp": "2026-05-18T12:00:00",
            "agent_id": 1,
            "agent_name": "Plaintiff Counsel",
            "action_type": "CREATE_POST",
            "action_args": {"content": "The injunction evidence proves asset dissipation."},
            "success": True,
        },
        {
            "round": 1,
            "timestamp": "2026-05-18T12:01:00",
            "agent_id": 2,
            "agent_name": "Defendant Counsel",
            "action_type": "CREATE_COMMENT",
            "action_args": {"content": "This is generic chatter unrelated to evidence."},
            "success": True,
        },
        {
            "round": 2,
            "timestamp": "2026-05-18T12:02:00",
            "agent_id": 2,
            "agent_name": "Defendant Counsel",
            "action_type": "CREATE_COMMENT",
            "action_args": {"content": "This is generic chatter unrelated to evidence."},
            "success": False,
        },
    ]
    with open(sim_dir / "actions.jsonl", "w", encoding="utf-8") as f:
        for action in actions:
            f.write(json.dumps(action) + "\n")

    diagnostics = SimulationRunner.get_run_diagnostics("sim_1")

    assert diagnostics["simulation_id"] == "sim_1"
    assert diagnostics["summary"]["actions_count"] == 3
    assert diagnostics["summary"]["failed_actions"] == 1
    assert diagnostics["agents"][0]["agent_name"] == "Defendant Counsel"
    assert diagnostics["agents"][0]["repetitive_action_rate"] > 0
    assert diagnostics["topics"]["evidence"]["actions_count"] == 2
    assert diagnostics["quality_report"]["enough_signal"] is True
    assert any(agent["agent_id"] == 2 for agent in diagnostics["off_track_agents"])
