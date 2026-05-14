import sys
from pathlib import Path

from app.services.entity_reader import EntityNode
from app.services.forecasting import ForecastSynthesizer, normalize_forecast_settings
from app.services.graph_memory_updater import AgentActivity
from app.services.oasis_profile_generator import OasisProfileGenerator
from app.services.report_agent import Report, ReportManager, ReportStatus
from app.services.simulation_config_generator import SimulationConfigGenerator
from app.storage.neo4j_storage import Neo4jStorage

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import forecast_runtime  # noqa: E402


def test_forecast_settings_infer_legal_case_and_defaults():
    settings = normalize_forecast_settings(
        {},
        simulation_requirement="Predict the outcome of a Mareva injunction motion in court.",
        entity_types=["Plaintiff", "Defendant", "Judge"],
    )

    assert settings.forecast_mode == "legal_case"
    assert settings.forecast_horizon == "next_procedural_decision"
    assert settings.ensemble_runs == 5
    assert "Moving party substantially succeeds" in settings.prediction_target["outcomes"]


def test_legal_config_defaults_create_domain_events_and_agent_rules():
    generator = SimulationConfigGenerator.__new__(SimulationConfigGenerator)
    settings = {
        "forecast_mode": "legal_case",
        "scenario_pack": "baseline_adverse_favorable",
        "prediction_target": {"question": "Will the motion succeed?"},
    }
    event_config = generator._parse_event_config(
        generator._get_default_event_config(settings),
        settings,
    )
    judge = EntityNode(
        uuid="judge-1",
        name="Ontario Superior Court",
        labels=["Entity", "Judge"],
        summary="Court handling the motion.",
        attributes={},
    )
    agent_cfg = generator._generate_agent_config_by_rule(judge, settings)

    assert event_config.scheduled_events[0]["event_type"] == "procedural_update"
    assert event_config.scenario_pack == "baseline_adverse_favorable"
    assert agent_cfg["stance"] == "neutral_decision_maker"
    assert agent_cfg["influence_weight"] >= 3.0
    assert agent_cfg["posts_per_hour"] < 0.1


def test_oasis_profile_forecast_fields_avoid_generic_personas():
    generator = OasisProfileGenerator.__new__(OasisProfileGenerator)
    generator.forecast_settings = normalize_forecast_settings(
        {"forecast_mode": "legal_case", "prediction_target": {"question": "Will the injunction remain?"}},
    ).to_dict()

    profile = generator._apply_forecast_profile_fields(
        {"persona": "A court actor."},
        "Ontario Superior Court",
        "Judge",
        "Court hearing the injunction motion.",
    )

    assert profile["forecast_role"] == "judge/court decision-maker"
    assert "Will the injunction remain?" in profile["private_beliefs"]
    assert "Never produce generic greetings" in profile["persona"]
    assert profile["risk_tolerance"] == "medium"


def test_forecast_synthesizer_outputs_probability_table_data():
    synthesizer = ForecastSynthesizer(
        "sim-forecast",
        {
            "forecast_mode": "legal_case",
            "forecast_horizon": "next_hearing",
            "prediction_target": {
                "question": "Will the injunction continue?",
                "outcomes": ["Plaintiff wins", "Mixed result", "Defendant wins"],
            },
            "ensemble_runs": 5,
        },
    )

    forecast = synthesizer.synthesize(
        "The evidence is strong but missing records create risk. The court has procedural leverage.",
        [{"fact": "The court scheduled a motion hearing.", "tool_name": "quick_search"}],
    )

    assert forecast["forecast_mode"] == "legal_case"
    assert len(forecast["probabilities"]) == 3
    assert round(sum(row["probability"] for row in forecast["probabilities"]), 2) == 1.0
    assert forecast["evidence_references"][0]["source"] == "quick_search"


def test_report_forecast_serializes_and_reload_preserves_structured_data(tmp_path, monkeypatch):
    monkeypatch.setattr(ReportManager, "REPORTS_DIR", str(tmp_path))
    report = Report(
        report_id="report_forecast",
        simulation_id="sim_1",
        graph_id="graph_1",
        simulation_requirement="test",
        status=ReportStatus.COMPLETED,
        markdown_content="# Report",
        forecast={"forecast_mode": "legal_case", "probabilities": [{"outcome": "A", "probability": 0.6}]},
    )

    ReportManager.save_report(report)
    loaded = ReportManager.get_report("report_forecast")

    assert loaded.forecast["forecast_mode"] == "legal_case"
    assert loaded.to_dict()["forecast"]["probabilities"][0]["probability"] == 0.6


def test_runtime_uses_activity_config_scheduled_events_and_memory(tmp_path, monkeypatch):
    class FakeAgentGraph:
        def get_agent(self, agent_id):
            return f"agent-{agent_id}"

    class FakeEnv:
        agent_graph = FakeAgentGraph()

    config = {
        "time_config": {"agents_per_hour_min": 2, "agents_per_hour_max": 2, "peak_hours": [], "off_peak_hours": []},
        "agent_configs": [
            {"agent_id": 1, "active_hours": [9], "activity_level": 1.0, "posts_per_hour": 2.0, "comments_per_hour": 2.0, "influence_weight": 2.0},
            {"agent_id": 2, "active_hours": [22], "activity_level": 1.0, "posts_per_hour": 2.0, "comments_per_hour": 2.0, "influence_weight": 2.0},
        ],
        "event_config": {
            "scheduled_events": [{"round": 1, "platform": "twitter", "content": "New filing changes settlement leverage."}]
        },
    }
    monkeypatch.setattr(forecast_runtime.random, "uniform", lambda _a, b: b)
    monkeypatch.setattr(forecast_runtime.random, "random", lambda: 0.0)

    active = forecast_runtime.select_active_agents_for_round(FakeEnv(), config, 9, 1)
    due = forecast_runtime.due_scheduled_events(config, 1, 9, "twitter")
    forecast_runtime.record_actions_to_short_memory(
        str(tmp_path),
        "twitter",
        1,
        [{"agent_id": 1, "action_type": "CREATE_POST", "action_args": {"content": "Evidence signal"}}],
    )
    forecast_runtime.update_memory_summaries(str(tmp_path), 5)

    assert active == [(1, "agent-1")]
    assert due[0]["content"] == "New filing changes settlement leverage."
    assert forecast_runtime.load_short_memory(str(tmp_path))["1"][0]["content"] == "Evidence signal"
    assert "CREATE_POST" in forecast_runtime.load_memory_summaries(str(tmp_path))["1"]["summary"]


def test_agent_activity_memory_visibility_and_neo4j_write(monkeypatch):
    activity = AgentActivity(
        platform="twitter",
        agent_id=7,
        agent_name="Counsel",
        action_type="SEARCH_POSTS",
        action_args={"query": "asset dissipation"},
        round_num=3,
        timestamp="2026-05-14T12:00:00",
    )
    record = activity.to_memory_record()

    assert record["visibility"] == "private"
    assert record["importance"] == 0.25

    written = {}

    class FakeEmbedding:
        def embed_batch(self, texts):
            written["texts"] = texts
            return [[0.1] * 768 for _ in texts]

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute_write(self, fn):
            return fn(self)

        def run(self, query, **kwargs):
            written["query"] = query
            written["rows"] = kwargs["rows"]

    class FakeDriver:
        def session(self):
            return FakeSession()

    storage = Neo4jStorage.__new__(Neo4jStorage)
    storage._embedding = FakeEmbedding()
    storage._driver = FakeDriver()
    storage._call_with_retry = lambda func, *args, **kwargs: func(*args, **kwargs)

    ids = storage.add_agent_memories("graph-1", [record])

    assert len(ids) == 1
    assert written["texts"] == ["Counsel: Searched for \"asset dissipation\""]
    assert "embedding: row.embedding" in written["query"]
    assert written["rows"][0]["visibility"] == "private"
    assert written["rows"][0]["source"] == "simulation:twitter"


def test_report_interview_timeout_default_covers_slow_oasis_batch():
    from app.config import Config

    assert Config.REPORT_AGENT_INTERVIEW_TIMEOUT >= 180
