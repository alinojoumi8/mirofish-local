import pytest

from app.config import Config
from app.services import economy as economy_module
from app.services import report_agent as report_agent_module
from app.services.report_agent import ReportAgent, ReportOutline, ReportSection


class GraphToolsStub:
    def get_simulation_context(self, **_kwargs):
        return {
            "graph_statistics": {
                "total_nodes": 2,
                "total_edges": 1,
                "entity_types": {"person": 2},
            },
            "total_entities": 2,
            "related_facts": [],
        }


class RecordingLLM:
    def __init__(self):
        self.outline_messages = []
        self.section_messages = []

    def chat_json(self, messages, **_kwargs):
        self.outline_messages.append(messages)
        return {
            "title": "Test report",
            "summary": "Grounded summary",
            "sections": [{"title": "Economic outcomes", "description": "Settled facts"}],
        }

    def chat(self, messages, **_kwargs):
        self.section_messages.append(messages)
        raise RuntimeError("stop after prompt capture")


def make_agent(monkeypatch, config):
    monkeypatch.setattr(report_agent_module, "load_simulation_config", lambda _id: config)
    llm = RecordingLLM()
    return ReportAgent(
        graph_id="graph-1",
        simulation_id="sim-report-economy",
        simulation_requirement="Explain the simulated economy",
        llm_client=llm,
        graph_tools=GraphToolsStub(),
    ), llm


def test_economy_enabled_report_prefetches_once_for_outline_and_sections(monkeypatch):
    calls = []
    snapshot = {
        "summary": {
            "settings": {"enabled": True, "currency": "USD"},
            "latest_metrics": {
                "tick": 1,
                "agents_count": 2,
                "total_supply_cents": 200000,
                "trade_volume_cents": 5000,
                "gini": 0.05,
            },
            "ticks": {"completed": 1, "running": 0, "failed": 0},
            "latest_tick": {"tick": 1, "status": "completed", "error": None},
            "rejected_intents": 0,
        },
        "agents": [{"name": "Employer", "balance_cents": 95000}],
        "jobs": [],
        "trades": [],
        "events": [{"tick": 1, "event_type": "transfer", "actor_name": "Employer", "amount_cents": 5000}],
        "ledger": [],
    }

    def evidence(_simulation_id, *, limit):
        calls.append(limit)
        return snapshot

    monkeypatch.setattr(economy_module, "economy_evidence", evidence)
    agent, llm = make_agent(monkeypatch, {"economy": {"enabled": True}})

    outline = agent.plan_outline()
    assert calls == [25]
    assert "Auto-prefetched Settled Economic Evidence" in llm.outline_messages[0][1]["content"]
    assert "200000 USD cents" in llm.outline_messages[0][1]["content"]

    section = ReportSection(title="Economic outcomes", description="Settled facts")
    with pytest.raises(RuntimeError, match="prompt capture"):
        agent._generate_section_react(
            section,
            ReportOutline(title=outline.title, summary=outline.summary, sections=[section]),
            [],
        )

    assert calls == [25]
    assert "Auto-prefetched Settled Economic Evidence" in llm.section_messages[0][1]["content"]
    assert "economy_evidence" in agent._build_used_evidence_ledger([])
    assert section.evidence_cards
    assert all(card["tool_name"] == "economy_evidence" for card in section.evidence_cards)


def test_missing_economy_database_generates_explicit_report_warning(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "OASIS_SIMULATION_DATA_DIR", str(tmp_path))
    agent, _llm = make_agent(monkeypatch, {"economy": {"enabled": True}})

    context = agent._prefetch_economy_evidence()

    assert "Economic Evidence Warning" in context
    assert "settled economic evidence is unavailable" in context
    assert "Do not infer or fabricate" in context
    assert "economy_evidence warning" in agent._build_used_evidence_ledger([])
