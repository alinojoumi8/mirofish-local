from app.services.graph_tools import GraphToolsService
from app.services.simulation_runner import SimulationRunner


class FailingLLM:
    def chat_json(self, *args, **kwargs):
        raise AssertionError("agent selection should be skipped when environment is closed")


def test_interviews_skip_quickly_when_simulation_environment_closed(monkeypatch):
    service = GraphToolsService(storage=None, llm_client=FailingLLM())
    monkeypatch.setattr(
        service,
        "_load_agent_profiles",
        lambda simulation_id: [{"realname": "Reza Samvat", "profession": "Plaintiff", "bio": "Primary plaintiff"}],
    )
    monkeypatch.setattr(SimulationRunner, "check_env_alive", classmethod(lambda cls, simulation_id: False))

    result = service.interview_agents(
        simulation_id="sim_closed",
        interview_requirement="fraud case outcome",
        max_agents=6,
    )

    assert result.total_agents == 1
    assert result.interviewed_count == 0
    assert "skipped" in result.summary.lower()
