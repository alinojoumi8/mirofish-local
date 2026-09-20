from flask import Flask

from app.api import simulation_bp
from app.services.simulation_runner import SimulationRunner


def test_diagnostics_routes_preserve_existing_response_contract(monkeypatch):
    diagnostics = {
        "agents": [{"agent_id": 1}, {"agent_id": 2}],
        "high_impact_agents": [{"agent_id": 1}],
        "off_track_agents": [{"agent_id": 2}],
        "topics": {"housing": 4},
    }
    monkeypatch.setattr(
        SimulationRunner,
        "get_run_diagnostics",
        classmethod(lambda cls, simulation_id: diagnostics),
    )
    app = Flask(__name__)
    app.register_blueprint(simulation_bp, url_prefix="/api/simulation")
    client = app.test_client()

    summary = client.get("/api/simulation/sim_1/diagnostics").get_json()
    agents = client.get("/api/simulation/sim_1/diagnostics/agents?off_track=true").get_json()
    topics = client.get("/api/simulation/sim_1/diagnostics/topics").get_json()
    off_track = client.get("/api/simulation/sim_1/diagnostics/off-track").get_json()

    assert summary == {"success": True, "data": diagnostics}
    assert agents == {"success": True, "data": {"agents": [{"agent_id": 2}], "count": 1}}
    assert topics == {"success": True, "data": {"housing": 4}}
    assert off_track == {"success": True, "data": {"agents": [{"agent_id": 2}], "count": 1}}
