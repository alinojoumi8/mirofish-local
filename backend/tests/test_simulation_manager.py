from app.services.simulation_config_generator import SimulationConfigGenerator, SimulationParameters
from app.services.simulation_manager import SimulationManager, SimulationStatus


class FakeGraphStorage:
    def get_all_nodes(self, graph_id):
        return [
            {
                "uuid": "company-1",
                "name": "Acme Robotics",
                "labels": ["Entity", "Company"],
                "summary": "Delivery robot company running a Toronto pilot.",
                "attributes": {},
            }
        ]

    def get_all_edges(self, graph_id):
        return []


def test_prepare_simulation_accepts_prediction_settings_without_profile_generator_error(tmp_path, monkeypatch):
    monkeypatch.setattr(SimulationManager, "SIMULATION_DATA_DIR", str(tmp_path))

    def fake_generate_config(self, **kwargs):
        return SimulationParameters(
            simulation_id=kwargs["simulation_id"],
            project_id=kwargs["project_id"],
            graph_id=kwargs["graph_id"],
            simulation_requirement=kwargs["simulation_requirement"],
        )

    monkeypatch.setattr(SimulationConfigGenerator, "generate_config", fake_generate_config)

    manager = SimulationManager()
    state = manager.create_simulation(
        project_id="project-1",
        graph_id="graph-1",
        prediction_settings={"agent_count": 1, "profile_parallelism": 1},
    )

    prepared = manager.prepare_simulation(
        state.simulation_id,
        simulation_requirement="Predict reaction to delivery robots.",
        document_text="Acme Robotics is piloting delivery robots in Toronto.",
        use_llm_for_profiles=False,
        storage=FakeGraphStorage(),
        prediction_settings={"agent_count": 1, "profile_parallelism": 1},
    )

    assert prepared.status == SimulationStatus.READY
    assert prepared.profiles_count == 1
    assert (tmp_path / state.simulation_id / "reddit_profiles.json").exists()
    assert (tmp_path / state.simulation_id / "simulation_config.json").exists()
