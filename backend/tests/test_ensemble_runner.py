"""Regression tests for isolated, evidence-bearing ensemble samples."""
import json

import pytest

from app.services.ensemble_runner import EnsembleRunner
from app.services.forecasting import ForecastSynthesizer
from app.services.simulation_runner import SimulationRunner


@pytest.fixture
def harness(monkeypatch, tmp_path):
    from app.config import Config
    monkeypatch.setattr(Config, 'OASIS_SIMULATION_DATA_DIR', str(tmp_path))
    directory = tmp_path / 'sim_test'
    directory.mkdir()
    config = {'simulation_requirement': 'Market outlook', 'economy': {'enabled': True}}
    monkeypatch.setattr(EnsembleRunner, '_load_config', staticmethod(lambda _: config))
    calls = []
    monkeypatch.setattr(SimulationRunner, 'start_simulation', staticmethod(lambda **kw: calls.append(kw)))
    monkeypatch.setattr(SimulationRunner, 'close_simulation_env', staticmethod(lambda *a, **kw: None))
    monkeypatch.setattr(SimulationRunner, 'get_run_state', staticmethod(lambda *a: None))
    def cleanup(*args):
        for path in directory.iterdir():
            path.unlink()
    monkeypatch.setattr(SimulationRunner, 'cleanup_simulation_logs', staticmethod(cleanup))
    return directory, calls


def test_ensemble_excludes_failed_and_empty_samples(monkeypatch, harness):
    directory, calls = harness
    completions = iter([False, True, True])
    monkeypatch.setattr(EnsembleRunner, '_wait_for_completion', staticmethod(lambda *a: next(completions)))
    signals = iter([None, {'net_sentiment': 0.6, 'total_actions': 4}])
    monkeypatch.setattr(ForecastSynthesizer, 'build_simulation_signal', lambda self: next(signals))
    result = EnsembleRunner.run_ensemble('sim_test', runs=3, max_rounds=7)
    assert result['completed_runs'] == 1
    assert result['per_run_net'] == [0.6]
    assert all(call['economy']['enabled'] for call in calls)
    assert all(call['max_rounds'] == 7 for call in calls)
    assert json.loads((directory / 'ensemble_signal.json').read_text()) == result


def test_ensemble_with_no_usable_samples_fails(monkeypatch, harness):
    directory, _ = harness
    monkeypatch.setattr(EnsembleRunner, '_wait_for_completion', staticmethod(lambda *a: False))
    monkeypatch.setattr(ForecastSynthesizer, 'build_simulation_signal', lambda self: None)
    with pytest.raises(RuntimeError, match='usable'):
        EnsembleRunner.run_ensemble('sim_test', runs=2)
    assert not (directory / 'ensemble_signal.json').exists()


def test_ensemble_preserves_final_economic_evidence(monkeypatch, harness):
    directory, _ = harness
    def complete(*args):
        (directory / 'economy.db').write_text('last sample evidence')
        return True
    monkeypatch.setattr(EnsembleRunner, '_wait_for_completion', staticmethod(complete))
    monkeypatch.setattr(ForecastSynthesizer, 'build_simulation_signal', lambda self: {'net_sentiment': 0.2, 'total_actions': 3})
    EnsembleRunner.run_ensemble('sim_test', runs=2)
    assert (directory / 'economy.db').read_text() == 'last sample evidence'
