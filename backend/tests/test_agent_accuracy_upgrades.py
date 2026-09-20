"""Tests for the agent-accuracy / realism upgrades:
W1 graph-derived stance, W2 simulation-grounded probabilities, W4 ensemble aggregation.
"""

from app.services.stance import infer_stance, entity_role
from app.services.forecasting import ForecastSynthesizer, net_to_probabilities
from app.services.ensemble_runner import EnsembleRunner


class FakeEntity:
    def __init__(self, name, attributes=None, related_edges=None, etype="Person"):
        self.name = name
        self.attributes = attributes or {}
        self.related_edges = related_edges or []
        self._etype = etype

    def get_entity_type(self):
        return self._etype


PREDICTION_TARGET = {"question": "Will the downtown congestion pricing plan survive a referendum?"}


def test_infer_stance_opposing_from_edge():
    entity = FakeEntity(
        "Marcus Vale",
        attributes={"title": "Chamber President"},
        related_edges=[
            {"edge_name": "OPPOSES", "fact": "Marcus Vale opposes the congestion pricing plan, warning it hurts retailers", "direction": "outgoing"},
        ],
    )
    result = infer_stance(entity, PREDICTION_TARGET)
    assert result["stance"] == "opposing"
    assert result["sentiment_bias"] < 0
    assert "opposes" in result["rationale"].lower()
    assert entity_role(entity) == "Chamber President"


def test_infer_stance_supportive_and_neutral():
    supportive = FakeEntity(
        "RideForward",
        related_edges=[{"edge_name": "SUPPORTS", "fact": "RideForward praised the decision as a win for commuters", "direction": "outgoing"}],
    )
    assert infer_stance(supportive, PREDICTION_TARGET)["stance"] == "supportive"

    neutral = FakeEntity("Bystander", related_edges=[{"edge_name": "MENTIONED_IN", "fact": "appeared in the record", "direction": "incoming"}])
    assert infer_stance(neutral, PREDICTION_TARGET)["stance"] == "neutral"


def test_net_to_probabilities_moves_with_sentiment():
    outcomes = ["Base case materializes", "Upside case materializes", "Downside case materializes"]
    neutral = net_to_probabilities(0.0, outcomes, "general")
    supportive = net_to_probabilities(0.8, outcomes, "general")
    opposing = net_to_probabilities(-0.8, outcomes, "general")

    # Positive sentiment lifts the upside scenario (index 1) at the expense of the
    # base case (index 0); negative sentiment lifts the downside scenario (index 2).
    assert supportive[1] > neutral[1]
    assert opposing[2] > neutral[2]
    assert supportive[0] < neutral[0]
    assert opposing[0] < neutral[0]
    assert supportive[1] > supportive[2]   # support -> upside beats downside
    assert opposing[2] > opposing[1]       # oppose  -> downside beats upside
    for dist in (neutral, supportive, opposing):
        assert abs(sum(dist) - 1.0) < 1e-6


def test_build_simulation_signal_influence_weighted(monkeypatch):
    config = {
        "simulation_id": "unit",
        "agent_configs": [
            {"agent_id": 1, "stance": "supportive", "sentiment_bias": 1.0, "influence_weight": 3.0},
            {"agent_id": 2, "stance": "opposing", "sentiment_bias": -1.0, "influence_weight": 1.0},
        ],
    }
    fake_actions = (
        [{"agent_id": 1, "action_type": "CREATE_POST", "action_args": {"content": "x"}}] * 3
        + [{"agent_id": 2, "action_type": "CREATE_POST", "action_args": {"content": "y"}}] * 1
    )
    monkeypatch.setattr(ForecastSynthesizer, "_load_simulation_actions", lambda self: fake_actions)

    synth = ForecastSynthesizer("unit", config)
    signal = synth.build_simulation_signal()
    assert signal is not None
    assert signal["total_actions"] == 4
    assert signal["net_sentiment"] > 0  # high-influence supporter dominates
    assert signal["agents_supporting"] == 1 and signal["agents_opposing"] == 1


def test_neutral_agents_signal_driven_by_post_content(monkeypatch):
    # Macro/market graphs rarely yield SUPPORTS/OPPOSES edges, so agents have
    # sentiment_bias 0 (neutral). Their posted views must still drive the signal
    # instead of collapsing the whole simulation to net~0.
    config = {
        "simulation_id": "unit",
        "agent_configs": [
            {"agent_id": 1, "stance": "signal_interpreter", "sentiment_bias": 0.0, "influence_weight": 2.0},
            {"agent_id": 2, "stance": "position_taker", "sentiment_bias": 0.0, "influence_weight": 2.0},
        ],
    }
    actions = (
        [{"agent_id": 1, "action_type": "CREATE_POST", "action_args": {"content": "bullish rally, strong growth, optimistic upside"}}] * 3
        + [{"agent_id": 2, "action_type": "CREATE_POST", "action_args": {"content": "some downside risk"}}] * 1
    )
    monkeypatch.setattr(ForecastSynthesizer, "_load_simulation_actions", lambda self: actions)
    signal = ForecastSynthesizer("unit", config).build_simulation_signal()
    assert signal is not None
    assert signal["net_sentiment"] > 0.1  # bullish posts move it despite neutral stance


def test_safe_haven_assets_invert_risk_sentiment(monkeypatch):
    # Risk-off simulation (negative net): a risk-on asset should fall, a safe-haven
    # asset (gold) should rise.
    bands = [
        {"label": "Down > 4%", "lo": None, "hi": -0.04},
        {"label": "Down 0-4%", "lo": -0.04, "hi": 0.0},
        {"label": "Up 0-4%", "lo": 0.0, "hi": 0.04},
        {"label": "Up > 4%", "lo": 0.04, "hi": None},
    ]
    monkeypatch.setattr(ForecastSynthesizer, "_load_simulation_actions", lambda self: [])

    def numeric_for(target):
        cfg = {"simulation_id": "u", "forecast_mode": "market_economy",
               "forecast_horizon": "1m", "prediction_target": target,
               "agent_configs": []}
        synth = ForecastSynthesizer("u", cfg)
        # bypass calibration file/network: feed bands + vol directly via target
        return synth._numeric_forecast(synth.settings.prediction_target, net=-0.8)

    # Identical (symmetric) base rates so the only difference is the sign inversion;
    # base_rates provided inline so no network/market-data enrichment is triggered.
    flat = [0.25, 0.25, 0.25, 0.25]
    risk_on = numeric_for({"kind": "price", "variable": "S&P 500", "symbol": "^GSPC",
                           "current_value": 5000, "volatility": 0.04, "bands": bands,
                           "base_rates": flat, "outcomes": [b["label"] for b in bands]})
    safe_haven = numeric_for({"kind": "price", "variable": "Gold", "symbol": "GC=F",
                              "current_value": 4000, "volatility": 0.04, "bands": bands,
                              "base_rates": flat, "outcomes": [b["label"] for b in bands]})
    assert risk_on["safe_haven"] is False and safe_haven["safe_haven"] is True
    # Under the same risk-off net, the safe-haven asset rises relative to the risk-on one.
    assert safe_haven["expected_change"] > risk_on["expected_change"]
    assert risk_on["expected_change"] < 0 < safe_haven["expected_change"]


def test_build_simulation_signal_none_without_actions(monkeypatch):
    monkeypatch.setattr(ForecastSynthesizer, "_load_simulation_actions", lambda self: [])
    synth = ForecastSynthesizer("unit", {"simulation_id": "unit", "agent_configs": []})
    assert synth.build_simulation_signal() is None


def test_ensemble_aggregate_mean_and_spread():
    outcomes = ["Base case materializes", "Upside case materializes", "Downside case materializes"]
    per_run_net = [0.4, 0.6, 0.5]
    per_run_probs = [net_to_probabilities(n, outcomes, "general") for n in per_run_net]
    per_run_signals = [{"total_actions": 10, "active_agents": 5, "agents_supporting": 3, "agents_opposing": 1}] * 3

    agg = EnsembleRunner._aggregate(3, outcomes, "general", per_run_net, per_run_probs, per_run_signals)
    assert agg["runs"] == 3 and agg["completed_runs"] == 3
    assert abs(agg["mean_net"] - 0.5) < 1e-6
    assert agg["std_net"] > 0
    assert len(agg["mean_probabilities"]) == 3
    assert abs(sum(agg["mean_probabilities"]) - 1.0) < 1e-3
    assert agg["total_actions"] == 30


def test_partial_ensemble_does_not_overstate_confidence_or_sample_count(monkeypatch):
    synth = ForecastSynthesizer('partial', {})
    monkeypatch.setattr(synth, 'build_simulation_signal', lambda: None)
    monkeypatch.setattr(synth, '_load_ensemble_signal', lambda: {
        'runs': 5, 'completed_runs': 1,
        'mean_probabilities': [0.3, 0.4, 0.3],
        'probability_std': [0, 0, 0], 'mean_net': 0, 'total_actions': 3,
    })
    result = synth.synthesize('')
    assert result['confidence'] == 'low'
    assert '1 completed seeded simulation runs' in result['limitations']
