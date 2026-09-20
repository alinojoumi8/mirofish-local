"""Forecast mode helpers and deterministic forecast synthesis."""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..config import Config
from ..utils.logger import get_logger

logger = get_logger("mirofish.forecast")


FORECAST_MODES = {"general", "legal_case", "market_economy"}
REPORT_MODE_BY_FORECAST = {
    "general": "prediction",
    "legal_case": "legal_case",
    "market_economy": "market_prediction",
}

DEFAULT_OUTCOMES = {
    "general": [
        "Base case materializes",
        "Upside case materializes",
        "Downside case materializes",
    ],
    "legal_case": [
        "Moving party substantially succeeds",
        "Mixed or partial result",
        "Opposing party substantially succeeds",
    ],
    "market_economy": [
        "Base case market path",
        "Upside growth or risk-on path",
        "Downside contraction or risk-off path",
    ],
}

DEFAULT_HORIZONS = {
    "general": "medium_term",
    "legal_case": "next_procedural_decision",
    "market_economy": "1m",
}


@dataclass
class ForecastSettings:
    forecast_mode: str = "general"
    forecast_horizon: str = "medium_term"
    prediction_target: Dict[str, Any] = field(default_factory=dict)
    scenario_pack: str = "baseline_adverse_favorable"
    ensemble_runs: int = 5
    memory_mode: str = "practical"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "forecast_mode": self.forecast_mode,
            "forecast_horizon": self.forecast_horizon,
            "prediction_target": self.prediction_target,
            "scenario_pack": self.scenario_pack,
            "ensemble_runs": self.ensemble_runs,
            "memory_mode": self.memory_mode,
        }


def normalize_forecast_mode(mode: Optional[str]) -> str:
    normalized = (mode or "").strip().lower().replace("-", "_")
    if normalized in {"market", "economy", "market_prediction", "market_economic"}:
        return "market_economy"
    if normalized in {"legal", "legal_case_analysis", "case"}:
        return "legal_case"
    return normalized if normalized in FORECAST_MODES else "general"


def infer_forecast_mode(
    simulation_requirement: str = "",
    entity_types: Optional[Iterable[str]] = None,
) -> str:
    text = " ".join([simulation_requirement or "", " ".join(entity_types or [])]).lower()
    legal_terms = [
        "court", "claim", "plaintiff", "defendant", "judge", "injunction",
        "motion", "trial", "settlement", "evidence", "legal", "lawyer",
        "counsel", "fraud", "litigation", "case", "damages",
    ]
    market_terms = [
        "market", "economy", "inflation", "rate", "cpi", "gdp", "stock",
        "revenue", "demand", "pricing", "consumer", "trader", "earnings",
        "macro", "liquidity", "central bank",
    ]
    legal_score = sum(1 for term in legal_terms if term in text)
    market_score = sum(1 for term in market_terms if term in text)
    if legal_score >= max(2, market_score):
        return "legal_case"
    if market_score >= 2:
        return "market_economy"
    return "general"


def _clamp_ensemble_runs(value: Any) -> int:
    try:
        runs = int(value)
    except (TypeError, ValueError):
        runs = 5
    return max(1, min(runs, 20))


def _coerce_float(value: Any) -> Optional[float]:
    try:
        return None if value is None or value == "" else float(value)
    except (TypeError, ValueError):
        return None


def _default_bands(kind: Optional[str], current_value: Optional[float]) -> Optional[List[Dict[str, Any]]]:
    """Default numeric outcome bands for a quantitative target when none are supplied.

    price -> fractional return bands relative to the current value.
    indicator -> absolute level bands around the current value.
    Returns None for qualitative/event targets (fall back to label-only outcomes).
    """
    if kind == "price":
        return [
            {"label": "Down > 5%", "lo": None, "hi": -0.05},
            {"label": "Down 0-5%", "lo": -0.05, "hi": 0.0},
            {"label": "Up 0-5%", "lo": 0.0, "hi": 0.05},
            {"label": "Up > 5%", "lo": 0.05, "hi": None},
        ]
    if kind == "indicator" and current_value is not None:
        c = current_value
        step = abs(c) * 0.02 if c else 0.5
        return [
            {"label": f"Below {round(c - step, 3)}", "lo": None, "hi": c - step},
            {"label": f"{round(c - step, 3)} to {round(c, 3)}", "lo": c - step, "hi": c},
            {"label": f"{round(c, 3)} to {round(c + step, 3)}", "lo": c, "hi": c + step},
            {"label": f"Above {round(c + step, 3)}", "lo": c + step, "hi": None},
        ]
    return None


def _coerce_bands(raw_bands: Any) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(raw_bands, list) or not raw_bands:
        return None
    bands: List[Dict[str, Any]] = []
    for item in raw_bands:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or item.get("outcome") or "").strip() or "Outcome"
        bands.append({"label": label, "lo": _coerce_float(item.get("lo")), "hi": _coerce_float(item.get("hi"))})
    return bands or None


def normalize_prediction_target(
    target: Any,
    mode: str,
    simulation_requirement: str = "",
) -> Dict[str, Any]:
    """Normalize a prediction target into {question, outcomes(labels), bands?(numeric),
    kind?, variable?, symbol?, horizon?, as_of?, current_value?}.

    `outcomes` is always a list of label strings (backward compatible with the
    qualitative modes). `bands` is an aligned list of {label, lo, hi} present only
    for numeric (price/indicator) targets and consumed by the numeric forecaster.
    """
    if isinstance(target, str):
        target = {"question": target}
    if not isinstance(target, dict):
        target = {}
    target = dict(target)

    kind = target.get("kind")
    kind = kind.strip().lower() if isinstance(kind, str) else None
    if kind not in {"event", "price", "indicator"}:
        kind = None
    current_value = _coerce_float(target.get("current_value"))

    # Resolve labels + numeric bands from outcomes/bands (either may be provided).
    bands = _coerce_bands(target.get("bands"))
    labels: Optional[List[str]] = None
    raw_outcomes = target.get("outcomes")
    if isinstance(raw_outcomes, list) and raw_outcomes:
        if all(isinstance(o, dict) for o in raw_outcomes):
            bands = bands or _coerce_bands(raw_outcomes)
        else:
            labels = [str(o).strip() for o in raw_outcomes if str(o).strip()]
    if bands and not labels:
        labels = [b["label"] for b in bands]
    if not labels:
        generated = _default_bands(kind, current_value)
        if generated:
            bands = bands or generated
            labels = [b["label"] for b in bands]
        else:
            labels = list(DEFAULT_OUTCOMES[mode])

    target["outcomes"] = labels[:6]
    if bands:
        target["bands"] = bands[:6]
    if kind:
        target["kind"] = kind
    if current_value is not None:
        target["current_value"] = current_value
    target.setdefault("question", simulation_requirement or "Forecast the most likely future paths.")
    return target


def normalize_forecast_settings(
    raw: Optional[Dict[str, Any]] = None,
    simulation_requirement: str = "",
    entity_types: Optional[Iterable[str]] = None,
) -> ForecastSettings:
    raw = raw or {}
    explicit_mode = raw.get("forecast_mode")
    mode = normalize_forecast_mode(explicit_mode)
    if not explicit_mode:
        mode = infer_forecast_mode(simulation_requirement, entity_types)

    target = normalize_prediction_target(raw.get("prediction_target"), mode, simulation_requirement)

    return ForecastSettings(
        forecast_mode=mode,
        forecast_horizon=(raw.get("forecast_horizon") or DEFAULT_HORIZONS[mode]),
        prediction_target=target,
        scenario_pack=raw.get("scenario_pack") or "baseline_adverse_favorable",
        ensemble_runs=_clamp_ensemble_runs(raw.get("ensemble_runs", 5)),
        memory_mode=(raw.get("memory_mode") or "practical"),
    )


def infer_report_mode_from_forecast(forecast_mode: str) -> str:
    return REPORT_MODE_BY_FORECAST.get(normalize_forecast_mode(forecast_mode), "prediction")


def load_simulation_config(simulation_id: str) -> Dict[str, Any]:
    config_path = os.path.join(Config.OASIS_SIMULATION_DATA_DIR, simulation_id, "simulation_config.json")
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def forecast_settings_from_simulation(simulation_id: str) -> ForecastSettings:
    config = load_simulation_config(simulation_id)
    return normalize_forecast_settings(
        config,
        simulation_requirement=config.get("simulation_requirement", ""),
    )


def domain_guidance_for_forecast(settings: Dict[str, Any]) -> str:
    mode = normalize_forecast_mode(settings.get("forecast_mode"))
    target = settings.get("prediction_target") or {}
    outcomes = target.get("outcomes") or DEFAULT_OUTCOMES[mode]
    base = [
        f"Forecast mode: {mode}",
        f"Forecast horizon: {settings.get('forecast_horizon') or DEFAULT_HORIZONS[mode]}",
        f"Prediction target: {target.get('question', 'Forecast the most likely future paths.')}",
        "Outcome set: " + "; ".join(str(outcome) for outcome in outcomes),
        "Write and act as a probabilistic forecaster. Use scenario evidence, not generic social chatter.",
    ]
    if mode == "legal_case":
        base.extend([
            "Legal behavior rules: separate public posture from private litigation strategy.",
            "Focus on evidence, burden, procedural leverage, settlement incentives, credibility, and missing proof.",
            "Do not claim legal certainty or invent legal authorities not present in the graph.",
        ])
    elif mode == "market_economy":
        base.extend([
            "Market behavior rules: reason from macro variables, demand, liquidity, pricing, incentives, and positioning.",
            "Separate base case, upside case, downside case, and leading indicators.",
        ])
    return "\n".join(f"- {line}" for line in base)


def net_to_probabilities(net: float, outcomes: List[str], mode: str, span: float = 0.5) -> List[float]:
    """Map a net sentiment in [-1, 1] to a normalized probability distribution over
    the outcome set. Shared by single-run synthesis and ensemble aggregation so both
    use identical math."""
    net = max(-1.0, min(1.0, net))
    up = max(0.0, net)
    down = max(0.0, -net)

    if len(outcomes) == 1:
        probs = [1.0]
    elif len(outcomes) == 2:
        probs = [0.5 + span * net, 0.5 - span * net]
    else:
        if mode == "legal_case":
            # [moving party succeeds, mixed/partial, opposing party succeeds]
            fav = 0.34 + span * up
            adv = 0.32 + span * down
            probs = [fav, max(0.1, 1.0 - fav - adv), adv]
        else:
            # [base case, upside case, downside case]
            upside = 0.25 + span * up
            downside = 0.25 + span * down
            probs = [max(0.1, 1.0 - upside - downside), upside, downside]
        if len(outcomes) > 3:
            tail_count = len(outcomes) - 3
            probs = [max(0.05, probs[0] - 0.05 * tail_count), probs[1], probs[2]] + [0.05] * tail_count

    total = sum(max(p, 0.01) for p in probs)
    return [max(p, 0.01) / total for p in probs]


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _band_edges(band: Dict[str, Any], scale: float, center: float) -> Tuple[float, float]:
    """Finite numeric edges for a band, substituting open ends with center ± one scale."""
    lo = band.get("lo")
    hi = band.get("hi")
    e_lo = (hi - scale) if lo is None else lo
    e_hi = (lo + scale) if hi is None else hi
    if e_lo is None:
        e_lo = center - scale
    if e_hi is None:
        e_hi = center + scale
    return float(e_lo), float(e_hi)


def _quantile_axis(bands: List[Dict[str, Any]], probs: List[float], q: float, scale: float, center: float) -> float:
    """Linear-interpolated quantile of the band distribution on the numeric axis."""
    cum = 0.0
    for band, p in zip(bands, probs):
        if p > 0 and cum + p >= q:
            e_lo, e_hi = _band_edges(band, scale, center)
            frac = (q - cum) / p
            return e_lo + frac * (e_hi - e_lo)
        cum += p
    e_lo, e_hi = _band_edges(bands[-1], scale, center)
    return e_hi


def numeric_band_forecast(
    bands: List[Dict[str, Any]],
    net: float,
    scale: float,
    center: float = 0.0,
    base_rates: Optional[List[float]] = None,
    sim_weight: float = 0.6,
    sensitivity: float = 1.0,
) -> Dict[str, Any]:
    """Map a net sentiment to a probability distribution over numeric bands.

    Models the outcome on a numeric axis (returns for prices, levels for indicators)
    as a Normal centered at `center + net*scale*sensitivity` with std `scale`, then
    integrates over each band. Optionally blends with an empirical base-rate prior so
    probabilities stay anchored to history rather than pure simulation sentiment.
    Returns band probabilities, the probability-weighted expected axis value, and a
    P10-P90 interval on the axis.
    """
    scale = scale if (scale and scale > 0) else 0.05
    mean = center + max(-1.0, min(1.0, net)) * scale * sensitivity

    sim = []
    for band in bands:
        lo = band.get("lo")
        hi = band.get("hi")
        lo_z = -math.inf if lo is None else (lo - mean) / scale
        hi_z = math.inf if hi is None else (hi - mean) / scale
        sim.append(max(0.0, _norm_cdf(hi_z) - _norm_cdf(lo_z)))
    s = sum(sim) or 1.0
    sim = [p / s for p in sim]

    if base_rates and len(base_rates) == len(bands) and sum(base_rates) > 0:
        w = max(0.0, min(1.0, sim_weight))
        blended = [w * sim[i] + (1.0 - w) * base_rates[i] for i in range(len(bands))]
        bs = sum(blended) or 1.0
        probs = [p / bs for p in blended]
    else:
        probs = sim

    mids = []
    for band in bands:
        e_lo, e_hi = _band_edges(band, scale, center)
        mids.append((e_lo + e_hi) / 2.0)
    expected_axis = sum(p * m for p, m in zip(probs, mids))

    return {
        "probabilities": probs,
        "expected_axis": expected_axis,
        "mean_axis": mean,
        "p10_axis": _quantile_axis(bands, probs, 0.10, scale, center),
        "p90_axis": _quantile_axis(bands, probs, 0.90, scale, center),
    }


class ForecastSynthesizer:
    """Build a structured probability forecast from report text and simulation metadata."""

    POSITIVE_TERMS = {
        "strong", "strength", "succeeds", "favorable", "support", "credible",
        "upside", "growth", "improve", "catalyst", "leverage",
        # market-directional cues
        "bullish", "rally", "gains", "optimistic", "rebound", "dovish", "easing",
        "cut", "resilient", "expansion", "outperform", "buy",
    }
    NEGATIVE_TERMS = {
        "weak", "weakness", "risk", "fails", "unfavorable", "defect",
        "downside", "contraction", "uncertain", "missing", "challenge",
        # market-directional cues
        "bearish", "selloff", "sell-off", "decline", "plunge", "recession",
        "hawkish", "tightening", "hike", "inflationary", "slowdown", "underperform", "sell",
    }

    def __init__(self, simulation_id: str, simulation_config: Optional[Dict[str, Any]] = None):
        self.simulation_id = simulation_id
        self.simulation_config = simulation_config or load_simulation_config(simulation_id)
        self.settings = normalize_forecast_settings(
            self.simulation_config,
            simulation_requirement=self.simulation_config.get("simulation_requirement", ""),
        )

    def _load_simulation_actions(self) -> List[Dict[str, Any]]:
        """Read recorded agent actions (twitter + reddit) for this simulation."""
        actions: List[Dict[str, Any]] = []
        base = os.path.join(Config.OASIS_SIMULATION_DATA_DIR, self.simulation_id)
        for platform in ("twitter", "reddit"):
            path = os.path.join(base, platform, "actions.jsonl")
            if not os.path.exists(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        # Only real agent actions carry an agent_id (skip round markers).
                        if entry.get("agent_id") is None or not entry.get("action_type"):
                            continue
                        entry["platform"] = platform
                        actions.append(entry)
            except OSError:
                continue
        return actions

    @staticmethod
    def _action_text(action: Dict[str, Any]) -> str:
        args = action.get("action_args") or {}
        return " ".join(
            str(args.get(key, "")) for key in ("content", "quote_content", "original_content")
        ).strip()

    def _text_sentiment(self, text: str) -> float:
        low = (text or "").lower()
        pos = sum(low.count(term) for term in self.POSITIVE_TERMS)
        neg = sum(low.count(term) for term in self.NEGATIVE_TERMS)
        if pos + neg == 0:
            return 0.0
        return max(-1.0, min(1.0, (pos - neg) / (pos + neg)))

    def build_simulation_signal(self) -> Optional[Dict[str, Any]]:
        """Aggregate recorded agent actions into an influence- and stance-weighted
        sentiment signal toward the prediction target.

        This is what makes the headline probabilities *simulation-derived*: they
        reflect which agents actually participated and the real (graph-derived)
        stance they hold, blended with a light reading of what they posted. Returns
        None when no actions were recorded so callers fall back to the text heuristic.
        """
        actions = self._load_simulation_actions()
        if not actions:
            return None

        agent_cfgs = {
            cfg.get("agent_id"): cfg
            for cfg in self.simulation_config.get("agent_configs", [])
        }
        weighted_sum = 0.0
        weight_total = 0.0
        support_w = oppose_w = neutral_w = 0.0
        participants: Dict[Any, Dict[str, Any]] = {}

        for action in actions:
            aid = action.get("agent_id")
            cfg = agent_cfgs.get(aid, {})
            influence = float(cfg.get("influence_weight", 1.0) or 1.0)
            stance_bias = float(cfg.get("sentiment_bias", 0.0) or 0.0)
            text_sentiment = self._text_sentiment(self._action_text(action))
            # If the agent has a graph-derived stance (e.g. SUPPORTS/OPPOSES the target),
            # that dominates and the post nudges intensity. Macro/market graphs usually
            # have no such edges (AFFILIATED_WITH/WORKS_FOR), so the agent's actual posted
            # view must drive its signal — otherwise the whole simulation collapses to neutral.
            if abs(stance_bias) >= 0.1:
                blended = max(-1.0, min(1.0, 0.7 * stance_bias + 0.3 * text_sentiment))
            else:
                blended = text_sentiment
            weight = max(0.1, influence)
            weighted_sum += blended * weight
            weight_total += weight
            if blended > 0.12:
                support_w += weight
            elif blended < -0.12:
                oppose_w += weight
            else:
                neutral_w += weight
            participants[aid] = cfg

        if weight_total <= 0:
            return None

        net = max(-1.0, min(1.0, weighted_sum / weight_total))
        agents_supporting = sum(
            1 for cfg in participants.values()
            if cfg.get("stance") == "supportive" or float(cfg.get("sentiment_bias", 0) or 0) > 0.12
        )
        agents_opposing = sum(
            1 for cfg in participants.values()
            if cfg.get("stance") == "opposing" or float(cfg.get("sentiment_bias", 0) or 0) < -0.12
        )
        return {
            "net_sentiment": round(net, 3),
            "total_actions": len(actions),
            "active_agents": len(participants),
            "support_weight": round(support_w, 2),
            "oppose_weight": round(oppose_w, 2),
            "neutral_weight": round(neutral_w, 2),
            "agents_supporting": agents_supporting,
            "agents_opposing": agents_opposing,
        }

    def _load_ensemble_signal(self) -> Optional[Dict[str, Any]]:
        """Read an aggregated multi-run ensemble result for this simulation, if one
        was produced by EnsembleRunner."""
        path = os.path.join(Config.OASIS_SIMULATION_DATA_DIR, self.simulation_id, "ensemble_signal.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else None
        except (OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def _ensemble_confidence(ensemble: Dict[str, Any]) -> str:
        """Higher agreement across seeded runs (lower spread) → higher confidence."""
        stds = ensemble.get("probability_std") or []
        avg_std = (sum(stds) / len(stds)) if stds else 1.0
        runs = ensemble.get("completed_runs", ensemble.get("runs", 1))
        if runs < 2:
            return "low"
        if runs >= 3 and avg_std < 0.04:
            return "high"
        if runs >= 3 and avg_std < 0.08:
            return "medium_high"
        if avg_std < 0.15:
            return "medium"
        return "low"

    def _resolve_net(self, markdown_content: str, signal: Optional[Dict[str, Any]], ensemble: Optional[Dict[str, Any]]) -> float:
        """Single net-sentiment value, preferring ensemble mean, then the single-run
        signal, then a report-text fallback."""
        if ensemble and ensemble.get("runs", 0) > 1 and ensemble.get("mean_net") is not None:
            return float(ensemble["mean_net"])
        if signal and signal.get("total_actions"):
            return float(signal.get("net_sentiment", 0.0))
        lower = (markdown_content or "").lower()
        pos = sum(lower.count(term) for term in self.POSITIVE_TERMS)
        neg = sum(lower.count(term) for term in self.NEGATIVE_TERMS)
        return max(-0.4, min(0.4, (pos - neg) / max(pos + neg + 8, 1)))

    def _calibration_for(self, kind: str) -> Dict[str, float]:
        """Load calibration params (sim_weight, sensitivity) written by the backtest
        harness; fall back to sensible defaults."""
        defaults = {"sim_weight": 0.6, "sensitivity": 1.0}
        path = os.path.join(Config.UPLOAD_FOLDER, "forecast_calibration.json")
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return defaults
        params = data.get(kind) or data.get("default") or {}
        return {
            "sim_weight": float(params.get("sim_weight", defaults["sim_weight"])),
            "sensitivity": float(params.get("sensitivity", defaults["sensitivity"])),
        }

    # Symbols/keywords whose price moves *against* market risk appetite: when the
    # simulation is risk-off (negative net), these safe-haven assets tend to rise.
    _SAFE_HAVEN_HINTS = (
        "gold", "silver", "xau", "xag", "gc=f", "si=f", "gld", "slv",
        "treasury", "bond", "tlt", "vix", "yen", "jpy", "franc", "chf", "bullion",
    )

    @classmethod
    def _is_safe_haven(cls, target: Dict[str, Any]) -> bool:
        explicit = target.get("safe_haven")
        if isinstance(explicit, bool):
            return explicit
        blob = f"{target.get('variable', '')} {target.get('symbol', '')}".lower()
        return any(hint in blob for hint in cls._SAFE_HAVEN_HINTS)

    def _numeric_forecast(self, target: Dict[str, Any], net: float) -> Optional[Dict[str, Any]]:
        """Produce numeric outputs (point estimate + interval + band probabilities)
        for a quantitative price/indicator target with numeric bands."""
        bands = target.get("bands")
        if not bands:
            return None
        kind = target.get("kind") or "price"
        if kind not in {"price", "indicator"}:
            return None

        current = _coerce_float(target.get("current_value"))
        vol = _coerce_float(target.get("volatility"))
        base_rates = target.get("base_rates") if isinstance(target.get("base_rates"), list) else None
        symbol = target.get("symbol")

        # Enrich missing anchor/volatility/base-rates from the market-data layer.
        if symbol and (current is None or vol is None or base_rates is None):
            try:
                from ..utils import market_data
                if market_data.is_enabled():
                    resolved = market_data.resolve_target(symbol, kind=kind, horizon=self.settings.forecast_horizon)
                    current = current if current is not None else _coerce_float(resolved.get("current_value"))
                    vol = vol if vol is not None else _coerce_float(resolved.get("volatility"))
                    base_rates = base_rates if base_rates is not None else resolved.get("base_rates")
            except Exception as exc:
                logger.warning(
                    "Market-data enrichment failed for symbol=%s kind=%s: %s "
                    "(falling back to un-enriched forecast inputs)",
                    symbol, kind, exc,
                )

        calib = self._calibration_for(kind)

        # Safe-haven assets (gold, silver, treasuries, VIX…) move *against* risk
        # appetite: a risk-off simulation (negative net) should push them up. The
        # net sentiment captures market risk tone, so invert it for these assets.
        safe_haven = self._is_safe_haven(target)
        effective_net = -net if safe_haven else net

        if kind == "price":
            center = 0.0
            scale = vol if (vol and vol > 0) else 0.05
        else:  # indicator: axis is the level
            center = current if current is not None else 0.0
            widths = [b["hi"] - b["lo"] for b in bands if b.get("lo") is not None and b.get("hi") is not None]
            scale = (sorted(widths)[len(widths) // 2] if widths else (abs(center) * 0.02 if center else 0.5)) or 0.5

        res = numeric_band_forecast(
            bands, effective_net, scale, center=center, base_rates=base_rates,
            sim_weight=calib["sim_weight"], sensitivity=calib["sensitivity"],
        )

        band_probabilities = [
            {"label": bands[i]["label"], "lo": bands[i].get("lo"), "hi": bands[i].get("hi"),
             "probability": round(res["probabilities"][i], 3)}
            for i in range(len(bands))
        ]

        if kind == "price" and current is not None:
            expected_change = res["expected_axis"]
            point_estimate = current * (1.0 + expected_change)
            interval = [current * (1.0 + res["p10_axis"]), current * (1.0 + res["p90_axis"])]
        elif kind == "indicator":
            point_estimate = res["expected_axis"]
            interval = [res["p10_axis"], res["p90_axis"]]
            expected_change = (point_estimate - current) if current is not None else None
        else:
            point_estimate = res["expected_axis"]
            interval = [res["p10_axis"], res["p90_axis"]]
            expected_change = res["expected_axis"]

        return {
            "kind": kind,
            "anchor": current,
            "horizon": self.settings.forecast_horizon,
            "volatility": vol,
            "safe_haven": safe_haven,
            "point_estimate": round(point_estimate, 4) if point_estimate is not None else None,
            "interval": [round(interval[0], 4), round(interval[1], 4)],
            "expected_change": round(expected_change, 4) if expected_change is not None else None,
            "band_probabilities": band_probabilities,
            "prior_blend": {"sim_weight": calib["sim_weight"], "has_base_rates": bool(base_rates)},
            "sensitivity": calib["sensitivity"],
        }

    def synthesize(self, markdown_content: str, evidence_cards: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        mode = self.settings.forecast_mode
        target = self.settings.prediction_target
        outcomes = list(target.get("outcomes") or DEFAULT_OUTCOMES[mode])[:5]
        signal = self.build_simulation_signal()
        evidence_refs = self._evidence_references(markdown_content, evidence_cards or [])
        ensemble = self._load_ensemble_signal()

        if (
            ensemble
            and ensemble.get("runs", 0) > 1
            and ensemble.get("mean_probabilities")
            and len(ensemble["mean_probabilities"]) == len(outcomes)
        ):
            # Aggregated distribution across multiple seeded simulation runs.
            stds = ensemble.get("probability_std") or [0.0] * len(outcomes)
            probabilities = [
                {
                    "outcome": outcomes[idx],
                    "probability": round(float(prob), 3),
                    "probability_std": round(float(stds[idx]) if idx < len(stds) else 0.0, 3),
                    "confidence": "medium" if 0.2 <= prob <= 0.65 else "low",
                    "rationale": self._outcome_rationale(outcomes[idx], mode),
                }
                for idx, prob in enumerate(ensemble["mean_probabilities"])
            ]
            confidence = self._ensemble_confidence(ensemble)
            limitations = (
                f"Probabilities are the mean across {ensemble.get('completed_runs', ensemble['runs'])} completed seeded simulation runs "
                f"(net sentiment {ensemble.get('mean_net', 0):+.2f} ± {ensemble.get('std_net', 0):.2f}; "
                f"{ensemble.get('total_actions', 0)} total agent actions). Lower spread across runs means "
                "higher confidence. Treat as scenario estimates; add external base rates for decisions."
            )
        else:
            probabilities = self._probabilities(markdown_content, outcomes, mode, signal)
            confidence = self._confidence(markdown_content, evidence_refs)
            if signal:
                limitations = (
                    f"Probabilities are derived from {signal['total_actions']} simulated agent actions "
                    f"across {signal['active_agents']} agents, weighted by influence and graph-derived stance "
                    f"({signal['agents_supporting']} supporting / {signal['agents_opposing']} opposing). "
                    "Treat as scenario estimates, not certainty; add external base rates for operational decisions."
                )
            else:
                limitations = (
                    "No simulated agent actions were available, so probabilities fall back to report-text "
                    "sentiment near neutral baselines. Treat as low-confidence; re-run the simulation to ground them."
                )

        # Quantitative (price / indicator) targets: produce a numeric point estimate +
        # interval and use the band distribution as the headline probabilities.
        net = self._resolve_net(markdown_content, signal, ensemble)
        numeric = self._numeric_forecast(target, net) if target.get("bands") else None
        if numeric:
            probabilities = [
                {
                    "outcome": bp["label"],
                    "probability": bp["probability"],
                    "confidence": "medium" if 0.2 <= bp["probability"] <= 0.65 else "low",
                    "rationale": "Band probability from simulated participant behavior blended with historical base rates.",
                }
                for bp in numeric["band_probabilities"]
            ]
            pe = numeric.get("point_estimate")
            iv = numeric.get("interval") or [None, None]
            anchor = numeric.get("anchor")
            limitations = (
                f"Numeric {numeric['kind']} forecast: point estimate {pe} "
                f"(P10-P90 {iv[0]} to {iv[1]}), anchored to current value {anchor} and blended with historical "
                f"base rates (sim weight {numeric['prior_blend']['sim_weight']}). Probabilistic, not a guarantee. "
                + limitations
            )

        return {
            "forecast_mode": mode,
            "forecast_horizon": self.settings.forecast_horizon,
            "prediction_target": target,
            "scenario_pack": self.settings.scenario_pack,
            "ensemble_runs": self.settings.ensemble_runs,
            "memory_mode": self.settings.memory_mode,
            "confidence": confidence,
            "probabilities": probabilities,
            "numeric": numeric,
            "simulation_signal": signal,
            "ensemble": ensemble,
            "assumptions": self._assumptions(mode),
            "sensitivity": self._sensitivity(mode),
            "missing_information": self._missing_information(mode, confidence),
            "evidence_references": evidence_refs,
            "limitations": limitations,
        }

    def _probabilities(
        self,
        text: str,
        outcomes: List[str],
        mode: str,
        signal: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        # Primary driver: net sentiment aggregated from actual simulated agent
        # behavior. Only fall back to counting words in the report when the
        # simulation produced no actions.
        if signal and signal.get("total_actions"):
            net = float(signal.get("net_sentiment", 0.0))
            span = 0.5  # let the simulation move probabilities meaningfully
        else:
            lower = text.lower()
            positive = sum(lower.count(term) for term in self.POSITIVE_TERMS)
            negative = sum(lower.count(term) for term in self.NEGATIVE_TERMS)
            net = (positive - negative) / max(positive + negative + 8, 1)
            net = max(-0.4, min(0.4, net))
            span = 0.4

        normalized = net_to_probabilities(net, outcomes, mode, span)
        return [
            {
                "outcome": outcomes[idx],
                "probability": round(prob, 3),
                "confidence": "medium" if 0.2 <= prob <= 0.65 else "low",
                "rationale": self._outcome_rationale(outcomes[idx], mode),
            }
            for idx, prob in enumerate(normalized)
        ]

    def _confidence(self, text: str, evidence_refs: List[Dict[str, Any]]) -> str:
        if len(evidence_refs) >= 6 and len(text) > 2500:
            return "medium_high"
        if len(evidence_refs) >= 3 and len(text) > 1200:
            return "medium"
        return "low"

    def _evidence_references(self, text: str, evidence_cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        refs = []
        seen = set()
        for card in evidence_cards:
            fact = (card.get("fact") or card.get("quote") or "").strip()
            if fact and fact.lower() not in seen:
                refs.append({
                    "fact": fact[:300],
                    "source": card.get("tool_name") or "report_agent",
                    "section": card.get("section_title"),
                })
                seen.add(fact.lower())
            if len(refs) >= 8:
                return refs

        for sentence in re.split(r"(?<=[.!?])\s+", text):
            cleaned = re.sub(r"^[>#\s-]+", "", sentence).strip()
            if len(cleaned) < 60:
                continue
            key = cleaned.lower()[:180]
            if key in seen:
                continue
            refs.append({"fact": cleaned[:300], "source": "report_text", "section": None})
            seen.add(key)
            if len(refs) >= 8:
                break
        return refs

    def _outcome_rationale(self, outcome: str, mode: str) -> str:
        if mode == "legal_case":
            return "Weighted from simulated legal posture, evidence strength, procedural leverage, and settlement pressure."
        if mode == "market_economy":
            return "Weighted from simulated demand, macro conditions, liquidity, competitive response, and timing signals."
        return "Weighted from simulated agent behavior, event trajectory, and retrieved graph evidence."

    def _assumptions(self, mode: str) -> List[str]:
        if mode == "legal_case":
            return [
                "The uploaded record captures the main case facts and procedural posture.",
                "Agents represent strategic incentives, not actual legal advice.",
                "No external case-law or court-statistics base rate is applied unless explicitly present in the graph.",
            ]
        if mode == "market_economy":
            return [
                "The uploaded record captures the relevant market actors and current state.",
                "No live market or macro feed is applied unless an adapter is enabled.",
                "Scenario shocks are simulation assumptions, not real-time news.",
            ]
        return [
            "The uploaded record is the main evidence base.",
            "Forecasts depend on simulation assumptions and agent configuration quality.",
        ]

    def _sensitivity(self, mode: str) -> List[Dict[str, str]]:
        if mode == "legal_case":
            return [
                {"factor": "Missing evidence", "effect": "Could move probabilities sharply toward either party."},
                {"factor": "Procedural ruling", "effect": "Can change leverage before merits are resolved."},
                {"factor": "Settlement incentives", "effect": "Can dominate strict merits when cost or asset risk is high."},
            ]
        if mode == "market_economy":
            return [
                {"factor": "Rate and liquidity path", "effect": "Changes risk appetite and valuation pressure."},
                {"factor": "Demand signal quality", "effect": "Distinguishes durable trend from short-lived noise."},
                {"factor": "Policy or regulatory shock", "effect": "Can override actor-level expectations."},
            ]
        return [
            {"factor": "Evidence completeness", "effect": "More complete graph context improves forecast stability."},
            {"factor": "Agent incentives", "effect": "Incorrect incentives can distort simulated behavior."},
        ]

    def _missing_information(self, mode: str, confidence: str) -> List[str]:
        if mode == "legal_case":
            missing = ["Comparable case outcomes/base rates", "Current court deadlines or live docket changes"]
        elif mode == "market_economy":
            missing = ["Live price and volume data", "Current macro calendar/news feed"]
        else:
            missing = ["External base rates", "Recent real-world updates"]
        if confidence == "low":
            missing.insert(0, "More high-quality evidence and agent interviews")
        return missing


def forecast_to_markdown(forecast: Dict[str, Any]) -> str:
    rows = forecast.get("probabilities") or []
    lines = [
        "## Forecast Probability Summary",
        "",
        f"Mode: `{forecast.get('forecast_mode', 'general')}`",
        f"Horizon: `{forecast.get('forecast_horizon', 'medium_term')}`",
        f"Confidence: `{forecast.get('confidence', 'low')}`",
    ]
    numeric = forecast.get("numeric")
    if numeric and numeric.get("point_estimate") is not None:
        iv = numeric.get("interval") or [None, None]
        anchor = numeric.get("anchor")
        lines.extend([
            "",
            f"**Point estimate ({numeric.get('kind')}):** `{numeric['point_estimate']}` "
            f"(P10-P90 `{iv[0]}` to `{iv[1]}`)"
            + (f", from current `{anchor}`" if anchor is not None else ""),
        ])
    lines.extend([
        "",
        "| Outcome | Probability | Rationale |",
        "| --- | ---: | --- |",
    ])
    for row in rows:
        pct = f"{row.get('probability', 0) * 100:.1f}%"
        lines.append(f"| {row.get('outcome', '')} | {pct} | {row.get('rationale', '')} |")
    ensemble = forecast.get("ensemble")
    signal = forecast.get("simulation_signal")
    if ensemble and ensemble.get("runs", 0) > 1:
        lines.extend([
            "",
            (
                f"Aggregated across {ensemble['runs']} seeded simulation runs "
                f"({ensemble.get('total_actions', 0)} total agent actions; "
                f"net sentiment {ensemble.get('mean_net', 0):+.2f} ± {ensemble.get('std_net', 0):.2f})."
            ),
        ])
    elif signal:
        lines.extend([
            "",
            (
                f"Grounded in {signal.get('total_actions', 0)} simulated agent actions "
                f"({signal.get('active_agents', 0)} agents; {signal.get('agents_supporting', 0)} supporting, "
                f"{signal.get('agents_opposing', 0)} opposing; net sentiment {signal.get('net_sentiment', 0):+.2f})."
            ),
        ])
    if forecast.get("limitations"):
        lines.extend(["", f"Note: {forecast['limitations']}"])
    return "\n".join(lines).strip()
