"""Forecast mode helpers and deterministic forecast synthesis."""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from ..config import Config


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

    target = raw.get("prediction_target") or {}
    if isinstance(target, str):
        target = {"question": target}
    if not isinstance(target, dict):
        target = {}

    outcomes = target.get("outcomes")
    if not isinstance(outcomes, list) or not outcomes:
        target["outcomes"] = DEFAULT_OUTCOMES[mode]
    target.setdefault("question", simulation_requirement or "Forecast the most likely future paths.")

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


class ForecastSynthesizer:
    """Build a structured probability forecast from report text and simulation metadata."""

    POSITIVE_TERMS = {
        "strong", "strength", "succeeds", "favorable", "support", "credible",
        "upside", "growth", "improve", "catalyst", "leverage",
    }
    NEGATIVE_TERMS = {
        "weak", "weakness", "risk", "fails", "unfavorable", "defect",
        "downside", "contraction", "uncertain", "missing", "challenge",
    }

    def __init__(self, simulation_id: str, simulation_config: Optional[Dict[str, Any]] = None):
        self.simulation_id = simulation_id
        self.simulation_config = simulation_config or load_simulation_config(simulation_id)
        self.settings = normalize_forecast_settings(
            self.simulation_config,
            simulation_requirement=self.simulation_config.get("simulation_requirement", ""),
        )

    def synthesize(self, markdown_content: str, evidence_cards: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        mode = self.settings.forecast_mode
        target = self.settings.prediction_target
        outcomes = list(target.get("outcomes") or DEFAULT_OUTCOMES[mode])[:5]
        probabilities = self._probabilities(markdown_content, outcomes, mode)
        evidence_refs = self._evidence_references(markdown_content, evidence_cards or [])

        confidence = self._confidence(markdown_content, evidence_refs)
        return {
            "forecast_mode": mode,
            "forecast_horizon": self.settings.forecast_horizon,
            "prediction_target": target,
            "scenario_pack": self.settings.scenario_pack,
            "ensemble_runs": self.settings.ensemble_runs,
            "memory_mode": self.settings.memory_mode,
            "confidence": confidence,
            "probabilities": probabilities,
            "assumptions": self._assumptions(mode),
            "sensitivity": self._sensitivity(mode),
            "missing_information": self._missing_information(mode, confidence),
            "evidence_references": evidence_refs,
            "limitations": (
                "Probabilities are simulation-derived estimates, not certainty. "
                "External base rates and fresh real-world data should be added for operational decisions."
            ),
        }

    def _probabilities(self, text: str, outcomes: List[str], mode: str) -> List[Dict[str, Any]]:
        lower = text.lower()
        positive = sum(lower.count(term) for term in self.POSITIVE_TERMS)
        negative = sum(lower.count(term) for term in self.NEGATIVE_TERMS)
        balance = max(-0.2, min(0.2, (positive - negative) / max(positive + negative + 8, 1)))

        if len(outcomes) == 1:
            probs = [1.0]
        elif len(outcomes) == 2:
            probs = [0.5 + balance, 0.5 - balance]
        else:
            if mode == "legal_case":
                probs = [0.34 + balance, 0.34, 0.32 - balance]
            elif mode == "market_economy":
                probs = [0.50, 0.25 + balance, 0.25 - balance]
            else:
                probs = [0.50, 0.25 + balance, 0.25 - balance]
            if len(outcomes) > 3:
                tail_count = len(outcomes) - 3
                tail = 0.05 * tail_count
                probs = [max(0.05, probs[0] - tail), probs[1], probs[2]] + [0.05] * tail_count

        total = sum(max(p, 0.01) for p in probs)
        normalized = [max(p, 0.01) / total for p in probs]
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
        "",
        "| Outcome | Probability | Rationale |",
        "| --- | ---: | --- |",
    ]
    for row in rows:
        pct = f"{row.get('probability', 0) * 100:.1f}%"
        lines.append(f"| {row.get('outcome', '')} | {pct} | {row.get('rationale', '')} |")
    if forecast.get("limitations"):
        lines.extend(["", f"Note: {forecast['limitations']}"])
    return "\n".join(lines).strip()
