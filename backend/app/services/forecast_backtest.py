"""Backtesting & calibration for the numeric macro forecaster.

Scores forecast distributions against realized outcomes (Brier score, log-loss,
calibration curve) and fits the two free parameters of the numeric mapping
(`sim_weight`, the blend between simulation sentiment and the historical base-rate
prior; and `sensitivity`, how far net sentiment shifts the distribution mean) to
minimize log-loss on a set of historical cases. The fitted params are written to
`uploads/forecast_calibration.json`, which `ForecastSynthesizer` loads at runtime.

The scoring/fitting math here is pure and unit-tested. The CLI runner
(`scripts/backtest_macro.py`) assembles cases from real market data (look-ahead
safe) and calls `fit_calibration` + `write_calibration`.
"""

from __future__ import annotations

import json
import math
import os
import statistics
from typing import Any, Dict, List, Optional

from ..config import Config
from .forecasting import numeric_band_forecast

_GRID_SIM_WEIGHT = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
_GRID_SENSITIVITY = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]


def classify_realized(realized_value: float, bands: List[Dict[str, Any]], kind: str, anchor: Optional[float]) -> Optional[int]:
    """Index of the band the realized outcome fell into.

    price -> realized return relative to `anchor`; indicator -> absolute level.
    """
    if kind == "price":
        if not anchor:
            return None
        axis = realized_value / anchor - 1.0
    else:
        axis = realized_value
    for idx, band in enumerate(bands):
        lo = band.get("lo")
        hi = band.get("hi")
        if (lo is None or axis >= lo) and (hi is None or axis < hi):
            return idx
    # Above every finite band -> last; below -> first.
    return len(bands) - 1 if axis >= 0 else 0


def brier_score(probs: List[float], realized_idx: int) -> float:
    return sum((p - (1.0 if i == realized_idx else 0.0)) ** 2 for i, p in enumerate(probs))


def log_loss(probs: List[float], realized_idx: int, eps: float = 1e-9) -> float:
    p = max(eps, min(1.0 - eps, probs[realized_idx]))
    return -math.log(p)


def _case_probs(case: Dict[str, Any], sim_weight: float, sensitivity: float) -> List[float]:
    res = numeric_band_forecast(
        case["bands"], case["net"], case["scale"], center=case.get("center", 0.0),
        base_rates=case.get("base_rates"), sim_weight=sim_weight, sensitivity=sensitivity,
    )
    return res["probabilities"]


def score_cases(cases: List[Dict[str, Any]], sim_weight: float, sensitivity: float) -> Dict[str, float]:
    """Mean Brier and log-loss over cases for a given parameter pair.

    Each case: {net, bands, scale, center?, base_rates?, realized_idx}.
    """
    briers, lls = [], []
    for case in cases:
        probs = _case_probs(case, sim_weight, sensitivity)
        idx = case["realized_idx"]
        briers.append(brier_score(probs, idx))
        lls.append(log_loss(probs, idx))
    if not cases:
        return {"brier": float("nan"), "log_loss": float("nan"), "n": 0}
    return {"brier": statistics.fmean(briers), "log_loss": statistics.fmean(lls), "n": len(cases)}


def fit_calibration(cases: List[Dict[str, Any]]) -> Dict[str, float]:
    """Grid-search (sim_weight, sensitivity) minimizing mean log-loss."""
    best: Optional[Dict[str, float]] = None
    for sw in _GRID_SIM_WEIGHT:
        for sens in _GRID_SENSITIVITY:
            s = score_cases(cases, sw, sens)
            if best is None or s["log_loss"] < best["log_loss"]:
                best = {"sim_weight": sw, "sensitivity": sens, "log_loss": s["log_loss"], "brier": s["brier"]}
    return best or {"sim_weight": 0.6, "sensitivity": 1.0, "log_loss": float("nan"), "brier": float("nan")}


def calibration_curve(cases: List[Dict[str, Any]], sim_weight: float, sensitivity: float, n_buckets: int = 10) -> List[Dict[str, float]]:
    """Reliability curve: for predicted-probability buckets, the observed frequency
    that the predicted band was the realized one."""
    buckets = [[] for _ in range(n_buckets)]
    for case in cases:
        probs = _case_probs(case, sim_weight, sensitivity)
        idx = case["realized_idx"]
        for band_idx, p in enumerate(probs):
            b = min(n_buckets - 1, int(p * n_buckets))
            buckets[b].append(1.0 if band_idx == idx else 0.0)
    curve = []
    for b, hits in enumerate(buckets):
        if hits:
            curve.append({
                "bucket": round((b + 0.5) / n_buckets, 3),
                "observed": round(statistics.fmean(hits), 3),
                "count": len(hits),
            })
    return curve


def calibration_path() -> str:
    return os.path.join(Config.UPLOAD_FOLDER, "forecast_calibration.json")


def write_calibration(params_by_kind: Dict[str, Dict[str, float]]) -> str:
    """Merge and persist calibration params keyed by target kind (e.g. 'price')."""
    path = calibration_path()
    existing: Dict[str, Any] = {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            existing = json.load(handle)
    except (OSError, json.JSONDecodeError):
        existing = {}
    existing.update(params_by_kind)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(existing, handle, indent=2)
    return path
