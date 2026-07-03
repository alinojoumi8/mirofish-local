"""Tests for the macro forecast backtest/calibration math."""

import math

from app.services.forecast_backtest import (
    classify_realized, brier_score, log_loss, score_cases, fit_calibration, calibration_curve,
)

PRICE_BANDS = [
    {"label": "Down > 4%", "lo": None, "hi": -0.04},
    {"label": "Down 0-4%", "lo": -0.04, "hi": 0.0},
    {"label": "Up 0-4%", "lo": 0.0, "hi": 0.04},
    {"label": "Up > 4%", "lo": 0.04, "hi": None},
]


def test_classify_realized_price_bands():
    assert classify_realized(95.0, PRICE_BANDS, "price", anchor=100.0) == 0   # -5%
    assert classify_realized(98.0, PRICE_BANDS, "price", anchor=100.0) == 1   # -2%
    assert classify_realized(102.0, PRICE_BANDS, "price", anchor=100.0) == 2  # +2%
    assert classify_realized(110.0, PRICE_BANDS, "price", anchor=100.0) == 3  # +10%


def test_classify_realized_indicator_levels():
    bands = [
        {"label": "Below 2.8", "lo": None, "hi": 2.8},
        {"label": "2.8-3.2", "lo": 2.8, "hi": 3.2},
        {"label": "Above 3.2", "lo": 3.2, "hi": None},
    ]
    assert classify_realized(2.5, bands, "indicator", anchor=None) == 0
    assert classify_realized(3.0, bands, "indicator", anchor=None) == 1
    assert classify_realized(3.5, bands, "indicator", anchor=None) == 2


def test_brier_and_log_loss():
    probs = [0.1, 0.2, 0.4, 0.3]
    assert abs(brier_score([0, 0, 1, 0], 2)) < 1e-9          # perfect
    assert brier_score(probs, 2) > 0
    assert log_loss([0.0, 0.0, 1.0, 0.0], 2) < 1e-6          # confident + correct
    assert log_loss(probs, 0) > log_loss(probs, 2)           # wrong band penalized more


def test_fit_calibration_prefers_prior_when_sim_is_wrong():
    # Realized outcomes match the base-rate prior's mode; net sentiment points the
    # wrong way. Fitting should favour the prior (lower sim_weight) and/or low sensitivity.
    base = [0.05, 0.15, 0.6, 0.2]  # prior concentrated on band 2 (Up 0-4%)
    cases = []
    for _ in range(12):
        cases.append({
            "net": -0.9,              # sim says "down"
            "bands": PRICE_BANDS,
            "scale": 0.04,
            "center": 0.0,
            "base_rates": base,
            "realized_idx": 2,        # but reality matched the prior
        })
    default_ll = score_cases(cases, 0.6, 1.0)["log_loss"]
    fitted = fit_calibration(cases)
    fitted_ll = score_cases(cases, fitted["sim_weight"], fitted["sensitivity"])["log_loss"]
    assert fitted_ll <= default_ll                  # calibration improves (or matches) log-loss
    assert fitted["sim_weight"] <= 0.6              # leans on the prior since sim was wrong
    assert math.isfinite(fitted["log_loss"])


def test_calibration_curve_shape():
    base = [0.1, 0.2, 0.5, 0.2]
    cases = [{"net": 0.3, "bands": PRICE_BANDS, "scale": 0.04, "center": 0.0,
              "base_rates": base, "realized_idx": 2} for _ in range(10)]
    curve = calibration_curve(cases, 0.6, 1.0, n_buckets=10)
    assert curve and all(0.0 <= p["observed"] <= 1.0 and p["count"] > 0 for p in curve)
