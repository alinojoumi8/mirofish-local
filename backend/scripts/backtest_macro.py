#!/usr/bin/env python3
"""Backtest the numeric macro forecaster on historical price cases and fit calibration.

For each case we use only data available *before* `as_of` (no look-ahead) to build the
anchor, volatility-scaled bands, and the historical base-rate prior, then check which
band the realized horizon return actually fell into. We grid-fit (sim_weight,
sensitivity) to minimize log-loss and write the result to forecast_calibration.json.

Per-case simulation net sentiment defaults to neutral (0.0) unless provided in the
seed file; wiring a full per-date simulation run is the natural extension. Even with
neutral net, this calibrates how much weight to give the historical base-rate prior.

Usage:
    python scripts/backtest_macro.py [--cases seed_cases.json] [--horizon 1m]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.utils import market_data  # noqa: E402
from app.services import forecast_backtest as bt  # noqa: E402

# Built-in seed set: index/ETF symbols at past as-of dates (1-month horizon).
DEFAULT_CASES = [
    {"symbol": "^GSPC", "kind": "price", "as_of": "2025-01-15", "horizon": "1m"},
    {"symbol": "^GSPC", "kind": "price", "as_of": "2025-03-14", "horizon": "1m"},
    {"symbol": "^GSPC", "kind": "price", "as_of": "2025-05-15", "horizon": "1m"},
    {"symbol": "^GSPC", "kind": "price", "as_of": "2025-07-15", "horizon": "1m"},
    {"symbol": "^GSPC", "kind": "price", "as_of": "2025-09-15", "horizon": "1m"},
    {"symbol": "^IXIC", "kind": "price", "as_of": "2025-02-14", "horizon": "1m"},
    {"symbol": "^IXIC", "kind": "price", "as_of": "2025-06-16", "horizon": "1m"},
    {"symbol": "^IXIC", "kind": "price", "as_of": "2025-10-15", "horizon": "1m"},
]


def _anchor_index(dates, as_of):
    idx = None
    for i, d in enumerate(dates):
        if d <= as_of:
            idx = i
        else:
            break
    return idx


def build_case(spec):
    symbol = spec["symbol"]
    kind = spec.get("kind", "price")
    horizon = spec.get("horizon", "1m")
    horizon_days = market_data.parse_horizon_days(horizon)

    dated = market_data.get_history_dated(symbol, kind=kind, lookback="5y")
    dates = [d for d, _ in dated]
    values = [v for _, v in dated]
    a = _anchor_index(dates, spec["as_of"])
    if a is None or a + horizon_days >= len(values):
        return None, f"insufficient data around {spec['as_of']}"

    anchor = values[a]
    realized = values[a + horizon_days]
    hist = values[: a + 1]  # look-ahead safe

    vol = market_data.realized_volatility(hist, horizon_days)
    bands = market_data.suggest_price_bands(vol)
    base_rates = market_data.base_rate_frequencies(hist, bands, horizon_days)
    realized_idx = bt.classify_realized(realized, bands, kind, anchor)
    if realized_idx is None:
        return None, "could not classify realized outcome"

    case = {
        "label": f"{symbol}@{spec['as_of']}",
        "net": float(spec.get("net", 0.0)),
        "bands": bands,
        "scale": vol if (vol and vol > 0) else 0.05,
        "center": 0.0,
        "base_rates": base_rates,
        "realized_idx": realized_idx,
        "realized_return": round(realized / anchor - 1.0, 4),
    }
    return case, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", help="JSON file with a list of case specs", default=None)
    args = parser.parse_args()

    if not market_data.is_enabled():
        print("Market data is disabled (MARKET_DATA_PROVIDER=none). Cannot backtest.")
        return 1

    specs = DEFAULT_CASES
    if args.cases:
        with open(args.cases, "r", encoding="utf-8") as handle:
            specs = json.load(handle)

    cases = []
    print(f"Building {len(specs)} backtest cases (look-ahead safe)...")
    for spec in specs:
        case, err = build_case(spec)
        if case:
            cases.append(case)
            print(f"  OK  {case['label']}: realized {case['realized_return']:+.2%} -> band {case['realized_idx']}")
        else:
            print(f"  SKIP {spec.get('symbol')}@{spec.get('as_of')}: {err}")

    if not cases:
        print("No usable cases; aborting.")
        return 1

    before = bt.score_cases(cases, sim_weight=0.6, sensitivity=1.0)
    fitted = bt.fit_calibration(cases)
    after = bt.score_cases(cases, fitted["sim_weight"], fitted["sensitivity"])

    print("\n=== Backtest results ===")
    print(f"Cases scored: {len(cases)}")
    print(f"Default params (sim_weight=0.6, sensitivity=1.0): Brier {before['brier']:.4f}, log-loss {before['log_loss']:.4f}")
    print(f"Fitted  params (sim_weight={fitted['sim_weight']}, sensitivity={fitted['sensitivity']}): "
          f"Brier {after['brier']:.4f}, log-loss {after['log_loss']:.4f}")

    print("\nCalibration curve (predicted bucket -> observed frequency):")
    for point in bt.calibration_curve(cases, fitted["sim_weight"], fitted["sensitivity"]):
        print(f"  pred~{point['bucket']:.2f}  observed {point['observed']:.2f}  (n={point['count']})")

    path = bt.write_calibration({"price": {"sim_weight": fitted["sim_weight"], "sensitivity": fitted["sensitivity"]}})
    print(f"\nWrote calibration to {path}")
    print("Note: per-case net sentiment defaulted to neutral unless provided; plug full per-date "
          "simulations into the 'net' field to calibrate the behavioral signal too.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
