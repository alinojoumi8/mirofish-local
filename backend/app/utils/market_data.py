"""Online market & macro data layer for the forecasting engine.

Provides current values (anchors), historical series (for realized volatility and
empirical base-rate band frequencies), and band suggestions. Prices come from the
public Yahoo Finance chart endpoint; macro indicators come from FRED (needs a free
FRED_API_KEY). Everything is read-only public data; the LLM/graph/embeddings remain
fully local. Set MARKET_DATA_PROVIDER=none to disable and stay strictly offline.

Dependency-free beyond `requests` (already a backend dependency).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from ..config import Config
from ..utils.logger import get_logger

logger = get_logger("mirofish.market_data")

_YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
_FRED_OBS = "https://api.stlouisfed.org/fred/series/observations"
_HEADERS = {"User-Agent": "MiroFish-Offline/1.0 (+macro forecasting)"}
_TIMEOUT = 20


class MarketDataError(RuntimeError):
    pass


def is_enabled() -> bool:
    return (Config.MARKET_DATA_PROVIDER or "auto") != "none"


# ----------------------------------------------------------------- caching

def _cache_path(key: str) -> str:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:20]
    os.makedirs(Config.MARKET_DATA_CACHE_DIR, exist_ok=True)
    return os.path.join(Config.MARKET_DATA_CACHE_DIR, f"{digest}.json")


def _cache_get(key: str) -> Optional[Any]:
    path = _cache_path(key)
    try:
        if os.path.exists(path) and (time.time() - os.path.getmtime(path)) < Config.MARKET_DATA_CACHE_TTL:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return None


def _cache_put(key: str, value: Any) -> None:
    try:
        with open(_cache_path(key), "w", encoding="utf-8") as handle:
            json.dump(value, handle)
    except OSError as exc:
        logger.debug("market cache write failed: %s", exc)


# ----------------------------------------------------------------- horizons

def parse_horizon_days(horizon: Optional[str]) -> int:
    """Approximate trading days for a horizon string (default ~1 month)."""
    h = (horizon or "").strip().lower()
    mapping = {"1d": 1, "1w": 5, "2w": 10, "1m": 21, "3m": 63, "6m": 126, "1y": 252}
    if h in mapping:
        return mapping[h]
    if h.endswith("d") and h[:-1].isdigit():
        return max(1, int(h[:-1]))
    if h.endswith("w") and h[:-1].isdigit():
        return max(1, int(h[:-1]) * 5)
    if h.endswith("m") and h[:-1].isdigit():
        return max(1, int(h[:-1]) * 21)
    if h.endswith("y") and h[:-1].isdigit():
        return max(1, int(h[:-1]) * 252)
    return 21


# ----------------------------------------------------------------- providers

def _fetch_yahoo(symbol: str, rng: str = "1y", interval: str = "1d") -> List[Tuple[int, float]]:
    """Return [(timestamp, close), ...] ascending for a Yahoo symbol."""
    key = f"yahoo:{symbol}:{rng}:{interval}"
    cached = _cache_get(key)
    if cached is not None:
        return [(int(t), float(c)) for t, c in cached]
    resp = requests.get(
        _YAHOO_CHART.format(symbol=symbol),
        params={"range": rng, "interval": interval},
        headers=_HEADERS, timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    result = (data.get("chart", {}).get("result") or [None])[0]
    if not result:
        raise MarketDataError(f"No data for symbol {symbol}")
    timestamps = result.get("timestamp") or []
    closes = (((result.get("indicators") or {}).get("quote") or [{}])[0]).get("close") or []
    series = [(int(t), float(c)) for t, c in zip(timestamps, closes) if c is not None]
    if not series:
        raise MarketDataError(f"Empty price series for {symbol}")
    _cache_put(key, series)
    return series


def _fetch_fred(series_id: str, limit: int = 400) -> List[Tuple[str, float]]:
    """Return [(date, value), ...] ascending for a FRED series."""
    if not Config.FRED_API_KEY:
        raise MarketDataError("FRED_API_KEY not set; cannot fetch macro indicator")
    key = f"fred:{series_id}:{limit}"
    cached = _cache_get(key)
    if cached is not None:
        return [(d, float(v)) for d, v in cached]
    resp = requests.get(
        _FRED_OBS,
        params={
            "series_id": series_id, "api_key": Config.FRED_API_KEY, "file_type": "json",
            "sort_order": "desc", "limit": limit,
        },
        headers=_HEADERS, timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    obs = resp.json().get("observations") or []
    series = []
    for o in reversed(obs):  # ascending
        try:
            series.append((o["date"], float(o["value"])))
        except (KeyError, TypeError, ValueError):
            continue
    if not series:
        raise MarketDataError(f"Empty FRED series {series_id}")
    _cache_put(key, series)
    return series


# ----------------------------------------------------------------- public API

def get_history(symbol: str, kind: str = "price", lookback: str = "1y") -> List[float]:
    """Return a list of historical values (ascending) for a symbol."""
    if not is_enabled():
        raise MarketDataError("Market data disabled (MARKET_DATA_PROVIDER=none)")
    if kind == "indicator":
        return [v for _, v in _fetch_fred(symbol)]
    return [c for _, c in _fetch_yahoo(symbol, rng=lookback)]


def get_history_dated(symbol: str, kind: str = "price", lookback: str = "5y") -> List[Tuple[str, float]]:
    """Return [(date_iso, value), ...] ascending. Used by the backtest harness to
    slice data as-of a date (avoiding look-ahead)."""
    if not is_enabled():
        raise MarketDataError("Market data disabled (MARKET_DATA_PROVIDER=none)")
    if kind == "indicator":
        return _fetch_fred(symbol)
    import datetime as _dt
    out = []
    for ts, close in _fetch_yahoo(symbol, rng=lookback):
        out.append((_dt.datetime.utcfromtimestamp(int(ts)).strftime("%Y-%m-%d"), float(close)))
    return out


def get_current_value(symbol: str, kind: str = "price") -> float:
    history = get_history(symbol, kind=kind)
    if not history:
        raise MarketDataError(f"No current value for {symbol}")
    return history[-1]


def realized_volatility(history: List[float], horizon_days: int) -> Optional[float]:
    """Horizon-scaled volatility as a fraction (e.g. 0.04 = 4%) from daily returns."""
    if len(history) < 20:
        return None
    rets = []
    for prev, cur in zip(history[:-1], history[1:]):
        if prev and cur and prev > 0 and cur > 0:
            rets.append(math.log(cur / prev))
    if len(rets) < 10:
        return None
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    daily_sigma = math.sqrt(var)
    return daily_sigma * math.sqrt(max(1, horizon_days))


def suggest_price_bands(volatility: Optional[float]) -> List[Dict[str, Any]]:
    """Return fractional-return bands, scaled by horizon volatility when available."""
    sigma = volatility if (volatility and volatility > 0) else 0.05
    s = round(sigma, 4)
    return [
        {"label": f"Down > {round(s*100,1)}%", "lo": None, "hi": -s},
        {"label": f"Down 0-{round(s*100,1)}%", "lo": -s, "hi": 0.0},
        {"label": f"Up 0-{round(s*100,1)}%", "lo": 0.0, "hi": s},
        {"label": f"Up > {round(s*100,1)}%", "lo": s, "hi": None},
    ]


def base_rate_frequencies(history: List[float], bands: List[Dict[str, Any]], horizon_days: int) -> Optional[List[float]]:
    """Empirical frequency of each (return) band over rolling horizon windows.

    Returns a probability vector aligned with `bands`, or None if insufficient
    history or bands are not return-style (lo/hi as fractions around 0).
    """
    if len(history) <= horizon_days + 5:
        return None
    returns = []
    for i in range(len(history) - horizon_days):
        start, end = history[i], history[i + horizon_days]
        if start and start > 0:
            returns.append(end / start - 1.0)
    if not returns:
        return None
    counts = [0] * len(bands)
    for r in returns:
        for idx, band in enumerate(bands):
            lo = band.get("lo")
            hi = band.get("hi")
            if (lo is None or r >= lo) and (hi is None or r < hi):
                counts[idx] += 1
                break
    total = sum(counts)
    if total == 0:
        return None
    return [c / total for c in counts]


def resolve_target(symbol: str, kind: str = "price", horizon: str = "1m") -> Dict[str, Any]:
    """Resolve current value + suggested numeric bands for the UI / target builder."""
    history = get_history(symbol, kind=kind)
    current = history[-1]
    horizon_days = parse_horizon_days(horizon)
    if kind == "price":
        vol = realized_volatility(history, horizon_days)
        bands = suggest_price_bands(vol)
        base_rates = base_rate_frequencies(history, bands, horizon_days)
        return {
            "symbol": symbol, "kind": kind, "horizon": horizon,
            "current_value": current, "volatility": vol,
            "bands": bands, "base_rates": base_rates,
        }
    # indicator: simple level bands around current value
    step = abs(current) * 0.02 if current else 0.5
    bands = [
        {"label": f"Below {round(current - step, 3)}", "lo": None, "hi": current - step},
        {"label": f"{round(current - step, 3)} to {round(current, 3)}", "lo": current - step, "hi": current},
        {"label": f"{round(current, 3)} to {round(current + step, 3)}", "lo": current, "hi": current + step},
        {"label": f"Above {round(current + step, 3)}", "lo": current + step, "hi": None},
    ]
    return {"symbol": symbol, "kind": kind, "horizon": horizon, "current_value": current, "bands": bands}
