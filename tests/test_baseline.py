import math
from datetime import timedelta

import pytest

from trading_engine.allocation.baseline import BaselineAllocator
from trading_engine.timeutils import periods_per_year
from trading_engine.data.events import BarEvent
from trading_engine.features.feature_engine import FeatureEngine


def test_periods_per_year():
    assert periods_per_year("1d") == 252
    assert periods_per_year("1h") == pytest.approx(252 * 6.5)
    assert periods_per_year("5m") == pytest.approx(252 * 78)


def _feed(fe, t0, paths):
    """paths: symbol -> liste de closes 1h."""
    n = len(next(iter(paths.values())))
    for i in range(n):
        start = t0 + i * timedelta(hours=1)
        for sym, closes in paths.items():
            fe.on_bar(BarEvent(timestamp=start, end=start + timedelta(hours=1), symbol=sym,
                               source="t", timeframe="1h", open=closes[i], high=closes[i],
                               low=closes[i], close=closes[i]))


def _path(start, drift, wiggle, n=40):
    # L'oscillation (-1)^i s'annule sur 4 barres : mom_4h ne dépend que du drift.
    return [start * math.exp(drift * i + wiggle * (-1) ** i) for i in range(n)]


def test_baseline_long_only_risk_scaled(t0):
    fe = FeatureEngine(["1h"], momentum_horizons=["4h"], correlation_timeframes=["1h"])
    _feed(fe, t0, {
        "UP_CALM": _path(100, 0.002, 0.001),
        "UP_WILD": _path(100, 0.002, 0.01),
        "DOWN": _path(100, -0.003, 0.002),
    })
    alloc = BaselineAllocator(momentum_horizon="4h", target_vol=10.0, max_weight=1.0).allocate(
        fe, ["UP_CALM", "UP_WILD", "DOWN", "UNKNOWN"])

    w = alloc.weights
    assert w["DOWN"] == 0.0          # momentum négatif : long-only -> 0
    assert w["UNKNOWN"] == 0.0       # pas de données
    assert w["UP_CALM"] > 0 and w["UP_WILD"] >= 0
    assert sum(w.values()) == pytest.approx(1.0)
    assert alloc.vol_scale == 1.0
    assert set(alloc.attribution["UP_CALM"]) >= {"momentum", "vol", "vol_scale", "weight"}


def test_baseline_scales_down_to_target_vol(t0):
    fe = FeatureEngine(["1h"], momentum_horizons=["4h"], correlation_timeframes=["1h"])
    _feed(fe, t0, {"A": _path(100, 0.004, 0.02), "B": _path(50, 0.003, 0.02)})
    alloc = BaselineAllocator(momentum_horizon="4h", target_vol=0.05, max_weight=1.0).allocate(fe, ["A", "B"])
    assert alloc.vol_scale < 1.0
    assert alloc.portfolio_vol == pytest.approx(0.05, rel=1e-6)
    assert sum(alloc.weights.values()) < 1.0


def test_baseline_respects_max_weight(t0):
    fe = FeatureEngine(["1h"], momentum_horizons=["4h"], correlation_timeframes=["1h"])
    _feed(fe, t0, {"A": _path(100, 0.004, 0.001), "B": _path(100, -0.004, 0.001)})
    alloc = BaselineAllocator(momentum_horizon="4h", target_vol=10.0, max_weight=0.3).allocate(fe, ["A", "B"])
    assert alloc.weights["A"] == pytest.approx(0.3)
