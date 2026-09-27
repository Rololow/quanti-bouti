import math
import random
from datetime import timedelta

import numpy as np
import pytest

from trading_engine.data.events import BarEvent, TradeEvent
from trading_engine.features.correlations import BarReturnAligner, EWMACovariance
from trading_engine.features.feature_engine import FeatureEngine
from trading_engine.features.mean_reversion import SessionVWAP, ma_distance, reversal, zscore
from trading_engine.features.momentum import momentum, resolve_horizon
from trading_engine.features.returns import PriceHistory


def _bar(t0, i, symbol, close, tf="5m", **payload):
    start = t0 + i * timedelta(minutes=5)
    return BarEvent(timestamp=start, end=start + timedelta(minutes=5), symbol=symbol,
                    source="t", timeframe=tf, open=close, high=close, low=close,
                    close=close, payload=payload)


# ------------------------------------------------------------------ returns

def test_price_history_returns():
    h = PriceHistory(maxlen=3)
    assert h.update(100) is None
    assert h.update(110) == pytest.approx(math.log(1.1))
    h.update(121)
    assert h.log_return(2) == pytest.approx(math.log(1.21))
    assert h.log_return(3) is None
    h.update(133.1)  # fenêtre glissante : 100 sort
    assert list(h.prices) == [110, 121, 133.1]


# ------------------------------------------------------------------ momentum

@pytest.mark.parametrize("horizon,expected", [
    ("5m", ("5m", 1)), ("30m", ("5m", 6)), ("1h", ("1h", 1)),
    ("4h", ("1h", 4)), ("20d", ("1d", 20)), ("252d", ("1d", 252)),
])
def test_resolve_horizon(horizon, expected):
    assert resolve_horizon(horizon, ["5m", "1h", "1d"]) == expected


def test_resolve_horizon_without_matching_bar():
    with pytest.raises(ValueError):
        resolve_horizon("7m", ["5m", "1h"])


def test_momentum_is_vol_normalized():
    h = PriceHistory(10)
    for p in (100, 101, 102, 104, 108):
        h.update(p)
    expected = math.log(108 / 101) / (0.01 * math.sqrt(3))
    assert momentum(h, 0.01, 3) == pytest.approx(expected)
    assert momentum(h, None, 3) is None
    assert momentum(h, 0.01, 10) is None


# ------------------------------------------------------------------ mean reversion

def test_zscore_and_ma_distance():
    h = PriceHistory(10)
    for p in (10, 10, 10, 13):
        h.update(p)
    # moyenne 10.75, écart-type population 1.299
    assert zscore(h, 4) == pytest.approx((13 - 10.75) / math.sqrt(1.6875))
    assert ma_distance(h, 4) == pytest.approx(13 / 10.75 - 1)
    assert zscore(h, 5) is None


def test_zscore_constant_prices():
    h = PriceHistory(5)
    for _ in range(5):
        h.update(50)
    assert zscore(h, 5) == 0.0


def test_reversal():
    h = PriceHistory(3)
    h.update(100)
    h.update(105)
    assert reversal(h) == pytest.approx(-math.log(1.05))


def test_session_vwap_resets_each_day(t0):
    v = SessionVWAP()
    v.update(t0, 100, 1)
    v.update(t0, 110, 3)
    assert v.value == pytest.approx(107.5)
    assert v.distance(107.5) == pytest.approx(0.0)
    v.update(t0 + timedelta(days=1), 90, 2)
    assert v.value == 90


# ------------------------------------------------------------------ correlations

def test_aligner_emits_on_next_bucket(t0):
    al = BarReturnAligner()
    assert al.on_bar(_bar(t0, 0, "A", 100)) is None
    assert al.on_bar(_bar(t0, 0, "B", 50)) is None
    # Premier intervalle : pas encore de prix précédent -> rien à émettre.
    assert al.on_bar(_bar(t0, 1, "A", 110)) is None
    al.on_bar(_bar(t0, 1, "B", 55))
    rets = al.on_bar(_bar(t0, 2, "A", 110))
    assert rets == pytest.approx({"A": math.log(1.1), "B": math.log(1.1)})


def test_aligner_missing_bar_means_zero_return_and_late_bar_ignored(t0):
    al = BarReturnAligner()
    al.on_bar(_bar(t0, 0, "A", 100))
    al.on_bar(_bar(t0, 0, "B", 50))
    al.on_bar(_bar(t0, 1, "A", 101))          # B absent sur l'intervalle 1
    rets = al.on_bar(_bar(t0, 2, "A", 102))
    assert rets["B"] == 0.0
    assert al.on_bar(_bar(t0, 1, "B", 60)) is None  # barre en retard


def test_ewma_covariance_recovers_correlation():
    rng = random.Random(0)
    cov = EWMACovariance(lam=0.99)
    rho = 0.8
    for _ in range(5000):
        a = rng.gauss(0, 0.01)
        b = rho * a + math.sqrt(1 - rho**2) * rng.gauss(0, 0.01)
        c = rng.gauss(0, 0.02)
        cov.update({"A": a, "B": b, "C": c})
    assert cov.pair("A", "B") == pytest.approx(rho, abs=0.1)
    assert cov.pair("A", "C") == pytest.approx(0.0, abs=0.15)
    assert cov.volatilities()["C"] == pytest.approx(0.02, rel=0.2)
    symbols, corr = cov.correlation()
    assert symbols == ["A", "B", "C"]
    assert np.allclose(corr, corr.T)
    assert np.allclose(np.diag(corr), 1.0)


def test_ewma_covariance_adds_symbols_dynamically():
    cov = EWMACovariance(lam=0.9)
    cov.update({"A": 0.01})
    cov.update({"A": -0.01, "B": 0.02})
    symbols, matrix = cov.covariance()
    assert symbols == ["A", "B"]
    assert matrix.shape == (2, 2)
    assert matrix[1, 1] > 0


# ------------------------------------------------------------------ feature engine

def test_feature_engine_snapshot(t0):
    fe = FeatureEngine(["5m"], momentum_horizons=["5m", "30m", "1d"], zscore_window=5,
                       correlation_timeframes=["5m"])
    assert fe.momentum_specs["1d"] == ("5m", 288)
    prices = [100, 101, 99, 102, 103, 101, 104, 105]
    for i, p in enumerate(prices):
        fe.on_bar(_bar(t0, i, "SPY", p))
        fe.on_bar(_bar(t0, i, "QQQ", p * 2 + i))
    fe.on_trade(TradeEvent(timestamp=t0 + timedelta(minutes=41), symbol="SPY",
                           source="t", price=105, size=10))

    feats = fe.snapshot("SPY")
    assert feats["ret_5m"] == pytest.approx(math.log(105 / 104))
    assert {"vol_5m", "mom_5m", "mom_30m", "z_5m", "ma_dist_5m", "reversal_5m",
            "vwap_dist"} <= set(feats)
    assert "mom_1d" not in feats  # warm-up : 288 barres nécessaires
    assert "vol_tick" not in feats  # un seul trade : pas encore de rendement tick
    assert fe.correlation_updates("5m") >= 1


def test_feature_engine_ignores_corrected_bars(t0):
    fe = FeatureEngine(["5m"], momentum_horizons=[])
    fe.on_bar(_bar(t0, 0, "SPY", 100))
    fe.on_bar(_bar(t0, 0, "SPY", 120, correction=True))
    assert list(fe.history("SPY", "5m").prices) == [100]
