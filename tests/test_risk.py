import math
from datetime import timedelta

import numpy as np
import pytest

from trading_engine.data.events import BarEvent
from trading_engine.features.feature_engine import FeatureEngine
from trading_engine.portfolio.portfolio import Portfolio
from trading_engine.portfolio.positions import Position
from trading_engine.risk.covariance import annualized_covariance, nearest_psd, shrink_to_diagonal
from trading_engine.risk.limits import RiskLimits
from trading_engine.risk.portfolio_risk import (
    DrawdownTracker,
    diversification_ratio,
    effective_n,
    herfindahl,
    portfolio_vol,
    risk_contributions,
)
from trading_engine.risk.risk_engine import RiskEngine

COV = np.array([[0.04, 0.006, 0.0], [0.006, 0.09, 0.0], [0.0, 0.0, 0.01]])


def test_risk_contributions_sum_to_portfolio_vol():
    w = np.array([0.5, 0.3, 0.2])
    d = risk_contributions(w, COV)
    assert d.portfolio_vol == pytest.approx(portfolio_vol(w, COV))
    assert d.contribution.sum() == pytest.approx(d.portfolio_vol)
    assert d.relative.sum() == pytest.approx(1.0)
    # 30 % du capital mais une part de risque plus grande (actif le plus volatil)
    assert d.relative[1] > 0.3


def test_zero_portfolio_has_no_risk():
    d = risk_contributions(np.zeros(3), COV)
    assert d.portfolio_vol == 0.0 and not d.relative.any()


def test_concentration_measures():
    assert herfindahl(np.array([0.25] * 4)) == pytest.approx(0.25)
    assert effective_n(np.array([0.25] * 4)) == pytest.approx(4.0)
    assert effective_n(np.array([1.0, 0.0])) == pytest.approx(1.0)
    assert diversification_ratio(np.array([0.5, 0.5, 0.0]), COV) > 1.0


def test_drawdown_tracker(t0):
    dd = DrawdownTracker()
    for v in (100, 110, 99, 105, 120, 108):
        dd.update(v, t0)
    assert dd.drawdown == pytest.approx(108 / 120 - 1)
    assert dd.max_drawdown == pytest.approx(99 / 110 - 1)
    assert dd.peak == 120


def test_shrinkage_and_psd():
    shrunk = shrink_to_diagonal(COV, 1.0)
    assert np.allclose(shrunk, np.diag(np.diag(COV)))
    bad = np.array([[1.0, 2.0], [2.0, 1.0]])  # valeur propre négative
    fixed = nearest_psd(bad)
    assert np.linalg.eigvalsh(fixed).min() >= -1e-12
    with pytest.raises(ValueError):
        shrink_to_diagonal(COV, 1.5)


def _feature_engine_with_history(t0, n=60):
    fe = FeatureEngine(["1h"], momentum_horizons=[], correlation_timeframes=["1h"])
    rng = np.random.default_rng(0)
    prices = {"A": 100.0, "B": 50.0}
    for i in range(n):
        start = t0 + i * timedelta(hours=1)
        common = rng.normal(0, 0.004)
        for sym, own in (("A", 0.002), ("B", 0.006)):
            prices[sym] *= math.exp(common + rng.normal(0, own))
            fe.on_bar(BarEvent(timestamp=start, end=start + timedelta(hours=1), symbol=sym,
                               source="t", timeframe="1h", open=prices[sym], high=prices[sym],
                               low=prices[sym], close=prices[sym]))
    return fe


def test_annualized_covariance_alignment(t0):
    fe = _feature_engine_with_history(t0)
    cov = annualized_covariance(fe, ["B", "A", "C"], "1h", shrinkage=0.0)
    assert cov.shape == (3, 3)
    assert cov[0, 0] > cov[1, 1] > 0          # B plus volatil que A
    assert cov[2, 2] == 0.0                   # C inconnu
    assert cov[0, 1] > 0                       # facteur commun
    assert annualized_covariance(FeatureEngine(["1h"]), ["A"], "1h") is None


def test_risk_engine_report_and_breaches(t0):
    fe = _feature_engine_with_history(t0)
    pf = Portfolio(cash=0.0, positions={"A": Position("A", 5, 100.0), "B": Position("B", 10, 50.0)},
                   target_weights={"A": 0.8, "B": 0.2})
    pf.update_price("A", 100.0, t0)
    pf.update_price("B", 50.0, t0)
    risk = RiskEngine(fe, timeframe="1h",
                      limits=RiskLimits(max_weight=0.45, max_risk_contribution=0.6,
                                        max_drawdown=0.05, max_drift=0.1))
    for v in (1000, 1100, 1000):
        risk.on_value(v, t0)
    report = risk.evaluate(pf.snapshot())

    assert report.portfolio_vol > 0
    shares = [p.risk_share for p in report.positions.values()]
    assert sum(shares) == pytest.approx(1.0)
    assert report.positions["B"].risk_share > 0.5   # même poids, plus de risque
    assert report.effective_positions == pytest.approx(2.0)
    assert report.effective_bets < 2.0
    assert report.drawdown == pytest.approx(1000 / 1100 - 1)
    kinds = {(b.kind, b.symbol) for b in report.breaches}
    assert ("WEIGHT", "A") in kinds and ("WEIGHT", "B") in kinds
    assert ("DRAWDOWN", None) in kinds
    assert ("DRIFT", "A") in kinds
    assert all(b.severity > 0 for b in report.breaches)
