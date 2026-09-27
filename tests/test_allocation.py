import numpy as np
import pytest

from trading_engine.allocation.allocator import RiskAllocator
from trading_engine.allocation.constraints import ConstraintEngine, Constraints
from trading_engine.allocation.drift import DriftMonitor
from trading_engine.allocation.hrp import hrp_weights, quasi_diagonal_order, correlation_distance
from trading_engine.allocation.risk_parity import inverse_volatility_weights, risk_budget_weights
from trading_engine.allocation.targeting import volatility_target
from trading_engine.risk.portfolio_risk import portfolio_vol, risk_contributions
from trading_engine.signals.signal import Signal

COV = np.array([
    [0.040, 0.030, 0.000, 0.000],
    [0.030, 0.045, 0.000, 0.000],
    [0.000, 0.000, 0.010, 0.002],
    [0.000, 0.000, 0.002, 0.020],
])
SYMBOLS = ["SPY", "QQQ", "TLT", "GLD"]


# ------------------------------------------------------------------ risk parity / HRP

def test_equal_risk_contributions():
    w = risk_budget_weights(COV)
    assert w.sum() == pytest.approx(1.0)
    assert risk_contributions(w, COV).relative == pytest.approx([0.25] * 4, abs=1e-6)


def test_custom_budgets_and_zero_budget():
    b = np.array([0.4, 0.0, 0.4, 0.2])
    w = risk_budget_weights(COV, b)
    assert w[1] == 0.0
    assert risk_contributions(w, COV).relative == pytest.approx([0.4, 0.0, 0.4, 0.2], abs=1e-6)
    with pytest.raises(ValueError):
        risk_budget_weights(COV, np.zeros(4))


def test_inverse_volatility():
    w = inverse_volatility_weights(np.diag([0.04, 0.01]))
    assert w == pytest.approx([1 / 3, 2 / 3])


def test_hrp_clusters_correlated_assets():
    order = quasi_diagonal_order(correlation_distance(COV))
    # SPY/QQQ (très corrélés) sont adjacents dans l'ordre quasi-diagonal
    assert abs(order.index(0) - order.index(1)) == 1
    w = hrp_weights(COV)
    assert w.sum() == pytest.approx(1.0) and (w > 0).all()
    # le cluster actions (plus risqué) reçoit moins que le cluster défensif
    assert w[0] + w[1] < w[2] + w[3]


def test_hrp_ignores_assets_without_variance():
    cov = COV.copy()
    cov[3, :] = cov[:, 3] = 0.0
    w = hrp_weights(cov)
    assert w[3] == 0.0 and w.sum() == pytest.approx(1.0)


def test_volatility_target_respects_gross_cap():
    w = np.array([0.25] * 4)
    scaled, scale = volatility_target(w, COV, 0.05, max_gross=1.0)
    assert portfolio_vol(scaled, COV) == pytest.approx(0.05)
    capped, scale = volatility_target(w, COV, 5.0, max_gross=1.0)
    assert capped.sum() == pytest.approx(1.0) and scale == pytest.approx(1.0)


# ------------------------------------------------------------------ allocator + attribution

def _signals(t0, means):
    return {s: Signal(s, "30m", mean=m, std=0.01, n_obs=100, timestamp=t0, source="t")
            for s, m in zip(SYMBOLS, means)}


def test_attribution_sums_to_final_weight(t0):
    alloc = RiskAllocator("signal", target_vol=0.08)
    weights, attribution, _ = alloc.allocate(SYMBOLS, COV, _signals(t0, [0.002, -0.001, 0.001, 0.0]), 0.5)
    for sym in SYMBOLS:
        assert sum(attribution[sym].values()) == pytest.approx(weights[sym])
    assert weights["QQQ"] == 0.0 and weights["GLD"] == 0.0   # signal négatif / nul
    assert weights["SPY"] > 0 and weights["TLT"] > 0
    assert set(attribution["SPY"]) == {"risk_allocation", "signal_tilt", "vol_target"}


def test_no_skill_means_no_signal_budget(t0):
    alloc = RiskAllocator("signal")
    weights, _, _ = alloc.allocate(SYMBOLS, COV, _signals(t0, [0.01] * 4), skill=-0.1)
    assert sum(weights.values()) == 0.0


def test_unknown_method():
    with pytest.raises(ValueError):
        RiskAllocator("kelly")


# ------------------------------------------------------------------ constraints

def test_position_sector_and_gross_constraints():
    engine = ConstraintEngine(Constraints(
        max_weight=0.4, min_cash=0.1,
        sectors={"SPY": "equity", "QQQ": "equity"}, sector_limits={"equity": 0.5},
    ))
    out = engine.apply({"SPY": 0.5, "QQQ": 0.3, "TLT": 0.3, "GLD": -0.1})
    w = out.weights
    assert w["GLD"] == 0.0                                  # long-only
    assert w["SPY"] <= 0.4 + 1e-12
    assert w["SPY"] + w["QQQ"] <= 0.5 + 1e-12               # secteur
    assert sum(w.values()) <= 0.9 + 1e-12                   # cash minimum
    assert {"MAX_WEIGHT:SPY", "MIN_WEIGHT:GLD", "SECTOR:equity"} <= set(out.binding)
    for sym in w:
        assert w[sym] == pytest.approx({"SPY": 0.5, "QQQ": 0.3, "TLT": 0.3, "GLD": -0.1}[sym]
                                       + out.adjustments[sym])


def test_portfolio_vol_constraint():
    engine = ConstraintEngine(Constraints(max_weight=1.0, max_portfolio_vol=0.05))
    out = engine.apply(dict(zip(SYMBOLS, [0.25] * 4)), cov=COV, cov_symbols=SYMBOLS)
    w = np.array([out.weights[s] for s in SYMBOLS])
    assert portfolio_vol(w, COV) == pytest.approx(0.05)
    assert "PORTFOLIO_VOL" in out.binding


def test_turnover_constraint_moves_partially():
    engine = ConstraintEngine(Constraints(max_weight=1.0, max_turnover=0.1))
    out = engine.apply({"SPY": 0.5}, current={"SPY": 0.2, "TLT": 0.1})
    turnover = abs(out.weights["SPY"] - 0.2) + abs(out.weights["TLT"] - 0.1)
    assert turnover == pytest.approx(0.1)
    assert 0.2 < out.weights["SPY"] < 0.5
    assert out.binding == ("TURNOVER",)


# ------------------------------------------------------------------ drift

def test_drift_monitor():
    dm = DriftMonitor(threshold=0.05)
    assert dm.on_new_target({"A": 0.5, "B": 0.5}) is None
    assert dm.on_new_target({"A": 0.6, "B": 0.4}) == pytest.approx(0.2)
    report = dm.evaluate({"A": 0.5, "B": 0.42}, {"A": 0.6, "B": 0.4})
    assert report.max_abs_drift == pytest.approx(0.1)
    assert report.turnover_to_target == pytest.approx(0.12)
    assert report.over_threshold == ("A",)
    assert report.target_change == pytest.approx(0.2)
