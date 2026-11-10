"""Simulateur rapide : briques vectorisées contre les versions de référence du moteur."""

from datetime import date, timedelta

import numpy as np
import pytest

from trading_engine.allocation.risk_parity import risk_budget_weights
from trading_engine.backtest import fastsim
from trading_engine.backtest.runner import with_params
from trading_engine.config import PROJECT_ROOT, load_config


def _random_cov(rng, n):
    a = rng.normal(size=(n, n))
    return a @ a.T / n + 0.01 * np.eye(n)


def test_batch_risk_parity_matches_reference():
    rng = np.random.default_rng(3)
    covs = np.stack([_random_cov(rng, 8) for _ in range(12)])
    budgets = np.array([0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05])
    batch = fastsim.risk_budget_weights_batch(covs, budgets)
    for k in range(len(covs)):
        assert batch[k] == pytest.approx(risk_budget_weights(covs[k], budgets), abs=1e-6)


def _synthetic_market(days=400, seed=1):
    rng = np.random.default_rng(seed)
    start = date(2015, 1, 1)
    all_days = [start + timedelta(days=i) for i in range(days)]
    closes = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, size=(days, 8)), axis=0))
    rates = np.full(days, 0.01)
    symbols = ["DBC", "EEM", "EFA", "GLD", "IEF", "SPY", "TIP", "TLT"]
    return fastsim.Market(all_days, symbols, closes, rates, {})


def _profile_cfg():
    cfg = load_config(PROJECT_ROOT / "config" / "config.yaml",
                      (PROJECT_ROOT / "config" / "backtest_daily.yaml", PROJECT_ROOT / "config" / "profil_belge.yaml"))
    return with_params(cfg, {"allocation.gross_band": 0.10})


def test_bootstrap_keeps_calendar_and_is_reproducible():
    m = _synthetic_market()
    a = fastsim.block_bootstrap_market(m, 21, np.random.default_rng(5))
    b = fastsim.block_bootstrap_market(m, 21, np.random.default_rng(5))
    assert a.days == m.days and a.closes.shape == m.closes.shape
    assert np.allclose(a.closes[0], m.closes[0]) and np.allclose(a.closes, b.closes)
    # Les rendements sont des blocs de l'historique : chaque rendement existe dans la source.
    src = set(np.round(np.diff(np.log(m.closes), axis=0), 12).ravel())
    assert set(np.round(np.diff(np.log(a.closes), axis=0), 12).ravel()) <= src


def test_profile_simulation_runs_and_accounts_cash():
    m = _synthetic_market()
    cfg = _profile_cfg()
    run = fastsim.simulate_profile(m, cfg)
    assert len(run.equity) == len(m.days) == len(run.exposure)
    assert all(np.isfinite(v) for _, v in run.equity)
    assert run.stats["max_gross"] < 2.2                      # dérive des prix comprise, sous le plafond de 2×
    assert run.stats["liquidation_cost"] >= 0


def test_monte_carlo_summary_fields():
    m = _synthetic_market()
    cfg = _profile_cfg()
    results = fastsim.monte_carlo_profile(m, cfg, paths=3, block=21, seed=2)
    s = fastsim.summarize_paths(results)
    assert s["paths"] == 3 and set(s["max_drawdown"]) == {"p05", "p25", "p50", "p75", "p95"}
    assert 0.0 <= s["p_max_drawdown_below_15pct"] <= 1.0
