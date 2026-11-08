"""Audit de robustesse : formules et réglages de l'audit (le backtest complet tourne sur GitHub Actions)."""

from datetime import date, timedelta

import numpy as np
import pytest

from trading_engine.backtest import audit
from trading_engine.backtest.stats import deflated_sharpe
from trading_engine.config import PROJECT_ROOT, load_config
from trading_engine.backtest.runner import with_params


def test_deflated_for_k_matches_stats_for_the_same_trials():
    rng = np.random.default_rng(1)
    returns = list(rng.normal(0.0004, 0.01, 2500))
    trials = [0.2, 0.35, 0.5, 0.1, 0.45, 0.3]
    sd = float(np.std(trials, ddof=1))
    expected = deflated_sharpe(returns, trials)["dsr"]
    assert audit.deflated_for_k(returns, sd, len(trials)) == pytest.approx(expected, rel=1e-9)


def test_more_trials_lower_the_deflated_probability():
    rng = np.random.default_rng(2)
    returns = list(rng.normal(0.0005, 0.01, 2500))
    p_few = audit.deflated_for_k(returns, 0.1, 5)
    p_many = audit.deflated_for_k(returns, 0.1, 60)
    assert 0.0 <= p_many < p_few <= 1.0


def test_one_at_a_time_overrides_exist_in_the_profile_config():
    overlays = tuple(PROJECT_ROOT / o for o in audit.OVERLAYS)
    base = load_config(PROJECT_ROOT / "config" / "config.yaml", overlays)
    labels = [label for label, _ in audit.ONE_AT_A_TIME]
    assert len(labels) == len(set(labels))
    for label, params in audit.ONE_AT_A_TIME:
        cfg = with_params(base, params)                     # lève KeyError si le chemin n'existe pas
        assert cfg is not base, label
    cfg = with_params(base, {"allocation.trend.horizons": (63, 126, 252)})
    assert cfg.allocation.trend.horizons == (63, 126, 252)


def test_window_metrics_on_a_known_series():
    start = date(2020, 1, 1)
    points = [(start + timedelta(days=i), 100.0 * (1.0 + 0.001) ** i) for i in range(400)]
    m = audit.window_metrics(points, start, start + timedelta(days=399))
    assert m["cagr"] == pytest.approx((1.001 ** 365.25) - 1, rel=0.01)
    assert m["max_drawdown"] == pytest.approx(0.0, abs=1e-12)


def test_markdown_has_the_three_sections():
    row = {"label": "retenu", "params": {},
           "full": {"cagr_liquide": 0.056, "cagr": 0.057, "max_drawdown": -0.138, "sharpe": 0.71,
                    "vol": 0.083, "tob": 8597.0, "rotation": 2.22},
           "first_half": {"cagr": 0.05, "vol": 0.08, "sharpe": 0.6, "max_drawdown": -0.13},
           "second_half": {"cagr": 0.06, "vol": 0.08, "sharpe": 0.8, "max_drawdown": -0.1}}
    out = {"results": [row], "dsr": {"audit_trials": 1, "total_trials_estimate": 30, "p_single": 0.95,
                                     "sd_sharpe_annual": 0.0,
                                     "grid": {"mesurée": {"audit": 0.9, "total": 0.8}}}}
    md = audit.to_markdown(out)
    for section in ("## Sensibilité", "## Deux périodes", "## Sharpe dégonflé"):
        assert section in md
    assert "8 597" in md and "5,6 %" in md
