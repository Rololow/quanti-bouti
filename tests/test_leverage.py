"""Levier, financement, budgets de risque par classe et contrôle du drawdown."""

import asyncio
import dataclasses
import json
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest

from trading_engine.allocation.allocator import RiskAllocator
from trading_engine.allocation.constraints import ConstraintEngine, Constraints, gross_cap_of
from trading_engine.backtest.benchmarks import StrategyResult, simulate
from trading_engine.backtest.runner import walk_forward
from trading_engine.backtest.sources import yahoo_short_rates
from trading_engine.config import AllocationConfig, PROJECT_ROOT, load_config
from trading_engine.engine import Engine
from trading_engine.portfolio.financing import RateSeries, accrue
from trading_engine.risk.portfolio_risk import risk_contributions
from trading_engine.safety.invariants import check_portfolio

UTC = timezone.utc
LEVERAGE = PROJECT_ROOT / "config" / "leverage.yaml"


# ------------------------------------------------------------------ financement

def test_accrue_and_rate_series():
    assert accrue(10_000, 0.04, 365, borrow_spread=0.005, credit_spread=0.0) == pytest.approx(400.0)
    assert accrue(-10_000, 0.04, 365, borrow_spread=0.005, credit_spread=0.0) == pytest.approx(-450.0)
    assert accrue(10_000, 0.04, 365, borrow_spread=0.0, credit_spread=0.0, credit_cash=False) == 0.0
    assert accrue(10_000, 0.04, 0, borrow_spread=0.0, credit_spread=0.0) == 0.0
    s = RateSeries({date(2020, 1, 2): 0.015, date(2020, 1, 6): 0.02}, "test")
    assert s.rate(date(2020, 1, 4)) == 0.015 and s.rate(date(2019, 1, 1)) == 0.015 and s.rate(date(2021, 1, 1)) == 0.02
    assert RateSeries.from_payload(s.to_payload()).to_payload() == s.to_payload()
    with pytest.raises(ValueError):
        RateSeries({date(2020, 1, 1): 4.5})                 # un pourcentage, pas un décimal


def test_engine_accrues_interest_daily():
    cfg = load_config()
    e = Engine(dataclasses.replace(cfg, financing=dataclasses.replace(cfg.financing, enabled=True)))
    e.rates = RateSeries.fixed(0.05, date(2020, 1, 1))
    t0 = datetime(2026, 3, 2, 15, tzinfo=UTC)
    e.portfolio.cash = 1_000.0
    e._accrue_financing(t0)
    e._accrue_financing(t0 + timedelta(hours=3))           # même jour : rien
    e._accrue_financing(t0 + timedelta(days=10))
    assert e.portfolio.cash == pytest.approx(1_000 * (1 + 0.05 * 10 / 365))
    e.portfolio.cash = -1_000.0
    e._accrue_financing(t0 + timedelta(days=375))
    assert e.portfolio.cash == pytest.approx(-1_000 * (1 + 0.055))
    assert e.financing_cost == pytest.approx(55.0) and e.interest_earned > 0


def test_yahoo_short_rates_in_decimal():
    ts = int(datetime(2025, 3, 3, 14, 30, tzinfo=UTC).timestamp())
    payload = {"chart": {"error": None, "result": [{"timestamp": [ts], "indicators": {
        "quote": [{"open": [4.2], "high": [4.3], "low": [4.1], "close": [4.25], "volume": [0]}],
        "adjclose": [{"adjclose": [4.25]}]}}]}}
    assert yahoo_short_rates(date(2025, 3, 1), date(2025, 3, 5),
                             http_get=lambda u: json.dumps(payload)) == {date(2025, 3, 3): pytest.approx(0.0425)}


def test_benchmarks_earn_interest_on_cash():
    days = [date(2026, 1, 1) + timedelta(days=i) for i in range(366)]
    prices = {"A": {d: 100.0 for d in days}}
    plain = simulate("x", days, prices, lambda i: {"A": 0.5} if i == 0 else None, cash=1000.0, cost_bps=0.0)
    paid = simulate("x", days, prices, lambda i: {"A": 0.5} if i == 0 else None, cash=1000.0, cost_bps=0.0,
                    cash_rate=lambda d: 0.04)
    assert plain.equity[-1][1] == pytest.approx(1000.0)
    assert paid.equity[-1][1] == pytest.approx(1000.0 + 500.0 * 0.04, rel=1e-3)


# ------------------------------------------------------------------ budgets de classes

def _cov():
    vols = np.array([0.16, 0.18, 0.07, 0.15])
    corr = np.array([[1, .85, -.3, .05], [.85, 1, -.25, .1], [-.3, -.25, 1, .2], [.05, .1, .2, 1]])
    return np.outer(vols, vols) * corr


def test_adjustable_class_budgets():
    symbols = ["SPY", "EFA", "IEF", "GLD"]
    classes = {"SPY": "equity", "EFA": "equity", "IEF": "bonds", "GLD": "commodities"}
    alloc = RiskAllocator("class_parity", target_vol=10.0, classes=classes,
                          class_budgets={"equity": 0.5, "bonds": 0.25, "commodities": 0.25})
    assert alloc.class_budgets(symbols).tolist() == pytest.approx([0.25, 0.25, 0.25, 0.25])
    w, _, _ = alloc.allocate(symbols, _cov())
    rc = risk_contributions(np.array([w[s] for s in symbols]), _cov()).relative
    assert rc[0] + rc[1] == pytest.approx(0.5, abs=1e-6) and rc[2] == pytest.approx(0.25, abs=1e-6)
    # classe absente de la table : part moyenne des classes listées
    partial = RiskAllocator("class_parity", classes=classes, class_budgets={"equity": 0.6, "bonds": 0.2})
    b = partial.class_budgets(symbols)
    assert b[0] + b[1] == pytest.approx(0.6 / 1.2) and b[3] == pytest.approx(0.4 / 1.2)
    with pytest.raises(ValueError):
        AllocationConfig(class_budgets={"equity": -1})


# ------------------------------------------------------------------ levier

def test_gross_cap_with_leverage():
    assert gross_cap_of(Constraints(max_gross=1.0, min_cash=0.02)) == pytest.approx(0.98)
    assert gross_cap_of(Constraints(max_gross=2.0, min_cash=0.02)) == 2.0
    out = ConstraintEngine(Constraints(max_weight=0.8, max_gross=2.0, min_cash=0.02)).apply(
        {"A": 0.8, "B": 0.8, "C": 0.8}).weights
    assert sum(out.values()) == pytest.approx(2.0)


def test_cash_invariant_allows_only_authorized_borrowing():
    pos = {"A": (15.0, 100.0)}                               # 1 500 investis, valeur 1 000
    assert check_portfolio(-500.0, pos, max_gross=2.0, long_only=True) == []
    assert [v.name for v in check_portfolio(-500.0, pos, max_gross=1.0, long_only=True)] == ["CASH", "GROSS"]
    assert "CASH" in [v.name for v in check_portfolio(-1_200.0, {"A": (22.0, 100.0)}, max_gross=2.0, long_only=True)]


def test_drawdown_control_scale():
    cfg = load_config()
    e = Engine(dataclasses.replace(cfg, allocation=dataclasses.replace(cfg.allocation, drawdown_control=0.15)))
    dd = e.risk.portfolio_drawdown
    dd.update(100.0)
    assert e._drawdown_scale() == 1.0
    dd.update(92.5)                                          # -7,5 % : moitié de la limite
    assert e._drawdown_scale() == pytest.approx((1 - 0.85 / 0.925) / 0.15)
    dd.update(85.0)
    assert e._drawdown_scale() == 0.0
    dd.update(120.0)                                         # nouveau plus haut : pleine exposition
    assert e._drawdown_scale() == 1.0
    assert Engine(load_config())._drawdown_scale() == 1.0     # désactivé par défaut


def test_leveraged_engine_borrows_within_limits():
    cfg = load_config("config/config.yaml", (LEVERAGE,))
    cfg = dataclasses.replace(
        cfg, engine=dataclasses.replace(cfg.engine, max_events=12000, report_every=0),
        allocation=dataclasses.replace(cfg.allocation, target_vol=0.60, drawdown_control=None),
        risk=dataclasses.replace(cfg.risk, min_observations=5,
                                 limits=dataclasses.replace(cfg.risk.limits, max_drawdown=0.5)),
        decision=dataclasses.replace(cfg.decision, risk_aversion=5000.0, holding_period="20d"),
        execution=dataclasses.replace(cfg.execution, mode="paper"))
    e = Engine(cfg)
    lowest_cash = []
    e.bus.subscribe("bar", lambda b: lowest_cash.append(e.portfolio.cash))
    asyncio.run(e.run())
    assert min(lowest_cash) < 0                              # le moteur a emprunté
    assert e.invariants.violations == [] and e.bus.error_count == 0
    assert sum(abs(w) for w in e.portfolio.target_weights.values()) > 1.0


def test_walk_forward_maximizes_return_under_drawdown_limit():
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(2 * 365)]

    def path(daily, crash):
        v, out = 100.0, []
        for i, d in enumerate(days):
            v *= 1 + daily - (crash if i == 100 else 0.0)
            out.append((d, v))
        return out

    safe = StrategyResult("safe", path(0.0002, 0.05), params={"p": "safe"})
    risky = StrategyResult("risky", path(0.003, 0.40), params={"p": "risky"})     # bien plus rentable malgré -40 %
    _, sharpe_folds = walk_forward([safe, risky], days[0], days[-1], default=safe)
    _, dd_folds = walk_forward([safe, risky], days[0], days[-1], default=safe, max_drawdown=0.15)
    assert dd_folds[1]["chosen"] == "safe" and "≥ -15%" in dd_folds[1]["reason"]
    _, loose = walk_forward([safe, risky], days[0], days[-1], default=safe, max_drawdown=0.50)
    assert loose[1]["chosen"] == "risky"


# ------------------------------------------------------------------ étape 1 : levier sans blocage

def test_rolling_drawdown_forgets_old_peak():
    from trading_engine.risk.portfolio_risk import RollingDrawdown

    dd = RollingDrawdown(365)
    d0 = date(2020, 1, 1)
    dd.update(100.0, d0)
    dd.update(90.0, d0 + timedelta(days=10))
    assert dd.drawdown == pytest.approx(-0.10) and dd.peak == 100.0
    dd.update(88.0, d0 + timedelta(days=364))
    assert dd.drawdown == pytest.approx(88 / 100 - 1)
    dd.update(88.0, d0 + timedelta(days=366))                # le plus haut de 100 a expiré
    assert dd.peak == 90.0 and dd.drawdown == pytest.approx(88 / 90 - 1)
    dd.update(95.0, d0 + timedelta(days=366))                # même jour : plus haut du jour
    assert dd.peak == 95.0 and dd.drawdown == 0.0
    with pytest.raises(ValueError):
        RollingDrawdown(0)


def test_drawdown_control_uses_rolling_peak_when_configured():
    cfg = load_config("config/config.yaml", (LEVERAGE,))
    e = Engine(cfg)
    assert e.rolling_drawdown is not None and e.rolling_drawdown.window_days == 365
    d0 = date(2020, 1, 1)
    e.risk.portfolio_drawdown.update(100.0)
    e.risk.portfolio_drawdown.update(80.0)                   # -20 % du plus haut historique
    e.rolling_drawdown.update(100.0, d0)
    e.rolling_drawdown.update(80.0, d0 + timedelta(days=400))
    assert e._drawdown_scale() == 1.0                        # nouveau plus haut glissant : réexposé


def test_decision_forces_deleveraging_above_gross_cap():
    from trading_engine.decision.rebalance import UREBALANCE, DecisionConfig, DecisionContext, DecisionEngine

    cov = np.array([[0.04, 0.0], [0.0, 0.01]])
    base = dict(timestamp=datetime(2026, 3, 2, 15, tzinfo=UTC), portfolio_value=100_000.0,
                current={"A": 1.25, "B": 1.10}, target={"A": 1.0, "B": 1.0}, symbols=["A", "B"], cov=cov,
                prices={"A": 100.0, "B": 50.0}, spreads={"A": 0.0, "B": 0.0})
    # Coûts énormes et bande large : sans plafond, rien ne bouge.
    engine = DecisionEngine(DecisionConfig(default_spread_bps=500.0, no_trade_band=0.5, holding_period="5d"))
    assert engine.decide(DecisionContext(**base)).action != UREBALANCE
    d = engine.decide(DecisionContext(**base, max_gross=2.0))
    assert d.action == UREBALANCE and d.fraction == 1.0
    assert d.execution_weights == pytest.approx({"A": 1.0, "B": 1.0})
    assert "désendettement" in d.reasons[0]
    # Dans la tolérance : décision économique habituelle.
    near = dict(base, current={"A": 1.02, "B": 1.01})
    assert engine.decide(DecisionContext(**near, max_gross=2.0)).action != UREBALANCE


def test_leverage_overlay_does_not_halt_on_losses():
    cfg = load_config("config/config.yaml", (LEVERAGE,))
    assert cfg.safety.halt_on_breaches == ()
    assert cfg.safety.invariant_gross_tolerance >= 0.2 and cfg.decision.delever_tolerance == 0.05


def test_walk_forward_lookback_unlocks_after_old_breach():
    days = [date(2020, 1, 1) + timedelta(days=i) for i in range(6 * 365)]

    def path(daily, crash_day, crash):
        v, out = 100.0, []
        for i, d in enumerate(days):
            v *= 1 + daily - (crash if i == crash_day else 0.0)
            out.append((d, v))
        return out

    safe = StrategyResult("safe", path(0.0001, 10, 0.0), params={"p": "safe"})
    risky = StrategyResult("risky", path(0.0008, 100, 0.25), params={"p": "risky"})   # -25 % la 1re année
    _, locked = walk_forward([safe, risky], days[0], days[-1], default=safe, max_drawdown=0.15)
    _, recent = walk_forward([safe, risky], days[0], days[-1], default=safe, max_drawdown=0.15, lookback_years=2)
    assert [f["chosen"] for f in locked][-1] == "safe"
    assert [f["chosen"] for f in recent][-1] == "risky"


def test_variant_params_accept_none_and_tuples():
    from trading_engine.backtest.runner import VARIANTS, with_params

    cfg = with_params(load_config("config/config.yaml", (LEVERAGE,)), VARIANTS["lev15_ancien"])
    assert cfg.allocation.drawdown_window_days is None
    assert cfg.safety.halt_on_breaches == ("DRAWDOWN", "LEVERAGE")
    cfg = with_params(cfg, {"safety.halt_on_breaches": "DRAWDOWN+LEVERAGE"})
    assert cfg.safety.halt_on_breaches == ("DRAWDOWN", "LEVERAGE")


# ------------------------------------------------------------------ étape 2 : filtre de tendance

def test_trend_filter_scores_horizons():
    from trading_engine.allocation.trend import TrendFilter

    tf = TrendFilter((2, 4), floor=0.25)
    for p in (100, 101, 102, 103, 104):
        tf.on_close("UP", p)
    for p in (100, 99, 98, 97, 96):
        tf.on_close("DOWN", p)
    for p in (100, 110, 120, 100, 105):                      # +5 % sur 2, +5 % sur 4... puis
        tf.on_close("MIX", p)
    tf.on_close("SHORT", 100)
    assert tf.score("UP") == 1.0 and tf.score("DOWN") == 0.0 and tf.score("SHORT") is None
    assert tf.score("MIX") == 0.5                             # 120 -> 105 en baisse, 100 -> 105 en hausse
    assert tf.score("UP", cash_rate=252 * 0.01) == 0.0         # hausse < rendement du cash
    m = tf.multipliers(["UP", "DOWN", "SHORT"])
    assert m == {"UP": 1.0, "DOWN": 0.25, "SHORT": 1.0}
    with pytest.raises(ValueError):
        TrendFilter((0,))


def test_engine_trend_filter_scales_down_falling_assets():
    cfg = load_config("config/config.yaml", (LEVERAGE,))
    with pytest.raises(ValueError, match="trend.timeframe"):
        Engine(dataclasses.replace(cfg, allocation=dataclasses.replace(
            cfg.allocation, trend=dataclasses.replace(cfg.allocation.trend, enabled=True, timeframe="1w"))))
    cfg = dataclasses.replace(cfg, allocation=dataclasses.replace(
        cfg.allocation, trend=dataclasses.replace(cfg.allocation.trend, enabled=True, horizons=(2, 4))))
    tf = cfg.allocation.trend.timeframe
    assert tf in cfg.bar_timeframes
    e = Engine(cfg)
    for p in (100, 99, 98, 97, 96):
        e.trend.on_close("SPY", p)
        e.trend.on_close("TLT", 200 - p)
    requested, attribution = {"SPY": 0.5, "TLT": 0.5}, {}
    e._apply_trend(requested, attribution, ["SPY", "TLT"], None)
    assert requested == {"SPY": 0.0, "TLT": 0.5} and attribution["SPY"]["trend"] == -0.5
    redist = dataclasses.replace(cfg, allocation=dataclasses.replace(
        cfg.allocation, trend=dataclasses.replace(cfg.allocation.trend, redistribute=True)))
    e2 = Engine(redist)
    e2.trend = e.trend
    requested = {"SPY": 0.5, "TLT": 0.5}
    e2._apply_trend(requested, {}, ["SPY", "TLT"], np.array([[0.04, 0.0], [0.0, 0.01]]))
    # TLT seul à 0,5 : vol 5 % -> remis à la vol cible 12 %, soit 1,2 (sous le plafond brut 2).
    assert requested["SPY"] == 0.0 and requested["TLT"] == pytest.approx(0.5 * 0.12 / 0.05)


# ------------------------------------------------------------------ étape 3 : tendance lente, CPPI

def test_slow_trend_updates_weekly_and_ignores_small_changes():
    from trading_engine.allocation.trend import TrendFilter

    tf = TrendFilter((1, 2, 3, 4), update_every=5, min_change=0.5)
    for p in (100, 101, 102, 103, 104):
        tf.on_close("A", p)
    assert tf.held_score("A") == 1.0
    tf.on_close("A", 103.5)                                  # 1 horizon sur 4 bascule
    assert tf.score("A") == 0.75 and tf.held_score("A") == 1.0   # pas encore une semaine
    for p in (103.6, 103.7, 103.8, 103.75):                  # 5 clôtures : réévaluation
        tf.on_close("A", p)
    assert tf.score("A") == 0.75 and tf.held_score("A") == 1.0   # écart 0,25 < 0,5 : gardé
    for p in (102, 101, 100, 99, 98):                        # tout en baisse : 0 appliqué
        tf.on_close("A", p)
    assert tf.held_score("A") == 0.0
    fast = TrendFilter((1, 2, 3, 4))
    for p in (100, 101, 102, 103, 104, 103.5):
        fast.on_close("A", p)
    assert fast.held_score("A") == 0.75                      # défaut : immédiat


def test_cppi_multiplier_keeps_exposure_in_small_drawdowns():
    cfg = load_config()
    base = dataclasses.replace(cfg.allocation, drawdown_control=0.15)
    gz = Engine(dataclasses.replace(cfg, allocation=base))
    cppi = Engine(dataclasses.replace(cfg, allocation=dataclasses.replace(base, drawdown_multiplier=2.0)))
    for e in (gz, cppi):
        e.risk.portfolio_drawdown.update(100.0)
        e.risk.portfolio_drawdown.update(95.0)               # -5 %
    assert gz._drawdown_scale() == pytest.approx((1 - 0.85 / 0.95) / 0.15)
    assert cppi._drawdown_scale() == pytest.approx(min(1.0, 2 * (1 - 0.85 / 0.95) / 0.15))
    for e in (gz, cppi):
        e.risk.portfolio_drawdown.update(85.0)
    assert gz._drawdown_scale() == 0.0 and cppi._drawdown_scale() == 0.0
    with pytest.raises(ValueError):
        AllocationConfig(drawdown_multiplier=0.5)
