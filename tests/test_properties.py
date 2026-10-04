"""Tests de propriétés (hypothesis) : des milliers de cas tirés au hasard,
vérifiés contre des règles qui ne doivent jamais être violées."""

import asyncio
import math
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from trading_engine.allocation.constraints import ConstraintEngine, Constraints, fit_gross_after_freeze, gross_cap_of
from trading_engine.allocation.risk_parity import risk_budget_weights
from trading_engine.backtest.dataset import synthetic_trades
from trading_engine.config import load_config
from trading_engine.data.calendar import MarketCalendar
from trading_engine.data.events import BarEvent, OrderUpdateEvent, TradeEvent
from trading_engine.engine import Engine
from trading_engine.execution.hard_controls import HardControls, HardLimits
from trading_engine.execution.orders import OrderProposal
from trading_engine.portfolio.portfolio import Portfolio
from trading_engine.risk.portfolio_risk import portfolio_vol
from trading_engine.safety.safety_engine import SafetyState
from trading_engine.storage.event_log import dumps, loads
from trading_engine.tax.fx import FxRates
from trading_engine.tax.profile import Instrument, load_tax_profile
from trading_engine.config import PROJECT_ROOT
from trading_engine.tax.tax_model import TaxModel

UTC = timezone.utc
SYMS = ["A", "B", "C", "D", "E"]
weight = st.floats(-2.0, 2.0, allow_nan=False)
FAST = settings(max_examples=300, deadline=None)


def _cov(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = rng.normal(size=(n, n)) * 0.01
    return m @ m.T + np.eye(n) * 1e-4


# ------------------------------------------------------------------ allocation

@FAST
@given(targets=st.dictionaries(st.sampled_from(SYMS), weight, min_size=1),
       max_weight=st.floats(0.05, 1.0), min_cash=st.floats(0.0, 0.5), max_gross=st.floats(0.1, 1.5),
       sector_limit=st.floats(0.05, 1.0), max_vol=st.one_of(st.none(), st.floats(0.001, 0.5)),
       long_only=st.booleans(), seed=st.integers(0, 1000))
def test_constraint_engine_always_respects_limits(targets, max_weight, min_cash, max_gross, sector_limit,
                                                   max_vol, long_only, seed):
    c = Constraints(max_weight=max_weight, min_weight=0.0 if long_only else -max_weight, max_gross=max_gross,
                    min_cash=min_cash, max_portfolio_vol=max_vol, sectors={"A": "x", "B": "x"},
                    sector_limits={"x": sector_limit})
    symbols = sorted(set(targets))
    cov = _cov(len(symbols), seed)
    out = ConstraintEngine(c).apply(targets, {}, cov, symbols).weights
    w = np.array([out[s] for s in symbols])
    eps = 1e-9
    assert np.all(np.isfinite(w))
    assert np.all(w <= max_weight + eps) and np.all(w >= c.min_weight - eps)
    cap = gross_cap_of(c)                     # avec levier (max_gross > 1) : max_gross fait foi
    assert np.abs(w).sum() <= max(cap, 0) + eps or cap <= 0
    assert abs(out.get("A", 0)) + abs(out.get("B", 0)) <= sector_limit + eps
    if max_vol is not None:
        assert portfolio_vol(w, cov) <= max_vol * (1 + 1e-6)


@FAST
@given(targets=st.dictionaries(st.sampled_from(SYMS), st.floats(0.0, 0.4), min_size=2),
       current=st.dictionaries(st.sampled_from(SYMS), st.floats(0.0, 0.4)),
       turnover=st.floats(0.01, 2.0))
def test_turnover_limit_keeps_a_feasible_target_feasible(targets, current, turnover):
    """Rebalancement partiel = mélange de deux portefeuilles admissibles :
    reste admissible (contraintes convexes)."""
    gross_cap = 0.98
    assume(sum(current.values()) <= gross_cap)
    c = Constraints(max_weight=0.4, min_weight=0.0, max_gross=1.0, min_cash=0.02, max_turnover=turnover)
    out = ConstraintEngine(c).apply(targets, current).weights
    assert all(-1e-12 <= v <= 0.4 + 1e-9 for v in out.values())
    assert sum(out.values()) <= gross_cap + 1e-9
    moved = sum(abs(out[s] - current.get(s, 0.0)) for s in out)
    assert moved <= turnover + 1e-9


@FAST
@given(target=st.dictionaries(st.sampled_from(SYMS), st.floats(0.0, 0.6), min_size=1),
       current=st.dictionaries(st.sampled_from(SYMS), st.floats(0.0, 0.6)),
       frozen=st.sets(st.sampled_from(SYMS)), cap=st.floats(0.1, 1.0))
def test_frozen_symbols_never_push_gross_over_cap(target, current, frozen, cap):
    target = {**{s: current.get(s, 0.0) for s in frozen if s in current}, **target}
    for s in frozen:
        if s in target:
            target[s] = current.get(s, 0.0)                 # gelé = poids courant
    out = fit_gross_after_freeze(target, current, frozen, cap)
    assert set(out) == set(target)
    for s in frozen & set(target):
        assert out[s] == target[s]
    frozen_gross = sum(abs(target[s]) for s in frozen & set(target))
    if frozen_gross <= cap:
        assert sum(abs(v) for v in out.values()) <= cap + 1e-9
    if sum(abs(v) for v in target.values()) <= cap:
        assert out == target


@FAST
@given(budgets=st.lists(st.floats(0.01, 1.0), min_size=2, max_size=5), seed=st.integers(0, 1000))
def test_risk_parity_weights_are_valid(budgets, seed):
    cov = _cov(len(budgets), seed)
    w = risk_budget_weights(cov, np.array(budgets) / sum(budgets))
    assert np.all(np.isfinite(w)) and np.all(w >= -1e-12) and abs(w.sum() - 1) < 1e-6


# ------------------------------------------------------------------ hard controls

@FAST
@given(price=st.floats(1.0, 5000.0), value=st.floats(1_000.0, 5_000_000.0),
       volume=st.one_of(st.none(), st.floats(1.0, 1e7)), side=st.sampled_from([1, -1]))
def test_order_sized_at_cap_passes_and_above_fails(price, value, volume, side):
    t = datetime(2026, 3, 2, 15, tzinfo=UTC)
    limits = HardLimits(max_order_notional=50_000, max_order_weight=0.2, max_participation=0.1,
                        max_daily_turnover=1.0)
    cap = HardControls(limits).max_order_quantity(price, value, volume, t)
    assume(cap > 1e-6)

    def check(qty):
        order = OrderProposal("SPY", side * qty, price, t, "D1")
        return HardControls(limits).check_order(order, portfolio_value=value, last_price=price,
                                                recent_volume=volume, safety_state=SafetyState.NORMAL)
    assert check(cap).approved, check(cap).violations
    assert not check(cap * 1.01 + 1e-6).approved


@FAST
@given(weights=st.dictionaries(st.sampled_from(SYMS), st.one_of(st.floats(-3, 3), st.just(math.nan), st.just(math.inf)), min_size=1))
def test_targets_validation_matches_limits(weights):
    lim = HardLimits(max_target_weight=0.5, max_target_gross=1.0)
    ok = HardControls(lim).validate_targets(weights).approved
    finite = all(math.isfinite(w) for w in weights.values())
    expected = finite and all(abs(w) <= 0.5 + 1e-9 for w in weights.values()) and \
        sum(abs(w) for w in weights.values()) <= 1.0 + 1e-9
    assert ok == expected


# ------------------------------------------------------------------ fiscalité, change

BE = load_tax_profile(PROJECT_ROOT / "config" / "taxes" / "BE.toml")
trade = st.tuples(st.sampled_from(["SPY", "TLT"]), st.floats(-50, 50).filter(lambda q: abs(q) > 0.01),
                  st.floats(10.0, 1000.0), st.integers(0, 700))


@FAST
@given(trades=st.lists(trade, max_size=40), rate=st.floats(0.8, 1.5))
def test_tax_lots_track_holdings_and_ledger_roundtrips(trades, rate):
    model = TaxModel(BE, {"SPY": Instrument("SPY", "etf", "US"), "TLT": Instrument("TLT", "etf", "US")},
                     portfolio_currency="USD")
    model.set_fx(FxRates("EUR", "USD", {date(2025, 1, 1): rate, date(2026, 1, 1): rate * 1.1}))
    held = {"SPY": 0.0, "TLT": 0.0}
    for sym, qty, price, day in trades:
        qty = qty if qty > 0 else -min(-qty, held[sym])      # long-only : pas de vente à découvert
        if abs(qty) < 1e-9:
            continue
        charge, _ = model.record_fill(sym, qty, price, date(2025, 1, 1) + timedelta(days=day))
        held[sym] += qty
        rule = model.profile.transaction_rule(model.instrument(sym))
        assert 0 <= charge.amount <= rule.cap + 1e-9
        assert charge.portfolio_amount == pytest.approx(charge.amount * model.fx.rate(date(2025, 1, 1) + timedelta(days=day)))
    for sym in held:
        assert model.lot_quantity(sym) == pytest.approx(held[sym], abs=1e-6)
    clone = TaxModel(BE, model.instruments, portfolio_currency="USD")
    clone.restore(loads_json(model.snapshot()))
    assert clone.snapshot() == model.snapshot()
    assert clone.transaction_taxes_paid == pytest.approx(model.transaction_taxes_paid)


def loads_json(obj):
    import json
    return json.loads(json.dumps(obj))


@FAST
@given(amount=st.floats(-1e9, 1e9), rate=st.floats(0.5, 2.0), day=st.integers(0, 3000))
def test_fx_roundtrip(amount, rate, day):
    fx = FxRates("EUR", "USD", {date(2020, 1, 1): rate, date(2024, 6, 3): rate * 0.9})
    when = date(2019, 1, 1) + timedelta(days=day)
    assert fx.to_quote(fx.to_base(amount, when), when) == pytest.approx(amount, rel=1e-12, abs=1e-6)


# ------------------------------------------------------------------ calendrier, données

CAL = MarketCalendar.from_rules(date(2020, 1, 1), date(2030, 12, 31))


@settings(max_examples=500, deadline=None)
@given(minutes=st.integers(0, 11 * 365 * 24 * 60))
def test_calendar_consistency(minutes):
    ts = datetime(2020, 1, 1, tzinfo=UTC) + timedelta(minutes=minutes)
    session = CAL.session_at(ts)
    assert CAL.is_open(ts) == (session is not None)
    if session is not None:
        assert session.open <= ts < session.close and session.close - session.open <= timedelta(hours=6, minutes=30)
    nxt = CAL.next_open(ts)
    if nxt is not None:
        assert nxt > ts and CAL.is_open(nxt) and not CAL.is_open(nxt - timedelta(seconds=1))
    if CAL.trading_block(ts, 5, 15) is None:
        assert session is not None


bar_values = st.tuples(st.floats(1.0, 1000.0), st.floats(1.0, 1000.0), st.floats(1.0, 1000.0),
                       st.floats(1.0, 1000.0), st.floats(0.0, 1e7))


@FAST
@given(values=bar_values)
def test_synthetic_trades_stay_inside_the_bar(values):
    a, b, c, d, volume = values
    low, high = min(a, b, c, d), max(a, b, c, d)
    t = datetime(2026, 3, 2, 14, 30, tzinfo=UTC)
    bar = BarEvent(timestamp=t, end=t + timedelta(minutes=30), symbol="X", source="t", timeframe="30m",
                   open=b, high=high, low=low, close=c, volume=volume)
    trades = synthetic_trades(bar)
    assert trades[0].price == bar.open and trades[-1].price == bar.close
    assert all(low <= x.price <= high and t <= x.timestamp < bar.end for x in trades)
    assert sum(x.size for x in trades) == pytest.approx(volume)


@FAST
@given(price=st.floats(0.01, 1e6), size=st.floats(0, 1e9), qty=st.floats(0, 1e6),
       avg=st.one_of(st.none(), st.floats(0.01, 1e6)), seconds=st.integers(0, 10**9))
def test_event_log_roundtrip(price, size, qty, avg, seconds):
    t = datetime(2000, 1, 1, tzinfo=UTC) + timedelta(seconds=seconds)
    for ev in (TradeEvent(timestamp=t, received_at=t, symbol="X", source="t", price=price, size=size),
               OrderUpdateEvent(timestamp=t, received_at=t, symbol="X", source="t", client_order_id="qb-1-X",
                                update="fill", side="buy", fill_qty=qty, fill_price=price, filled_qty=qty,
                                filled_avg_price=avg)):
        assert dumps(loads(dumps(ev))) == dumps(ev)


# ------------------------------------------------------------------ fills broker : idempotence

def _update(cid, filled, avg, kind="partial_fill", qty=1.0, price=100.0):
    t = datetime(2026, 3, 2, 15, tzinfo=UTC)
    return OrderUpdateEvent(timestamp=t, received_at=t, symbol="SPY", source="alpaca_trading", client_order_id=cid,
                            update=kind, side="buy", fill_qty=qty, fill_price=price, filled_qty=filled,
                            filled_avg_price=avg)


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(executions=st.lists(st.tuples(st.integers(1, 50), st.floats(50.0, 150.0)), min_size=1, max_size=8),
       order=st.randoms(use_true_random=False), duplicates=st.integers(0, 6))
def test_broker_fills_are_idempotent(executions, order, duplicates):
    """Mises à jour dupliquées, perdues ou désordonnées : la position finale est
    toujours la quantité cumulée réellement exécutée, au bon coût total."""
    import dataclasses
    cfg = load_config()
    e = Engine(dataclasses.replace(cfg, tax=dataclasses.replace(cfg.tax, profile=None),
                                   portfolio=dataclasses.replace(cfg.portfolio, positions={}, cash=1e6)))
    updates, filled, value = [], 0.0, 0.0
    for i, (qty, price) in enumerate(executions):
        filled += qty
        value += qty * price
        kind = "fill" if i == len(executions) - 1 else "partial_fill"
        updates.append(_update("qb-D1-SPY", filled, value / filled, kind, qty, price))
    noisy = updates + [order.choice(updates) for _ in range(duplicates)]
    last = noisy.pop(len(updates) - 1)            # la dernière mise à jour arrive à la fin...
    order.shuffle(noisy)
    noisy = [u for u in noisy if order.random() > 0.3] + [last]   # ... d'autres sont perdues
    for u in noisy:
        e._on_order_update(u)
    pos = e.portfolio.positions["SPY"]
    assert pos.quantity == pytest.approx(filled)
    assert 1e6 - e.portfolio.cash == pytest.approx(value, rel=1e-9)
    assert sum(f.quantity for f in e.fills) == pytest.approx(filled)


# ------------------------------------------------------------------ moteur complet

@settings(max_examples=6, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10_000), method=st.sampled_from(["hrp", "risk_parity", "signal", "baseline"]),
       risk_aversion=st.sampled_from([5.0, 50.0, 5000.0]), max_weight=st.floats(0.2, 0.5),
       min_cash=st.floats(0.0, 0.2), fill_share=st.floats(0.1, 1.0), vol=st.floats(0.05, 0.6))
def test_engine_never_violates_invariants(seed, method, risk_aversion, max_weight, min_cash, fill_share, vol):
    """Configs et marchés aléatoires (y compris très volatils) : aucun invariant
    violé, aucune exception dans les gestionnaires."""
    import dataclasses
    cfg = load_config()
    c = cfg.allocation.constraints
    cfg = dataclasses.replace(
        cfg,
        engine=dataclasses.replace(cfg.engine, max_events=9000, report_every=0),
        feed=dataclasses.replace(cfg.feed, seed=seed, annual_vol=vol),
        allocation=dataclasses.replace(cfg.allocation, method=method,
                                       constraints=dataclasses.replace(c, max_weight=max_weight, min_cash=min_cash)),
        risk=dataclasses.replace(cfg.risk, min_observations=5),
        decision=dataclasses.replace(cfg.decision, risk_aversion=risk_aversion, holding_period="20d"),
        execution=dataclasses.replace(cfg.execution, mode="paper", fill_share=fill_share),
    )
    e = Engine(cfg)
    asyncio.run(e.run())
    assert e.invariants.checks > 0
    assert e.invariants.violations == [], [str(v) for v in e.invariants.violations]
    assert e.bus.error_count == 0
