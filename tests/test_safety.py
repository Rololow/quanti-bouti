import asyncio
import dataclasses
import math
from datetime import timedelta

import pytest

from trading_engine.config import load_config
from trading_engine.data.events import TradeEvent
from trading_engine.data.market_feed import MarketFeed
from trading_engine.engine import Engine
from trading_engine.execution.hard_controls import HardControls, HardLimits, OrderProposal
from trading_engine.safety.safety_engine import ModelHealth, SafetyConfig, SafetyEngine, SafetyState


# ------------------------------------------------------------------ safety engine

def test_escalation_and_recovery(t0):
    se = SafetyEngine(SafetyConfig(recovery_intervals=2))
    assert se.evaluate(t0, data_scores={"SPY": 1.0}).state is SafetyState.NORMAL
    st = se.evaluate(t0, data_scores={"SPY": 0.8})
    assert st.state is SafetyState.DEGRADED and "SPY" in st.frozen_symbols
    assert se.evaluate(t0, data_scores={"SPY": 1.0}).state is SafetyState.DEGRADED
    assert se.evaluate(t0, data_scores={"SPY": 1.0}).state is SafetyState.NORMAL


def test_model_drift_degrades_and_explosion_halts(t0):
    se = SafetyEngine()
    assert se.evaluate(t0, model_health=ModelHealth(drift_events=1)).state is SafetyState.DEGRADED
    st = se.evaluate(t0, model_health=ModelHealth(non_finite=True, details=["factor"]))
    assert st.state is SafetyState.HALTED and not st.can_decide


def test_halted_requires_manual_reset(t0):
    se = SafetyEngine()
    se.on_value(1000, t0)
    se.on_value(960, t0 + timedelta(hours=1))           # -4 % > 3 %
    st = se.evaluate(t0 + timedelta(hours=1))
    assert st.state is SafetyState.HALTED and st.reasons[0].startswith("DAILY_LOSS")
    se.on_value(1000, t0 + timedelta(hours=2))          # la perte disparaît...
    assert se.evaluate(t0 + timedelta(hours=2)).state is SafetyState.HALTED   # ...mais on reste arrêté
    assert se.reset("robin", t0).state is SafetyState.DEGRADED


def test_daily_loss_resets_each_day(t0):
    se = SafetyEngine()
    se.on_value(1000, t0)
    se.on_value(990, t0 + timedelta(hours=1))
    se.on_value(990, t0 + timedelta(days=1))            # nouveau jour : nouvelle référence
    assert se.daily_return == 0.0


def test_critical_breach_and_rejections_halt(t0):
    se = SafetyEngine(SafetyConfig(max_hard_control_rejections=2))
    assert se.evaluate(t0, risk_breaches=["DRIFT"]).state is SafetyState.NORMAL
    se.record_hard_control_rejection(t0, "x")
    se.record_hard_control_rejection(t0, "y")
    assert se.evaluate(t0).state is SafetyState.HALTED
    assert SafetyEngine().evaluate(t0, risk_breaches=["DRAWDOWN"]).state is SafetyState.HALTED


def test_corporate_action_freezes_symbol_temporarily(t0):
    se = SafetyEngine(SafetyConfig(corporate_action_hold=3600))
    st = se.evaluate(t0, corporate_actions={"AAPL": t0})
    assert st.state is SafetyState.DEGRADED and st.frozen_symbols == {"AAPL"}
    later = se.evaluate(t0 + timedelta(hours=2), corporate_actions={"AAPL": t0})
    assert "AAPL" not in later.frozen_symbols


# ------------------------------------------------------------------ hard controls

def test_target_validation_blocks_crazy_model():
    hc = HardControls(HardLimits(max_target_weight=0.5, max_target_gross=1.0))
    assert hc.validate_targets({"A": 0.4, "B": 0.4}).approved
    bad = hc.validate_targets({"A": 0.99, "B": 0.2})
    assert not bad.approved and any(v.startswith("TARGET_WEIGHT A") for v in bad.violations)
    assert not hc.validate_targets({"A": 0.4, "B": 0.4, "C": 0.4}).approved
    assert not hc.validate_targets({"A": math.nan}).approved


def test_order_checks(t0):
    hc = HardControls(HardLimits(max_order_notional=10_000, max_order_weight=0.2,
                                 max_participation=0.1, price_collar=0.05,
                                 max_orders_per_day=2, max_daily_turnover=0.3))
    ok = dict(portfolio_value=100_000, last_price=100.0, recent_volume=10_000,
              safety_state=SafetyState.NORMAL)

    assert hc.check_order(OrderProposal("A", 50, 100.0, t0), **ok).approved
    fat_finger = hc.check_order(OrderProposal("A", 50, 150.0, t0), **ok)
    assert any(v.startswith("PRICE_COLLAR") for v in fat_finger.violations)
    too_big = hc.check_order(OrderProposal("A", 2_000, 100.0, t0), **ok)
    assert {v.split()[0] for v in too_big.violations} >= {"ORDER_NOTIONAL", "PARTICIPATION"}
    halted = hc.check_order(OrderProposal("A", 1, 100.0, t0), **{**ok, "safety_state": SafetyState.HALTED})
    assert "SAFETY_HALTED" in halted.violations
    assert hc.check_order(OrderProposal("A", 1, 100.0, t0), **ok).approved
    assert "ORDERS_PER_DAY 2" in hc.check_order(OrderProposal("A", 1, 100.0, t0), **ok).violations


# ------------------------------------------------------------------ bout en bout

class ListFeed(MarketFeed):
    def __init__(self, events):
        self.events = events

    async def __aiter__(self):
        for ev in self.events:
            yield ev


def _config(**alloc):
    cfg = load_config()
    return dataclasses.replace(
        cfg,
        engine=dataclasses.replace(cfg.engine, report_every=0),
        allocation=dataclasses.replace(cfg.allocation, **alloc),
    )


def test_glitch_never_reaches_the_engine(t0):
    events = []
    for i in range(40):
        ts = t0 + timedelta(seconds=5 * i)
        price = 1000.0 if i == 30 else 100.0 + 0.01 * (i % 3)
        events.append(TradeEvent(timestamp=ts, received_at=ts, symbol="SPY", source="t",
                                 price=price, size=10))
    engine = Engine(_config(), feed=ListFeed(events))
    asyncio.run(engine.run())
    assert engine.market_state.get("SPY").price < 101
    assert engine.features.snapshot("SPY").get("vwap_dist", 0) < 0.01
    assert engine.integrity.counts.get("GLITCH") == 1


def test_halted_engine_stops_reallocating(t0):
    engine = Engine(_config(method="hrp"))
    engine.feed.max_events = 8000
    asyncio.run(engine.run())
    allocations_before = engine.last_allocation
    assert allocations_before is not None

    # Forçons un HALT (explosion d'un modèle) puis une nouvelle évaluation.
    engine.models.factor_model.regression.coef[0] = math.inf
    status = engine.evaluate_safety()
    assert status.state is SafetyState.HALTED
    engine._interval_due = True
    asyncio.run(engine._on_interval())
    assert engine.last_allocation is allocations_before   # aucune nouvelle cible
