import asyncio
import dataclasses
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from trading_engine.config import load_config
from trading_engine.data.events import TradeEvent
from trading_engine.decision.rebalance import DecisionConfig, DecisionContext, DecisionEngine
from trading_engine.engine import Engine
from trading_engine.execution.cost_model import CostConfig, ExecutionCostModel
from trading_engine.execution.feedback import ExecutionFeedback
from trading_engine.execution.fill_model import FillModel
from trading_engine.execution.hard_controls import HardControls, HardLimits
from trading_engine.execution.optimizer import OptimizerConfig, OrderOptimizer
from trading_engine.execution.order_pricer import distance_to_execution, limit_price
from trading_engine.execution.orders import OrderProposal
from trading_engine.execution.paper_broker import PaperBroker
from trading_engine.execution.volume import VolumeTracker

T = datetime(2026, 3, 2, 15, 0, tzinfo=timezone.utc)


# ------------------------------------------------------------------ coûts

def test_square_root_impact_and_spread():
    m = ExecutionCostModel(CostConfig(impact_eta=0.5, slippage_bps=0.0))
    small = m.estimate(1_000, 100.0, rel_spread=0.001, daily_vol=0.02, adv=1_000_000)
    big = m.estimate(4_000, 100.0, rel_spread=0.001, daily_vol=0.02, adv=1_000_000)
    assert big.impact_fraction == pytest.approx(2 * small.impact_fraction)   # racine carrée
    assert small.impact_fraction == pytest.approx(0.5 * 0.02 * math.sqrt(0.001))
    assert small.spread == pytest.approx(100_000 * 0.001 / 2)
    passive = m.estimate(1_000, 100.0, rel_spread=0.001, crossing=0.0)
    assert passive.spread == 0.0


def test_fees():
    m = ExecutionCostModel(CostConfig(commission_per_order=1.0, commission_per_share=0.005))
    assert m.estimate(200, 50.0).fees == pytest.approx(2.0)


# ------------------------------------------------------------------ fill

def test_touch_probability():
    f = FillModel()
    assert f.touch_probability(-0.001, 0.01) == 1.0                  # croise le spread
    assert f.touch_probability(0.0, 0.01) == 1.0
    assert f.touch_probability(0.01, 0.01) == pytest.approx(2 * (1 - 0.8413447), abs=1e-6)
    assert f.touch_probability(0.05, 0.01) < 1e-5


def test_participation_limits_fill():
    f = FillModel(max_participation=0.1)
    assert f.expected_fill(1_000, -1.0, 0.01, expected_market_volume=5_000) == pytest.approx(0.5)
    f.calibration = 0.8
    assert f.expected_fill(100, -1.0, 0.01, expected_market_volume=100_000) == pytest.approx(0.8)


# ------------------------------------------------------------------ prix limite

def test_limit_prices():
    assert limit_price("buy", 99.98, 100.02, 0.0) == pytest.approx(99.98)
    assert limit_price("buy", 99.98, 100.02, 1.0) == pytest.approx(100.02)
    assert limit_price("sell", 99.98, 100.02, 0.0) == pytest.approx(100.02)
    assert limit_price("sell", 99.98, 100.02, 1.0) == pytest.approx(99.98)
    assert limit_price("buy", 99.981, 100.019, 1.0) == pytest.approx(100.02)   # reste exécutable
    assert distance_to_execution("buy", 100.02, 99.98, 100.02) == 0.0
    assert distance_to_execution("buy", 99.98, 99.98, 100.02) > 0
    with pytest.raises(ValueError):
        limit_price("buy", 101, 100, 0.5)


# ------------------------------------------------------------------ optimiseur

def _optimizer(**cfg):
    volume = VolumeTracker("5m")
    volume.update("A", 50_000)
    params = dict(aggressiveness=(0.0, 1.0), durations=("5m", "1h"), timing_risk_aversion=0.0)
    params.update(cfg)
    return OrderOptimizer(OptimizerConfig(**params), ExecutionCostModel(CostConfig(slippage_bps=0.0)),
                          FillModel(), volume)


def test_valuable_urgent_order_crosses_the_spread():
    opt = _optimizer()
    o = opt.optimize("A", 10_000, value=1_000, urgency=1.0, bid=99.5, ask=100.5,
                     daily_vol=0.02, timestamp=T)
    assert o.aggressiveness == 1.0 and o.expected_fill == pytest.approx(1.0)
    assert o.quantity > 0 and o.limit_price == pytest.approx(100.5)


def test_low_value_order_waits_passively():
    opt = _optimizer()
    o = opt.optimize("A", -10_000, value=60, urgency=0.0, bid=99.5, ask=100.5,
                     daily_vol=0.02, timestamp=T)
    assert o.aggressiveness == 0.0 and o.quantity < 0
    assert o.limit_price == pytest.approx(100.5)                     # vente passive à l'ask
    assert o.duration == timedelta(hours=1)                           # laisse le temps au marché
    assert 0 < o.expected_fill < 1


def test_quantity_rounded_toward_zero():
    opt = _optimizer()
    assert opt.optimize("A", 150, value=10, urgency=0, bid=99.9, ask=100.1, daily_vol=0.02,
                        timestamp=T).quantity == 1
    assert opt.optimize("A", 50, value=10, urgency=0, bid=99.9, ask=100.1, daily_vol=0.02,
                        timestamp=T) is None


# ------------------------------------------------------------------ paper broker

def _order(qty, limit, minutes=30):
    return OrderProposal("A", qty, limit, T, "D1", duration=timedelta(minutes=minutes),
                         arrival_mid=100.0, expected_fill=0.9, expected_cost=5.0)


def _trade(seconds, price, size=100, sym="A"):
    ts = T + timedelta(seconds=seconds)
    return TradeEvent(timestamp=ts, received_at=ts, symbol=sym, source="t", price=price, size=size)


def test_paper_broker_fills_at_limit_within_trade_size():
    broker = PaperBroker(fill_share=0.5)
    broker.submit(_order(80, 100.0))
    assert broker.on_trade(_trade(1, 100.5)).fills == []            # au-dessus de la limite
    first = broker.on_trade(_trade(2, 99.9, size=100))
    assert first.fills[0].quantity == 50 and first.fills[0].price == 100.0
    second = broker.on_trade(_trade(3, 99.8, size=100))
    assert second.fills[0].quantity == 30 and second.closed[0].average_price == 100.0
    assert not broker.working


def test_paper_broker_expiry_and_replacement():
    broker = PaperBroker()
    broker.submit(_order(10, 90.0, minutes=5))
    replaced = broker.submit(_order(20, 91.0, minutes=5))
    assert replaced.order.quantity == 10
    update = broker.on_trade(_trade(301, 100.0))                     # expiré
    assert update.closed and update.closed[0].filled == 0 and not broker.working


# ------------------------------------------------------------------ feedback

def test_feedback_calibrates_fill_and_builds_confidence():
    fill, cost = FillModel(), ExecutionCostModel()
    fb = ExecutionFeedback(fill, cost, lam=0.5, prior_orders=10)
    assert fb.execution_confidence == 0.0                             # hypothèses seulement
    for _ in range(20):
        fb.on_order_closed(_order(10, 100.0), filled_quantity=5, average_price=100.1)
    assert fill.calibration == pytest.approx(0.5 / 0.9, rel=0.05)
    assert 0 < fb.execution_confidence < 1
    assert fb.records[-1].shortfall == pytest.approx(0.001)
    assert cost.eta == cost.config.impact_eta                         # paper : impact non appris


def test_feedback_learns_impact_when_enabled():
    cost = ExecutionCostModel(CostConfig(impact_eta=0.5))
    fb = ExecutionFeedback(FillModel(), cost, lam=0.5, learn_impact=True)
    for _ in range(20):                                               # coût réalisé 2× le prédit
        fb.on_order_closed(_order(10, 100.0), filled_quantity=10, average_price=101.0)
    assert cost.eta > 0.5


# ------------------------------------------------------------------ hard controls

def test_order_sizing_respects_limits_and_turnover_budget():
    hc = HardControls(HardLimits(max_order_notional=5_000, max_order_weight=0.2,
                                 max_participation=0.1, max_daily_turnover=0.3))
    assert hc.max_order_quantity(100.0, 100_000, recent_volume=1_000) == pytest.approx(50)
    assert hc.max_order_quantity(100.0, 100_000, recent_volume=100) == pytest.approx(10)
    from trading_engine.safety.safety_engine import SafetyState
    ok = dict(portfolio_value=10_000, last_price=100.0, recent_volume=None,
              safety_state=SafetyState.NORMAL)
    assert hc.check_order(OrderProposal("A", 20, 100.0, T), **ok).approved   # 20 % du budget de 30 %
    assert hc.max_order_quantity(100.0, 10_000, None, T) == pytest.approx(10)
    assert hc.check_order(OrderProposal("A", 10, 100.0, T), **ok).approved
    assert hc.max_order_quantity(100.0, 10_000, None, T) == 0.0      # budget épuisé


# ------------------------------------------------------------------ décision + exécution

def test_decision_uses_execution_cost_model_and_confidence():
    ctx = dict(timestamp=T, portfolio_value=1_000_000.0, current={"A": 0.0}, target={"A": 0.5},
               symbols=["A"], cov=np.array([[0.04]]), prices={"A": 100.0}, spreads={"A": 0.001},
               daily_vols={"A": 0.02}, adv={"A": 100_000})
    engine = DecisionEngine(DecisionConfig(), cost_model=ExecutionCostModel())
    unknown = engine.decide(DecisionContext(**ctx))
    confident = engine.decide(DecisionContext(**ctx, execution_confidence=1.0))
    doubtful = engine.decide(DecisionContext(**ctx, execution_confidence=0.0))
    assert unknown.costs.impact > 0
    assert confident.uncertainty == pytest.approx(unknown.uncertainty)
    assert doubtful.uncertainty > confident.uncertainty                # coûts incertains = marge exigée
    assert doubtful.execution_confidence == 0.0


# ------------------------------------------------------------------ bout en bout

def _paper_config(mode="paper", tax=False, risk_aversion=50.0):
    cfg = load_config()
    return dataclasses.replace(
        cfg,
        engine=dataclasses.replace(cfg.engine, max_events=12000, report_every=0),
        allocation=dataclasses.replace(cfg.allocation, method="hrp"),
        risk=dataclasses.replace(cfg.risk, min_observations=5),
        decision=dataclasses.replace(cfg.decision, risk_aversion=risk_aversion, holding_period="20d"),
        execution=dataclasses.replace(cfg.execution, mode=mode),
        tax=dataclasses.replace(cfg.tax, profile=cfg.tax.profile if tax else None),
    )


def test_paper_trading_moves_positions_safely():
    e = Engine(_paper_config())
    before = {s: p.quantity for s, p in e.portfolio.positions.items()}
    asyncio.run(e.run())
    state = e.snapshot()
    assert e.fills and e.execution_feedback.records
    assert {s: p.quantity for s, p in e.portfolio.positions.items()} != before
    assert state.leverage <= 1.0 + 1e-9 and state.cash >= 0
    assert sum(len(p.rejected) for p in e.plans) == 0      # plafonds respectés dès le plan
    assert e.safety.status.state.value != "HALTED"
    assert 0 < e.execution_feedback.execution_confidence < 1


def test_paper_fills_pay_transaction_tax():
    # γ très élevé : l'écart de risque justifie de payer la TOB (0,35 %).
    e = Engine(_paper_config(tax=True, risk_aversion=5_000.0))
    asyncio.run(e.run())
    assert e.fills and e.tax.transaction_taxes_paid > 0


def test_proposals_mode_never_fills():
    e = Engine(_paper_config(mode="proposals"))
    asyncio.run(e.run())
    assert e.plans and any(p.orders for p in e.plans)
    assert e.fills == [] and e.broker is None


def test_paper_trading_replays_identically(tmp_path):
    log = tmp_path / "events.jsonl"
    cfg = _paper_config()
    live = Engine(dataclasses.replace(cfg, storage=dataclasses.replace(cfg.storage, event_log=str(log))))
    asyncio.run(live.run())
    replay = Engine(dataclasses.replace(cfg, feed=dataclasses.replace(cfg.feed, provider="replay",
                                                                      replay_path=str(log))))
    asyncio.run(replay.run())
    assert replay.fills == live.fills
    assert list(replay.plans) == list(live.plans)
    assert replay.snapshot() == live.snapshot()
