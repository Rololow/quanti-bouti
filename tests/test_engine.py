import asyncio
import pytest

from trading_engine.config import load_config
from trading_engine.engine import Engine


def _run(max_events=3000):
    cfg = load_config()
    cfg = cfg.__class__(**{**cfg.__dict__,
                           "engine": cfg.engine.__class__(max_events=max_events, report_every=0)})
    engine = Engine(cfg)
    state = asyncio.run(engine.run())
    return engine, state


def test_engine_runs_end_to_end():
    engine, state = _run()
    assert engine.bus.error_count == 0
    assert engine.bus.published_count >= 3000
    assert {b.timeframe for b in engine.bars} >= {"5m"}
    assert state.total_value > 0
    assert {p.symbol for p in state.positions} == {"SPY", "QQQ", "TLT", "GLD"}


def test_engine_is_deterministic():
    _, a = _run(1000)
    _, b = _run(1000)
    assert a == b


def test_bars_close_on_market_clock_for_idle_symbols(t0):
    from datetime import timedelta

    from trading_engine.data.events import TradeEvent
    from trading_engine.data.market_feed import MarketFeed

    class ListFeed(MarketFeed):
        def __init__(self, events):
            self.events = events

        async def __aiter__(self):
            for ev in self.events:
                yield ev

    def trade(sym, minutes, price):
        return TradeEvent(timestamp=t0 + timedelta(minutes=minutes), symbol=sym,
                          source="t", price=price, size=1)

    feed = ListFeed([trade("SPY", 0, 100), trade("TLT", 1, 90), trade("SPY", 6, 101)])
    engine = Engine(load_config(), feed=feed)
    asyncio.run(engine.run())

    closed_5m = {(b.symbol, b.timeframe) for b in engine.bars}
    assert ("TLT", "5m") in closed_5m  # TLT n'a plus tradé mais sa barre est close


def test_models_can_be_disabled():
    import dataclasses

    cfg = load_config()
    models = dataclasses.replace(
        cfg.models,
        regimes=dataclasses.replace(cfg.models.regimes, enabled=False),
        factor=dataclasses.replace(cfg.models.factor, enabled=False),
    )
    cfg = dataclasses.replace(
        cfg, models=models,
        engine=dataclasses.replace(cfg.engine, max_events=3000, report_every=0),
    )
    engine = Engine(cfg)
    asyncio.run(engine.run())
    assert engine.models.factor_model is None
    assert all(not engine.models.regimes(s) for s in engine.features.symbols())


def test_risk_breaches_are_published_once():
    import dataclasses

    from trading_engine.data.events import EventType

    cfg = load_config()
    limits = dataclasses.replace(cfg.risk.limits, max_drift=0.01)  # dépassé dès le départ
    cfg = dataclasses.replace(
        cfg, risk=dataclasses.replace(cfg.risk, limits=limits),
        engine=dataclasses.replace(cfg.engine, max_events=8000, report_every=0),
    )
    engine = Engine(cfg)
    risk_events = []
    engine.bus.subscribe(EventType.RISK, risk_events.append)
    asyncio.run(engine.run())

    assert engine.risk_report is not None
    assert risk_events, "au moins un dépassement de drift attendu"
    first = risk_events[0].payload["new"]
    assert any(key.startswith("DRIFT:") for key in first)

    # Un dépassement persistant n'est publié qu'une fois ; il l'est de nouveau
    # s'il disparaît puis réapparaît.
    report = engine.risk_report
    breach = next(b for b in report.breaches if b.kind == "DRIFT")
    engine._active_breaches = frozenset()
    risk_events.clear()
    persistent = dataclasses.replace(report, breaches=(breach,))
    asyncio.run(engine._publish_breaches(persistent))
    asyncio.run(engine._publish_breaches(persistent))
    assert len(risk_events) == 1
    asyncio.run(engine._publish_breaches(dataclasses.replace(report, breaches=())))
    asyncio.run(engine._publish_breaches(persistent))
    assert len(risk_events) == 2


def test_fill_pays_transaction_tax_and_creates_tax_lot(t0):
    import dataclasses

    cfg = load_config()
    engine = Engine(dataclasses.replace(cfg, engine=dataclasses.replace(cfg.engine, report_every=0)))
    cash = engine.portfolio.cash
    engine.record_fill("GLD", 10, 100.0, t0)             # ETF US : TOB 0,35 %
    assert engine.portfolio.cash == pytest.approx(cash - 1000.0 - 3.5)
    assert engine.tax.gains.lots["GLD"][0].quantity == 10


def test_tax_can_be_disabled():
    import dataclasses

    cfg = load_config()
    engine = Engine(dataclasses.replace(cfg, tax=dataclasses.replace(cfg.tax, profile=None)))
    assert engine.tax is None and engine.estimate_rebalance_tax() is None
