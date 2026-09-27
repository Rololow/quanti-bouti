import asyncio

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
