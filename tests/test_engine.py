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
