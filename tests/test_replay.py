import asyncio
import dataclasses
from datetime import timedelta

import pytest

from trading_engine.config import load_config
from trading_engine.data.events import BarEvent, FundamentalEvent, QuoteEvent, TradeEvent
from trading_engine.data.replay_feed import ReplayFeed
from trading_engine.engine import Engine
from trading_engine.storage.event_log import EventLogWriter, dumps, loads, read_events


def _config(**overrides):
    cfg = load_config()
    engine = dataclasses.replace(cfg.engine, max_events=overrides.pop("max_events", 8000),
                                 report_every=0)
    feed = dataclasses.replace(cfg.feed, **overrides.pop("feed", {}))
    storage = dataclasses.replace(cfg.storage, **overrides.pop("storage", {}))
    allocation = dataclasses.replace(cfg.allocation, **overrides.pop("allocation", {}))
    return dataclasses.replace(cfg, engine=engine, feed=feed, storage=storage,
                               allocation=allocation, **overrides)


def test_event_roundtrip(t0):
    events = [
        TradeEvent(timestamp=t0, received_at=t0 + timedelta(milliseconds=3), symbol="SPY",
                   source="alpaca", price=650.25, size=100, payload={"conditions": ("@", "T")}),
        QuoteEvent(timestamp=t0, symbol="SPY", source="alpaca", bid=650.1, ask=650.3),
        BarEvent(timestamp=t0, end=t0 + timedelta(minutes=1), symbol="SPY", source="alpaca",
                 timeframe="1m", open=1, high=2, low=0.5, close=1.5, volume=10, trade_count=3),
        FundamentalEvent(timestamp=t0, symbol="AAPL", source="t", name="eps", value=1.5,
                         period="Q2", available_at=t0 + timedelta(days=2)),
    ]
    for ev in events:
        back = loads(dumps(ev))
        assert type(back) is type(ev)
        assert back.timestamp == ev.timestamp
        assert back.received_at == ev.received_at
        assert {k: v for k, v in vars(back).items() if k != "payload"} == \
               {k: v for k, v in vars(ev).items() if k != "payload"}
    assert list(loads(dumps(events[0])).payload["conditions"]) == ["@", "T"]


def test_writer_and_reader(tmp_path, t0):
    path = tmp_path / "log" / "events.jsonl"
    with EventLogWriter(path, flush_every=1) as w:
        for i in range(3):
            w.write(TradeEvent(timestamp=t0 + timedelta(seconds=i), symbol="SPY",
                               source="t", price=100 + i))
    prices = [ev.price for ev in read_events(path)]
    assert prices == [100, 101, 102]


def test_reader_reports_bad_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"type": "TradeEvent"}\n')
    with pytest.raises(ValueError, match="bad.jsonl:1"):
        list(read_events(path))


def test_replay_feed_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        ReplayFeed(tmp_path / "nope.jsonl")


@pytest.mark.parametrize("method", ["static", "baseline", "risk_parity", "hrp", "signal"])
def test_live_and_replay_produce_identical_state(tmp_path, method):
    """Même moteur, source différente -> même état (README §43)."""
    log = tmp_path / "events.jsonl"

    live = Engine(_config(storage={"event_log": str(log)}, allocation={"method": method}))
    live_state = asyncio.run(live.run())

    replay = Engine(_config(feed={"provider": "replay", "replay_path": str(log)},
                            allocation={"method": method}))
    replay_state = asyncio.run(replay.run())

    assert replay.bus.published_count == live.bus.published_count
    assert replay_state == live_state
    for sym in live.features.symbols():
        assert replay.features.snapshot(sym) == live.features.snapshot(sym)
    assert [b.close for b in replay.bars] == [b.close for b in live.bars]
    for sym in live.features.symbols():
        assert replay.models.regimes(sym) == live.models.regimes(sym)
        assert replay.models.signals(sym) == live.models.signals(sym)
    assert any(live.models.regimes(sym) for sym in live.features.symbols())
    assert replay.risk_report == live.risk_report
    assert replay.safety.status == live.safety.status
    assert list(replay.decisions) == list(live.decisions)
    assert list(replay.alerts) == list(live.alerts)
    assert live.decisions, "une décision par intervalle"
    assert replay.integrity.counts == live.integrity.counts
    assert replay.integrity.scores() == live.integrity.scores()
    if method != "static":
        assert live.last_allocation is not None
        assert replay.last_allocation == live.last_allocation
