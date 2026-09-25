from datetime import timedelta

import pytest

from trading_engine.data.bar_builder import BarBuilder
from trading_engine.data.events import TradeEvent
from trading_engine.timeutils import parse_timeframe


def _trade(ts, price, size=1.0):
    return TradeEvent(timestamp=ts, symbol="SPY", source="t", price=price, size=size)


def test_parse_timeframe():
    assert parse_timeframe("5m") == timedelta(minutes=5)
    assert parse_timeframe("1d") == timedelta(days=1)
    with pytest.raises(ValueError):
        parse_timeframe("5x")


def test_bar_ohlcv_emitted_on_next_bucket(t0):
    bb = BarBuilder(["5m"])
    assert bb.on_trade(_trade(t0, 100, 1)) == []
    assert bb.on_trade(_trade(t0 + timedelta(minutes=1), 105, 2)) == []
    assert bb.on_trade(_trade(t0 + timedelta(minutes=2), 95, 3)) == []
    assert bb.on_trade(_trade(t0 + timedelta(minutes=3), 101, 4)) == []

    bars = bb.on_trade(_trade(t0 + timedelta(minutes=5), 110))
    assert len(bars) == 1
    bar = bars[0]
    assert (bar.open, bar.high, bar.low, bar.close) == (100, 105, 95, 101)
    assert bar.volume == 10
    assert bar.trade_count == 4
    assert bar.timestamp == t0
    assert bar.end == t0 + timedelta(minutes=5)


def test_slow_timeframe_not_emitted_on_fast_bucket(t0):
    bb = BarBuilder(["5m", "1h"])
    bb.on_trade(_trade(t0, 100))
    bars = bb.on_trade(_trade(t0 + timedelta(minutes=5), 101))
    assert [b.timeframe for b in bars] == ["5m"]


def test_flush(t0):
    bb = BarBuilder(["5m"])
    bb.on_trade(_trade(t0, 100))
    assert bb.flush(t0 + timedelta(minutes=4)) == []
    assert len(bb.flush(t0 + timedelta(minutes=5))) == 1
    assert bb.flush() == []
