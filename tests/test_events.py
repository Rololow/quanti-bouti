from datetime import datetime, timedelta

import pytest

from trading_engine.data.events import EventType, FundamentalEvent, QuoteEvent, TradeEvent


def test_trade_event_fields(t0):
    ev = TradeEvent(timestamp=t0, symbol="SPY", source="test", price=100.0, size=5)
    assert ev.event_type is EventType.TRADE
    assert ev.payload == {}
    with pytest.raises(Exception):
        ev.price = 1.0  # immuable


def test_naive_timestamp_rejected():
    with pytest.raises(ValueError):
        TradeEvent(timestamp=datetime(2026, 1, 1), symbol="SPY", source="t", price=1.0)


def test_payload_is_read_only(t0):
    ev = TradeEvent(timestamp=t0, symbol="SPY", source="t", price=1.0, payload={"a": 1})
    with pytest.raises(TypeError):
        ev.payload["a"] = 2


def test_non_positive_price_rejected(t0):
    with pytest.raises(ValueError):
        TradeEvent(timestamp=t0, symbol="SPY", source="t", price=0.0)


def test_quote_mid_and_spread(t0):
    q = QuoteEvent(timestamp=t0, symbol="SPY", source="t", bid=99.0, ask=101.0)
    assert q.mid == 100.0
    assert q.spread == 2.0


def test_fundamental_event_availability(t0):
    ev = FundamentalEvent(
        timestamp=t0, symbol="AAPL", source="t", name="eps", value=1.2,
        period="Q2", available_at=t0 + timedelta(days=1),
    )
    assert not ev.is_available(t0)
    assert ev.is_available(t0 + timedelta(days=1))
