from datetime import timedelta

from trading_engine.data.events import QuoteEvent, TradeEvent
from trading_engine.data.market_state import MarketStateStore


def test_trade_then_quote(t0):
    store = MarketStateStore()
    store.update(TradeEvent(timestamp=t0, symbol="SPY", source="t", price=100.0))
    st = store.update(QuoteEvent(timestamp=t0 + timedelta(seconds=1), symbol="SPY",
                                 source="t", bid=99.9, ask=100.1))
    assert st.price == 100.0
    assert round(st.spread, 6) == 0.2
    assert store.prices() == {"SPY": 100.0}


def test_stale_event_ignored(t0):
    store = MarketStateStore()
    store.update(TradeEvent(timestamp=t0, symbol="SPY", source="t", price=100.0))
    st = store.update(TradeEvent(timestamp=t0 - timedelta(seconds=1), symbol="SPY",
                                 source="t", price=50.0))
    assert st.price == 100.0
