import asyncio

from trading_engine.data.event_bus import EventBus
from trading_engine.data.events import EventType, QuoteEvent, TradeEvent


def _trade(t0):
    return TradeEvent(timestamp=t0, symbol="SPY", source="t", price=100.0)


def test_dispatch_by_type_and_wildcard(t0):
    bus = EventBus()
    seen = []

    @bus.subscribe(EventType.TRADE)
    async def on_trade(ev):
        seen.append(("trade", ev.symbol))

    bus.subscribe("*", lambda ev: seen.append(("all", ev.event_type.value)))

    asyncio.run(bus.publish(_trade(t0)))
    asyncio.run(bus.publish(QuoteEvent(timestamp=t0, symbol="SPY", source="t", bid=1, ask=2)))

    assert seen == [("trade", "SPY"), ("all", "trade"), ("all", "quote")]
    assert bus.published_count == 2


def test_failing_handler_does_not_block_others(t0):
    bus = EventBus()
    seen = []

    def boom(ev):
        raise RuntimeError("boom")

    bus.subscribe(EventType.TRADE, boom)
    bus.subscribe(EventType.TRADE, lambda ev: seen.append(ev))

    asyncio.run(bus.publish(_trade(t0)))

    assert len(seen) == 1
    assert bus.error_count == 1


def test_unsubscribe(t0):
    bus = EventBus()
    seen = []
    handler = bus.subscribe(EventType.TRADE, lambda ev: seen.append(ev))
    bus.unsubscribe(EventType.TRADE, handler)
    asyncio.run(bus.publish(_trade(t0)))
    assert seen == []
