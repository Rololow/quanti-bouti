import asyncio
import json
from datetime import datetime, timezone

import pytest

from trading_engine.data.alpaca_feed import (
    AlpacaFatalError,
    AlpacaMarketFeed,
)
from trading_engine.data.events import BarEvent, QuoteEvent, TradeEvent

NOW = datetime(2026, 1, 5, 15, 0, 1, tzinfo=timezone.utc)

CONNECTED = [{"T": "success", "msg": "connected"}]
AUTHENTICATED = [{"T": "success", "msg": "authenticated"}]
SUBSCRIBED = [{"T": "subscription", "trades": ["SPY"], "bars": ["SPY"]}]
TRADE = {"T": "t", "S": "SPY", "i": 1, "x": "V", "p": 650.2, "s": 100,
         "c": ["@"], "t": "2026-01-05T15:00:00.123456789Z", "z": "B"}
QUOTE = {"T": "q", "S": "SPY", "bx": "V", "bp": 650.1, "bs": 2, "ax": "V",
         "ap": 650.3, "as": 3, "c": ["R"], "t": "2026-01-05T15:00:00.5Z", "z": "B"}
BAR = {"T": "b", "S": "SPY", "o": 650, "h": 651, "l": 649, "c": 650.5,
       "v": 1200, "n": 42, "vw": 650.4, "t": "2026-01-05T14:59:00Z"}

HANG = object()  # le serveur ne répond plus : recv() bloque


class FakeWebSocket:
    def __init__(self, script, pong=True):
        self.script = list(script)
        self.sent = []
        self.pong = pong
        self.pings = 0
        self.closed = False

    async def recv(self):
        if not self.script:
            raise ConnectionError("server closed the connection")
        item = self.script.pop(0)
        if item is HANG:
            await asyncio.sleep(3600)
        if isinstance(item, Exception):
            raise item
        return json.dumps(item)

    async def send(self, data):
        self.sent.append(json.loads(data))

    async def ping(self):
        self.pings += 1
        fut = asyncio.get_running_loop().create_future()
        if self.pong:
            fut.set_result(0.0)
        return fut

    async def close(self):
        self.closed = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.closed = True


class FakeServer:
    """Une FakeWebSocket par connexion successive."""

    def __init__(self, *sockets):
        self.sockets = list(sockets)
        self.urls = []

    def __call__(self, url):
        self.urls.append(url)
        if not self.sockets:
            raise ConnectionError("no more connections")
        return self.sockets.pop(0)


def make_feed(server, sleeps=None, **kwargs):
    async def fake_sleep(delay):
        if sleeps is not None:
            sleeps.append(delay)

    params = dict(key_id="key", secret_key="secret", connect=server, sleep=fake_sleep,
                  clock=lambda: NOW, heartbeat_interval=0.01, heartbeat_timeout=0.01,
                  handshake_timeout=0.05)
    params.update(kwargs)
    return AlpacaMarketFeed(["SPY"], **params)


async def collect(feed):
    return [ev async for ev in feed]


def test_handshake_subscription_and_parsing():
    ws = FakeWebSocket([CONNECTED, AUTHENTICATED, SUBSCRIBED + [TRADE, QUOTE], [BAR]])
    feed = make_feed(FakeServer(ws), max_events=3, quotes=True)

    events = asyncio.run(collect(feed))

    assert ws.sent[0] == {"action": "auth", "key": "key", "secret": "secret"}
    assert ws.sent[1] == {"action": "subscribe", "trades": ["SPY"], "quotes": ["SPY"], "bars": ["SPY"]}

    trade, quote, bar = events
    assert isinstance(trade, TradeEvent)
    assert trade.price == 650.2 and trade.size == 100
    assert trade.timestamp == datetime(2026, 1, 5, 15, 0, 0, 123456, tzinfo=timezone.utc)
    assert trade.received_at == NOW
    assert trade.payload["exchange"] == "V"

    assert isinstance(quote, QuoteEvent)
    assert (quote.bid, quote.ask, quote.ask_size) == (650.1, 650.3, 3)

    assert isinstance(bar, BarEvent)
    assert bar.timeframe == "1m"
    assert bar.end == datetime(2026, 1, 5, 15, 0, tzinfo=timezone.utc)
    assert bar.payload["correction"] is False

    assert feed.status.events_emitted == 3
    assert feed.status.last_latency is not None


def test_url_from_data_feed():
    server = FakeServer()
    feed = make_feed(server, data_feed="sip")
    assert feed.url == "wss://stream.data.alpaca.markets/v2/sip"


def test_reconnects_after_disconnect_with_backoff():
    first = FakeWebSocket([CONNECTED, AUTHENTICATED, [TRADE]])  # puis coupure
    second = FakeWebSocket([CONNECTED, AUTHENTICATED, [TRADE]])
    sleeps = []
    feed = make_feed(FakeServer(first, second), sleeps=sleeps, max_events=2)

    events = asyncio.run(collect(feed))

    assert len(events) == 2
    assert feed.status.reconnect_count == 1
    assert len(sleeps) == 1
    assert "ConnectionError" in feed.status.last_error


def test_backoff_grows_and_is_capped():
    feed = make_feed(FakeServer(), backoff_initial=1, backoff_max=8)
    delays = [feed.backoff_delay(a) for a in range(6)]
    assert all(0.5 <= d <= 1 for d in delays[:1])
    assert all(d <= 8 for d in delays)
    assert delays[5] >= 4  # plafonné à 8, jitter >= 50 %


def test_auth_failure_is_fatal():
    ws = FakeWebSocket([CONNECTED, [{"T": "error", "code": 402, "msg": "auth failed"}]])
    sleeps = []
    feed = make_feed(FakeServer(ws), sleeps=sleeps)

    with pytest.raises(AlpacaFatalError) as info:
        asyncio.run(collect(feed))

    assert info.value.code == 402
    assert sleeps == []  # pas de boucle de reconnexion


def test_connection_limit_is_retried():
    limited = FakeWebSocket([CONNECTED, [{"T": "error", "code": 406, "msg": "connection limit exceeded"}]])
    ok = FakeWebSocket([CONNECTED, AUTHENTICATED, [TRADE]])
    feed = make_feed(FakeServer(limited, ok), max_events=1)

    events = asyncio.run(collect(feed))

    assert len(events) == 1
    assert feed.status.reconnect_count == 1


def test_heartbeat_ping_keeps_quiet_connection_alive():
    ws = FakeWebSocket([CONNECTED, AUTHENTICATED, HANG, [TRADE]], pong=True)
    feed = make_feed(FakeServer(ws), max_events=1)

    events = asyncio.run(collect(feed))

    assert len(events) == 1
    assert ws.pings == 1
    assert feed.status.reconnect_count == 0


def test_missing_pong_triggers_reconnect():
    dead = FakeWebSocket([CONNECTED, AUTHENTICATED, HANG], pong=False)
    ok = FakeWebSocket([CONNECTED, AUTHENTICATED, [TRADE]])
    feed = make_feed(FakeServer(dead, ok), max_events=1)

    events = asyncio.run(collect(feed))

    assert len(events) == 1
    assert feed.status.reconnect_count == 1
    assert "HeartbeatTimeout" in feed.status.last_error


def test_invalid_message_is_skipped():
    bad_trade = dict(TRADE, p=0)
    ws = FakeWebSocket([CONNECTED, AUTHENTICATED, [bad_trade, {"T": "t", "S": "SPY"}, TRADE]])
    feed = make_feed(FakeServer(ws), max_events=1)

    events = asyncio.run(collect(feed))

    assert len(events) == 1
    assert feed.status.parse_errors == 2


def test_updated_bar_flagged_as_correction():
    feed = make_feed(FakeServer())
    bar = feed.parse_message(dict(BAR, T="u"))
    assert bar.payload["correction"] is True


def test_from_env_requires_credentials(monkeypatch):
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    with pytest.raises(RuntimeError, match="APCA_API_KEY_ID"):
        AlpacaMarketFeed.from_env(["SPY"])


def test_repr_hides_secret():
    feed = make_feed(FakeServer())
    assert "secret" not in repr(feed)
