"""Alpaca paper : historique de démarrage, synchronisation du compte,
broker paper et flux trade_updates (tout avec des faux, sans réseau)."""

import asyncio
import dataclasses
import json
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlparse

import pytest

from trading_engine.config import load_config
from trading_engine.data.alpaca_history import AlpacaHistoricalClient
from trading_engine.data.events import BarEvent, OrderUpdateEvent, PortfolioEvent
from trading_engine.engine import Engine
from trading_engine.execution.alpaca_trading import (
    AlpacaPaperBroker,
    AlpacaTradeUpdatesFeed,
    AlpacaTradingClient,
    BrokerAccount,
    BrokerError,
    BrokerPosition,
    client_order_id,
    parse_trade_update,
)
from trading_engine.execution.orders import OrderProposal
from trading_engine.storage.event_log import dumps, loads, read_events

NOW = datetime(2026, 1, 6, 15, 7, tzinfo=timezone.utc)


# ------------------------------------------------------------------ historique

def _row(t, c):
    return {"t": t.isoformat().replace("+00:00", "Z"), "o": c, "h": c + 1, "l": c - 1, "c": c,
            "v": 1000, "n": 10, "vw": c}


def test_history_paginates_and_skips_open_bars():
    calls = []
    t = NOW.replace(minute=0)

    def get(url, headers):
        q = parse_qs(urlparse(url).query)
        calls.append(q)
        assert headers["APCA-API-KEY-ID"] == "k" and q["timeframe"] == ["1Hour"]
        if "page_token" not in q:
            return {"bars": {"SPY": [_row(t - timedelta(hours=2), 100)]}, "next_page_token": "p2"}
        # la barre 15:00-16:00 est encore ouverte à 15:07 : exclue
        return {"bars": {"SPY": [_row(t - timedelta(hours=1), 101), _row(t, 102)]}, "next_page_token": None}

    client = AlpacaHistoricalClient(key_id="k", secret_key="SECRET-XYZ", http_get=get, clock=lambda: NOW)
    bars = client.bars(["SPY"], "1h", t - timedelta(days=1), t)
    assert len(calls) == 2 and calls[1]["page_token"] == ["p2"]
    assert [b.close for b in bars] == [100, 101]
    assert all(b.payload["warmup"] and b.end <= NOW and b.source == "alpaca_history" for b in bars)
    assert "SECRET" not in repr(client)


def test_warmup_bars_sorted_and_failures_tolerated():
    def get(url, headers):
        q = parse_qs(urlparse(url).query)
        if q["timeframe"] == ["1Day"]:
            raise OSError("network down")
        base = NOW.replace(minute=0, second=0) - timedelta(hours=3)
        return {"bars": {s: [_row(base + timedelta(hours=i), 100 + i) for i in range(3)]
                         for s in q["symbols"][0].split(",")}}

    client = AlpacaHistoricalClient(key_id="k", secret_key="s", http_get=get, clock=lambda: NOW)
    bars = asyncio.run(client.warmup_bars(["QQQ", "SPY"], {"1h": 1, "1d": 30}))
    assert len(bars) == 6 and {b.timeframe for b in bars} == {"1h"}
    assert [b.end for b in bars] == sorted(b.end for b in bars)
    assert [b.symbol for b in bars[:2]] == ["QQQ", "SPY"]


def test_warmup_bars_initialize_features_without_decisions(t0):
    e = Engine(load_config())
    bars = []
    for i in range(40):
        for sym in ("SPY", "QQQ"):
            c = 100 + i + (0.5 if sym == "QQQ" else 0) + (i % 3)
            bars.append(BarEvent(timestamp=t0 + timedelta(hours=i), end=t0 + timedelta(hours=i + 1),
                                 received_at=t0 + timedelta(days=5), symbol=sym, source="alpaca_history",
                                 timeframe="1h", open=c, high=c + 1, low=c - 1, close=c, volume=1000,
                                 payload={"warmup": True}))
    bad = dataclasses.replace(bars[0], high=1.0, low=2.0)
    for bar in bars + [bad]:
        asyncio.run(e._ingest(bar))
    assert e.features.snapshot("SPY") and e.portfolio.prices["SPY"] == bars[-2].close
    assert not e.decisions and not e.alerts and e.bus.published_count == 0
    assert e.integrity.counts == {"INCONSISTENT_BAR": 1}
    # le retard n'est pas appliqué à l'historique (barres anciennes, reçues maintenant)
    assert e.integrity.stale_symbols() == []


# ------------------------------------------------------------------ verrou paper

def test_trading_client_refuses_live_endpoint():
    with pytest.raises(ValueError, match="PAPER"):
        AlpacaTradingClient(key_id="k", secret_key="s", base_url="https://api.alpaca.markets")
    with pytest.raises(ValueError, match="PAPER"):
        AlpacaTradeUpdatesFeed(key_id="k", secret_key="s", url="wss://api.alpaca.markets/stream")
    with pytest.raises(ValueError, match="PAPER"):
        AlpacaTradingClient(key_id="k", secret_key="s", base_url="https://paper-api.alpaca.markets.evil.com")
    with pytest.raises(ValueError, match="PAPER"):
        AlpacaTradingClient(key_id="k", secret_key="s", base_url="http://paper-api.alpaca.markets")
    with pytest.raises(ValueError, match="PAPER"):
        AlpacaTradingClient(key_id="k", secret_key="s", base_url="https://user@evil.com/paper-api.alpaca.markets")
    AlpacaTradingClient(key_id="k", secret_key="s", base_url="https://paper-api.alpaca.markets/")
    assert "secret" not in repr(AlpacaTradingClient(key_id="key", secret_key="secret"))


def test_trading_client_requests():
    sent = []

    def http(method, url, headers, body=None):
        sent.append((method, url, body))
        if url.endswith("/v2/account"):
            return {"cash": "1000.5", "equity": "2000", "status": "ACTIVE", "trading_blocked": False}
        if url.endswith("/v2/positions"):
            return [{"symbol": "SPY", "qty": "3", "side": "long", "avg_entry_price": "600", "current_price": "610"},
                    {"symbol": "TLT", "qty": "2", "side": "short", "avg_entry_price": "90", "current_price": None}]
        return {"id": "b1"}

    c = AlpacaTradingClient(key_id="k", secret_key="s", http=http)
    assert c.account() == BrokerAccount(1000.5, 2000.0, "ACTIVE", False)
    assert c.positions() == [BrokerPosition("SPY", 3.0, 600.0, 610.0), BrokerPosition("TLT", -2.0, 90.0, None)]
    c.submit_limit_order(symbol="SPY", quantity=-2.5, limit_price=601.234, client_order_id="qb-x")
    method, url, body = sent[-1]
    assert (method, url) == ("POST", "https://paper-api.alpaca.markets/v2/orders")
    assert body == {"symbol": "SPY", "qty": "2.5", "side": "sell", "type": "limit", "time_in_force": "day",
                    "limit_price": "601.23", "client_order_id": "qb-x"}


# ------------------------------------------------------------------ broker

class FakeClient:
    """Compte paper simulé : ordres exécutés immédiatement (moitié puis reste)."""

    def __init__(self, cash=100_000.0, positions=None, sink=None, status="ACTIVE", reject=()):
        self.cash = cash
        self.positions_ = dict(positions or {})     # sym -> [qty, avg]
        self.prices = {}
        self.sink = sink                            # liste où poser les trade_updates
        self.status = status
        self.reject = set(reject)
        self.submitted, self.cancelled = [], []
        self._n = 0

    def credentials(self):
        return "k", "s"

    def account(self):
        return BrokerAccount(self.cash, self.cash, self.status, False)

    def positions(self):
        return [BrokerPosition(s, q, a, self.prices.get(s)) for s, (q, a) in sorted(self.positions_.items()) if q]

    def submit_limit_order(self, *, symbol, quantity, limit_price, client_order_id, time_in_force="day"):
        if symbol in self.reject:
            raise BrokerError("HTTP 403 insufficient buying power")
        self._n += 1
        bid = f"b{self._n}"
        self.submitted.append((client_order_id, symbol, quantity, limit_price))
        if self.sink is not None:
            ts = None                               # horodaté à la réception par le test
            side = "buy" if quantity > 0 else "sell"
            first = round(abs(quantity) / 2, 6)
            for update, qty, filled in (("partial_fill", first, first), ("fill", abs(quantity) - first, abs(quantity))):
                self._apply(symbol, qty if quantity > 0 else -qty, limit_price)
                self.sink.append(dict(stream="trade_updates", data=dict(
                    event=update, qty=str(qty), price=str(limit_price), timestamp=ts,
                    order=dict(id=bid, client_order_id=client_order_id, symbol=symbol, side=side,
                               filled_qty=str(filled), filled_avg_price=str(limit_price)))))
        return {"id": bid}

    def _apply(self, sym, qty, price):
        q, a = self.positions_.get(sym, [0.0, 0.0])
        nq = q + qty
        self.positions_[sym] = [nq, (q * a + qty * price) / nq if qty > 0 and nq else a]
        self.cash -= qty * price
        self.prices[sym] = price

    def cancel_order(self, broker_id):
        self.cancelled.append(broker_id)


def _order(sym="SPY", qty=10.0, ts=NOW, did="D20260106T150000-00001"):
    return OrderProposal(sym, qty, 100.0, ts, did, duration=timedelta(minutes=30))


def _update(cid, event, **kw):
    order = {"id": "b1", "client_order_id": cid, "symbol": "SPY", "side": "buy",
             "filled_qty": kw.pop("filled", "0"), "filled_avg_price": kw.pop("avg", None)}
    return {"stream": "trade_updates", "data": {"event": event, "timestamp": "2026-01-06T15:10:00Z",
                                                "order": order, **kw}}


def test_broker_submits_cancels_only_own_orders_and_expires():
    client = FakeClient()
    broker = AlpacaPaperBroker(client)
    o = _order()
    cid = asyncio.run(broker.submit(o))
    assert cid == client_order_id(o) == "qb-D20260106T150000-00001-SPY"
    asyncio.run(broker.cancel_all())
    asyncio.run(broker.cancel_all())
    assert client.cancelled == ["b1"]                      # pas d'annulation répétée
    broker.on_update(parse_trade_update(_update(cid, "canceled")))
    assert broker.working == {}
    # expiration puis oubli si aucune mise à jour terminale n'arrive
    cid2 = asyncio.run(broker.submit(_order(did="D2")))
    asyncio.run(broker.expire(NOW + timedelta(minutes=29)))
    assert cid2 in broker.working and len(client.cancelled) == 1
    asyncio.run(broker.expire(NOW + timedelta(minutes=31)))
    assert cid2 in broker.working and len(client.cancelled) == 2
    asyncio.run(broker.expire(NOW + timedelta(minutes=36)))
    assert broker.working == {} and broker.forgotten == 1


def test_broker_rejection_is_recorded():
    broker = AlpacaPaperBroker(FakeClient(reject={"SPY"}))
    assert asyncio.run(broker.submit(_order())) is None
    assert broker.working == {} and "insufficient" in broker.errors[0]


def test_parse_trade_update():
    cid = "qb-D1-SPY"
    ev = parse_trade_update(_update(cid, "partial_fill", qty="4", price="100.5", filled="4", avg="100.5"))
    assert (ev.update, ev.fill_qty, ev.fill_price, ev.filled_qty, ev.terminal) == \
           ("partial_fill", 4.0, 100.5, 4.0, False)
    assert ev.timestamp == datetime(2026, 1, 6, 15, 10, tzinfo=timezone.utc)
    assert parse_trade_update(_update(cid, "fill", qty="6", price="100", filled="10")).terminal
    assert parse_trade_update(_update(cid, "rejected")).terminal
    assert parse_trade_update(_update("manual-order", "fill")) is None       # pas un ordre du moteur
    assert parse_trade_update({"stream": "authorization", "data": {}}) is None
    back = loads(dumps(ev))
    assert back == ev


class FakeWS:
    def __init__(self, frames):
        self.frames = list(frames)
        self.sent = []

    async def send(self, msg):
        self.sent.append(json.loads(msg))

    async def recv(self):
        if not self.frames:
            raise ConnectionError("closed")
        return self.frames.pop(0)


def test_trade_updates_feed_auth_listen_and_reconnect():
    auth = json.dumps({"stream": "authorization", "data": {"status": "authorized", "action": "authenticate"}})
    sockets = [
        FakeWS([auth, json.dumps(_update("qb-D1-SPY", "new")).encode()]),
        FakeWS([auth.encode(), json.dumps(_update("qb-D1-SPY", "fill", qty="1", price="1", filled="1"))]),
    ]
    opened = []

    @asynccontextmanager
    async def connect(url):
        ws = sockets[len(opened)]
        opened.append(url)
        yield ws

    async def no_sleep(_):
        pass

    feed = AlpacaTradeUpdatesFeed(key_id="k", secret_key="s", connect=connect, sleep=no_sleep, max_reconnects=1)

    async def collect():
        return [ev async for ev in feed]

    events = asyncio.run(collect())
    assert [e.update for e in events] == ["new", "fill"]
    assert opened == ["wss://paper-api.alpaca.markets/stream"] * 2
    assert sockets[0].sent == [{"action": "auth", "key": "k", "secret": "s"},
                               {"action": "listen", "data": {"streams": ["trade_updates"]}}]


def test_trade_updates_feed_auth_failure_raises():
    @asynccontextmanager
    async def connect(url):
        yield FakeWS([json.dumps({"stream": "authorization", "data": {"status": "unauthorized"}})])

    feed = AlpacaTradeUpdatesFeed(key_id="k", secret_key="bad", connect=connect)

    async def collect():
        return [ev async for ev in feed]

    with pytest.raises(BrokerError, match="auth failed"):
        asyncio.run(collect())


# ------------------------------------------------------------------ moteur

def _cfg(**execution):
    cfg = load_config()
    return dataclasses.replace(
        cfg,
        engine=dataclasses.replace(cfg.engine, max_events=12000, report_every=0),
        allocation=dataclasses.replace(cfg.allocation, method="hrp"),
        risk=dataclasses.replace(cfg.risk, min_observations=5),
        decision=dataclasses.replace(cfg.decision, risk_aversion=50.0, holding_period="20d"),
        execution=dataclasses.replace(cfg.execution, **execution),
    )


def test_alpaca_paper_mode_requires_alpaca_feed():
    with pytest.raises(ValueError, match="alpaca_paper"):
        Engine(_cfg(mode="alpaca_paper"))


def _portfolio_event(kind, cash, positions, status="ACTIVE", blocked=False, ts=NOW):
    return PortfolioEvent(timestamp=ts, received_at=ts, symbol=None, source="alpaca_trading",
                          payload={"kind": kind, "cash": cash, "equity": cash, "status": status,
                                   "trading_blocked": blocked, "positions": positions})


def test_sync_adopts_broker_state_and_rebuilds_tax_lots():
    cfg = _cfg(mode="proposals")
    e = Engine(cfg)
    asyncio.run(e._ingest(_portfolio_event("sync", 5_000.0, {"SPY": [10, 600.0, 610.0], "GLD": [4, 300.0, None]})))
    assert e.portfolio.cash == 5_000.0
    assert {s: (p.quantity, p.avg_price) for s, p in e.portfolio.positions.items()} == \
           {"SPY": (10, 600.0), "GLD": (4, 300.0)}
    assert e.portfolio.prices["SPY"] == 610.0
    lots = e.tax.gains.lots
    assert set(lots) == {"SPY", "GLD"} and lots["SPY"][0].quantity == 10 and lots["SPY"][0].unit_cost == 600.0
    assert not [a for a in e.alerts if a.kind == "RECONCILIATION"]
    # rapprochement identique : rien ; écart : alerte + état broker adopté
    asyncio.run(e._ingest(_portfolio_event("reconcile", 5_000.0, {"SPY": [10, 600.0, 610.0], "GLD": [4, 300.0, None]})))
    assert not [a for a in e.alerts if a.kind == "RECONCILIATION"]
    asyncio.run(e._ingest(_portfolio_event("reconcile", 3_780.0, {"SPY": [12, 605.0, 610.0], "GLD": [4, 300.0, 305.0]})))
    rec = [a for a in e.alerts if a.kind == "RECONCILIATION"]
    assert len(rec) == 1 and "SPY" in rec[0].message
    assert e.portfolio.positions["SPY"].quantity == 12 and e.portfolio.cash == 3_780.0
    assert sum(lot.quantity for lot in lots["SPY"]) == 12
    assert e.safety.status.state.value != "HALTED"


def test_blocked_account_halts_engine():
    e = Engine(_cfg(mode="proposals"))
    asyncio.run(e._ingest(_portfolio_event("sync", 1_000.0, {}, status="ACCOUNT_UPDATED", blocked=True)))
    assert e.safety.status.state.value == "HALTED"
    assert "BROKER_ACCOUNT" in e.safety.status.reasons[0]


def _run_with_fake_alpaca(cfg, client):
    """Moteur simulé branché sur un faux compte paper. Les trade_updates du
    faux compte entrent comme ceux du flux websocket (journalisés)."""
    e = Engine(cfg)
    e.remote_broker = AlpacaPaperBroker(client)
    e.account_synced = False                 # comme en mode alpaca_paper : sync requise
    sink = client.sink

    orig_drain = e._drain_injected

    async def drain():
        while sink:
            msg = sink.pop(0)
            ev = parse_trade_update(msg, clock=lambda: e.integrity.last_event_time)
            e._injected.append(dataclasses.replace(ev, timestamp=e.integrity.last_event_time))
        await orig_drain()

    e._drain_injected = drain
    asyncio.run(e.run())
    return e


def test_engine_trades_on_fake_paper_account_and_replays(tmp_path):
    log = tmp_path / "events.jsonl"
    cfg = _cfg(mode="proposals")
    cfg = dataclasses.replace(cfg, storage=dataclasses.replace(cfg.storage, event_log=str(log)))
    client = FakeClient(cash=50_000.0, positions={"SPY": [20, 600.0]}, sink=[])
    live = _run_with_fake_alpaca(cfg, client)

    assert client.submitted and live.fills
    assert live.execution_feedback.records
    # l'état du moteur correspond au compte (le rapprochement ne trouve aucun écart)
    engine_pos = {s: round(p.quantity, 6) for s, p in live.portfolio.positions.items() if p.quantity}
    broker_pos = {s: round(q, 6) for s, (q, _) in client.positions_.items() if q}
    assert engine_pos == broker_pos
    assert live.portfolio.cash == pytest.approx(client.cash)
    assert not [a for a in live.alerts if a.kind == "RECONCILIATION"]
    assert live.taxes_outside_broker > 0                   # TOB due, non prélevée par le broker
    assert live.safety.status.state.value != "HALTED"

    types = {type(ev).__name__ for ev in read_events(log)}
    assert {"PortfolioEvent", "OrderUpdateEvent"} <= types

    # replay : aucun broker, fills et état du compte relus du journal
    replay_cfg = dataclasses.replace(
        cfg, execution=dataclasses.replace(cfg.execution, mode="alpaca_paper"),
        feed=dataclasses.replace(cfg.feed, provider="replay", replay_path=str(log)),
        storage=dataclasses.replace(cfg.storage, event_log=None))
    replay = Engine(replay_cfg)
    assert replay.remote_broker is None
    asyncio.run(replay.run())
    assert replay.fills == live.fills
    assert list(replay.plans) == list(live.plans)
    assert replay.snapshot() == live.snapshot()
    assert replay.execution_feedback.records == live.execution_feedback.records
