"""Broker Alpaca **paper** : compte, positions, ordres et fills réels (simulés
côté Alpaca, sans argent réel).

Verrou : ce module refuse toute URL autre que `paper-api.alpaca.markets`.
Aucun chemin vers un compte réel n'existe dans ce code.

- `AlpacaTradingClient`  : REST (compte, positions, ordres) ;
- `AlpacaPaperBroker`     : envoie les OrderProposal validés par les hard
  controls, annule les ordres **du moteur uniquement** (jamais les autres
  ordres du compte), gère l'expiration ;
- `AlpacaTradeUpdatesFeed`: flux `trade_updates` -> `OrderUpdateEvent`,
  fusionné avec le flux de marché et enregistré dans le journal (le replay
  rejoue les fills sans broker).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable, Mapping

from trading_engine.data.alpaca_feed import KEY_ENV, SECRET_ENV
from trading_engine.data.events import OrderUpdateEvent
from trading_engine.data.market_feed import MarketFeed
from trading_engine.execution.orders import OrderProposal
from trading_engine.timeutils import parse_rfc3339, utcnow

logger = logging.getLogger(__name__)

PAPER_URL = "https://paper-api.alpaca.markets"
PAPER_STREAM_URL = "wss://paper-api.alpaca.markets/stream"


PAPER_HOST = "paper-api.alpaca.markets"


def _assert_paper(url: str) -> None:
    # Hôte exact (un préfixe laisserait passer paper-api.alpaca.markets.exemple.com).
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ("https", "wss") or parsed.hostname != PAPER_HOST or parsed.port not in (None, 443):
        raise ValueError(f"only the Alpaca PAPER endpoint is allowed, got {url!r}")


class BrokerError(Exception):
    pass


def _http(method: str, url: str, headers: Mapping[str, str], body: Any = None) -> Any:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method, headers={**headers, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
            return json.loads(raw.decode("utf-8")) if raw else None
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise BrokerError(f"HTTP {exc.code} {method} {url.split('?')[0]}: {detail}") from None


@dataclass(frozen=True)
class BrokerPosition:
    symbol: str
    quantity: float
    avg_price: float
    price: float | None


@dataclass(frozen=True)
class BrokerAccount:
    cash: float
    equity: float
    status: str
    trading_blocked: bool


class AlpacaTradingClient:
    def __init__(
        self,
        *,
        key_id: str,
        secret_key: str,
        base_url: str = PAPER_URL,
        http: Callable[..., Any] = _http,
    ) -> None:
        _assert_paper(base_url)
        if not key_id or not secret_key:
            raise ValueError("Alpaca API key id and secret are required")
        self.base_url = base_url.rstrip("/")
        self._headers = {"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret_key}
        self._key_id, self._secret = key_id, secret_key
        self._http = http

    def __repr__(self) -> str:  # ne jamais afficher les clés
        return f"AlpacaTradingClient({self.base_url!r})"

    @classmethod
    def from_env(cls, **kwargs: Any) -> "AlpacaTradingClient":
        key, secret = os.environ.get(KEY_ENV), os.environ.get(SECRET_ENV)
        if not key or not secret:
            raise RuntimeError(f"Alpaca credentials missing: set {KEY_ENV} and {SECRET_ENV}")
        return cls(key_id=key, secret_key=secret, **kwargs)

    def credentials(self) -> tuple[str, str]:
        return self._key_id, self._secret

    def _call(self, method: str, path: str, body: Any = None) -> Any:
        return self._http(method, f"{self.base_url}{path}", self._headers, body)

    def account(self) -> BrokerAccount:
        a = self._call("GET", "/v2/account")
        return BrokerAccount(float(a["cash"]), float(a["equity"]), str(a.get("status", "")),
                             bool(a.get("trading_blocked", False)))

    def positions(self) -> list[BrokerPosition]:
        out = []
        for p in self._call("GET", "/v2/positions") or []:
            qty = float(p["qty"])
            if p.get("side") == "short" and qty > 0:
                qty = -qty
            price = p.get("current_price")
            out.append(BrokerPosition(p["symbol"], qty, float(p["avg_entry_price"]),
                                      None if price is None else float(price)))
        return out

    def submit_limit_order(self, *, symbol: str, quantity: float, limit_price: float,
                           client_order_id: str, time_in_force: str = "day") -> dict:
        qty = abs(quantity)
        body = {
            "symbol": symbol, "qty": f"{qty:g}", "side": "buy" if quantity > 0 else "sell",
            "type": "limit", "time_in_force": time_in_force, "limit_price": f"{limit_price:.2f}",
            "client_order_id": client_order_id,
        }
        return self._call("POST", "/v2/orders", body)

    def cancel_order(self, broker_order_id: str) -> None:
        self._call("DELETE", f"/v2/orders/{broker_order_id}")


def client_order_id(order: OrderProposal) -> str:
    """Identifiant déterministe (le replay retrouve l'ordre à partir des plans recalculés)."""
    return f"qb-{order.decision_id or 'na'}-{order.symbol}"[:128]


class AlpacaPaperBroker:
    """Envoie et annule les ordres du moteur sur le compte paper."""

    def __init__(self, client: AlpacaTradingClient, *, time_in_force: str = "day",
                 forget_after: float = 300.0) -> None:
        self.client = client
        self.time_in_force = time_in_force
        # Sans mise à jour terminale (flux coupé) `forget_after` s après
        # l'expiration, l'ordre est oublié : le rapprochement corrige l'état.
        self.forget_after = forget_after
        self.forgotten = 0
        self.working: dict[str, tuple[OrderProposal, str]] = {}    # client_id -> (ordre, id broker)
        self.cancel_requested: set[str] = set()
        self.errors: list[str] = []

    async def submit(self, order: OrderProposal) -> str | None:
        cid = client_order_id(order)
        try:
            resp = await asyncio.to_thread(
                self.client.submit_limit_order, symbol=order.symbol, quantity=order.quantity,
                limit_price=order.limit_price, client_order_id=cid, time_in_force=self.time_in_force,
            )
        except Exception as exc:
            self.errors.append(f"{order.symbol}: {exc}")
            logger.warning("order rejected by broker: %s", exc)
            return None
        self.working[cid] = (order, str(resp.get("id", "")))
        return cid

    async def cancel(self, cid: str) -> None:
        entry = self.working.get(cid)
        if entry is None or not entry[1] or cid in self.cancel_requested:
            return
        self.cancel_requested.add(cid)
        try:
            await asyncio.to_thread(self.client.cancel_order, entry[1])
        except Exception as exc:          # déjà exécuté / déjà annulé : la mise à jour arrivera par le flux
            logger.info("cancel %s: %s", cid, exc)

    async def cancel_all(self) -> None:
        """Annule les ordres du moteur (pas les autres ordres du compte)."""
        for cid in list(self.working):
            await self.cancel(cid)

    async def expire(self, now: datetime) -> None:
        for cid, (order, _) in list(self.working.items()):
            if now >= order.expires_at:
                await self.cancel(cid)
                if (now - order.expires_at).total_seconds() >= self.forget_after:
                    logger.warning("no terminal update for %s, forgotten", cid)
                    self.working.pop(cid, None)
                    self.cancel_requested.discard(cid)
                    self.forgotten += 1

    def on_update(self, ev: OrderUpdateEvent) -> None:
        if ev.terminal:
            self.working.pop(ev.client_order_id, None)
            self.cancel_requested.discard(ev.client_order_id)


def parse_trade_update(message: Mapping[str, Any], clock: Callable[[], datetime] = utcnow) -> OrderUpdateEvent | None:
    if message.get("stream") != "trade_updates":
        return None
    data = message.get("data") or {}
    order = data.get("order") or {}
    cid = order.get("client_order_id") or ""
    if not cid.startswith("qb-"):
        return None                     # ordre d'une autre origine : ignoré
    ts_raw = data.get("timestamp") or order.get("updated_at")
    ts = parse_rfc3339(ts_raw) if ts_raw else clock()

    def num(v):
        return None if v in (None, "") else float(v)

    return OrderUpdateEvent(
        timestamp=ts, received_at=clock(), symbol=order.get("symbol"), source="alpaca_trading",
        client_order_id=cid, broker_order_id=str(order.get("id", "")), update=str(data.get("event", "")),
        side=str(order.get("side", "")), fill_qty=num(data.get("qty")) or 0.0, fill_price=num(data.get("price")),
        filled_qty=num(order.get("filled_qty")) or 0.0, filled_avg_price=num(order.get("filled_avg_price")),
        payload={"status": order.get("status"), "execution_id": data.get("execution_id")},
    )


class AlpacaTradeUpdatesFeed(MarketFeed):
    """Flux websocket `trade_updates` du compte paper -> OrderUpdateEvent."""

    def __init__(
        self,
        *,
        key_id: str,
        secret_key: str,
        url: str = PAPER_STREAM_URL,
        connect: Callable[..., Any] | None = None,
        sleep: Callable[[float], Any] = asyncio.sleep,
        clock: Callable[[], datetime] = utcnow,
        max_reconnects: int | None = None,
    ) -> None:
        _assert_paper(url)
        self.url = url
        self._key, self._secret = key_id, secret_key
        self._connect = connect or _ws_connect
        self._sleep = sleep
        self._clock = clock
        self.max_reconnects = max_reconnects
        self.reconnects = 0

    async def __aiter__(self) -> AsyncIterator[OrderUpdateEvent]:
        attempt = 0
        while self.max_reconnects is None or self.reconnects <= self.max_reconnects:
            try:
                async with self._connect(self.url) as ws:
                    await ws.send(json.dumps({"action": "auth", "key": self._key, "secret": self._secret}))
                    auth = _decode(await ws.recv())
                    if (auth.get("data") or {}).get("status") != "authorized":
                        raise BrokerError(f"trade_updates auth failed: {auth}")
                    await ws.send(json.dumps({"action": "listen", "data": {"streams": ["trade_updates"]}}))
                    attempt = 0
                    while True:
                        ev = parse_trade_update(_decode(await ws.recv()), self._clock)
                        if ev is not None:
                            yield ev
            except BrokerError:
                raise
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("trade_updates interrupted: %s", exc)
            self.reconnects += 1
            attempt += 1
            await self._sleep(min(60.0, 2 ** attempt) * (0.5 + random.random() / 2))


def _decode(frame: Any) -> dict:
    if isinstance(frame, (bytes, bytearray)):
        frame = frame.decode("utf-8")
    data = json.loads(frame)
    return data if isinstance(data, dict) else {}


def _ws_connect(url: str) -> Any:
    import websockets

    return websockets.connect(url, ping_interval=20, ping_timeout=20)
