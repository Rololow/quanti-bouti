"""Flux de marché temps réel Alpaca (README §4, Phase 2).

Protocole (market data stream v2) :

    serveur : [{"T": "success", "msg": "connected"}]
    client  : {"action": "auth", "key": ..., "secret": ...}
    serveur : [{"T": "success", "msg": "authenticated"}]
    client  : {"action": "subscribe", "trades": [...], "quotes": [...], "bars": [...]}
    serveur : [{"T": "subscription", ...}] puis les données : T = t / q / b / u

Robustesse :

- reconnexion automatique avec backoff exponentiel + jitter ;
- heartbeat : si aucun message n'arrive pendant `heartbeat_interval`, on envoie
  un ping WebSocket ; sans pong sous `heartbeat_timeout`, la connexion est
  considérée morte et on se reconnecte. Un marché fermé (aucune donnée) ne
  provoque donc pas de reconnexion tant que le serveur répond aux pings ;
- les erreurs non récupérables (clés invalides, abonnement insuffisant,
  limite de symboles) arrêtent le flux au lieu de boucler.

Chaque événement porte le timestamp de la bourse (`timestamp`) et l'heure de
réception locale (`received_at`) ; la latence est suivie dans `FeedStatus`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, AsyncIterator, Awaitable, Callable, Iterable

from trading_engine.data.events import BarEvent, MarketEvent, QuoteEvent, TradeEvent
from trading_engine.data.market_feed import MarketFeed
from trading_engine.timeutils import parse_rfc3339, utcnow

logger = logging.getLogger(__name__)

STREAM_URL = "wss://stream.data.alpaca.markets/v2/{feed}"
KEY_ENV = "APCA_API_KEY_ID"
SECRET_ENV = "APCA_API_SECRET_KEY"

# Codes d'erreur Alpaca pour lesquels une reconnexion ne changera rien.
FATAL_ERROR_CODES = {
    400,  # invalid syntax : bug de notre côté
    402,  # auth failed
    404,  # auth timeout répété = configuration invalide
    405,  # symbol limit exceeded
    409,  # insufficient subscription
}

ONE_MINUTE = timedelta(minutes=1)


class AlpacaError(Exception):
    def __init__(self, code: int | None, msg: str) -> None:
        super().__init__(f"alpaca error {code}: {msg}")
        self.code = code
        self.msg = msg


class AlpacaFatalError(AlpacaError):
    """Erreur non récupérable : le flux s'arrête."""


class HeartbeatTimeout(Exception):
    pass


@dataclass
class FeedStatus:
    connected: bool = False
    authenticated: bool = False
    connect_count: int = 0
    reconnect_count: int = 0
    messages_received: int = 0
    events_emitted: int = 0
    parse_errors: int = 0
    last_message_at: datetime | None = None
    last_latency: timedelta | None = None
    last_error: str | None = None


def _raise_for_error(message: dict[str, Any]) -> None:
    code = message.get("code")
    text = message.get("msg", "")
    if code in FATAL_ERROR_CODES:
        raise AlpacaFatalError(code, text)
    raise AlpacaError(code, text)


class AlpacaMarketFeed(MarketFeed):
    SOURCE = "alpaca"

    def __init__(
        self,
        symbols: Iterable[str],
        *,
        key_id: str,
        secret_key: str,
        data_feed: str = "iex",
        url: str | None = None,
        trades: bool = True,
        quotes: bool = False,
        bars: bool = True,
        heartbeat_interval: float = 20.0,
        heartbeat_timeout: float = 10.0,
        handshake_timeout: float = 10.0,
        backoff_initial: float = 1.0,
        backoff_max: float = 60.0,
        max_events: int | None = None,
        connect: Callable[..., Any] | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], datetime] = utcnow,
        rng: random.Random | None = None,
    ) -> None:
        self.symbols = sorted(set(symbols))
        if not self.symbols:
            raise ValueError("at least one symbol is required")
        if not key_id or not secret_key:
            raise ValueError("Alpaca API key id and secret are required")
        self._key_id = key_id
        self._secret_key = secret_key
        self.url = url or STREAM_URL.format(feed=data_feed)
        self.channels = {"trades": trades, "quotes": quotes, "bars": bars}
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_timeout = heartbeat_timeout
        self.handshake_timeout = handshake_timeout
        self.backoff_initial = backoff_initial
        self.backoff_max = backoff_max
        self.max_events = max_events
        self._connect = connect or _default_connect
        self._sleep = sleep
        self._clock = clock
        self._rng = rng or random.Random()
        self._stopped = False
        self._ws: Any = None
        self.status = FeedStatus()

    def __repr__(self) -> str:  # ne jamais afficher les clés
        return f"AlpacaMarketFeed(url={self.url!r}, symbols={self.symbols!r})"

    @classmethod
    def from_env(cls, symbols: Iterable[str], **kwargs: Any) -> "AlpacaMarketFeed":
        key_id = os.environ.get(KEY_ENV)
        secret = os.environ.get(SECRET_ENV)
        if not key_id or not secret:
            raise RuntimeError(
                f"Alpaca credentials missing: set {KEY_ENV} and {SECRET_ENV} "
                "in the environment (see .env.example)"
            )
        return cls(symbols, key_id=key_id, secret_key=secret, **kwargs)

    async def stop(self) -> None:
        self._stopped = True
        if self._ws is not None:
            await self._ws.close()

    def backoff_delay(self, attempt: int) -> float:
        base = min(self.backoff_max, self.backoff_initial * (2**attempt))
        return base * (0.5 + 0.5 * self._rng.random())

    # ------------------------------------------------------------------ stream

    async def __aiter__(self) -> AsyncIterator[MarketEvent]:
        attempt = 0
        while not self._stopped:
            try:
                async with self._connect(self.url) as ws:
                    self._ws = ws
                    self.status.connected = True
                    self.status.connect_count += 1
                    if self.status.connect_count > 1:
                        self.status.reconnect_count += 1
                    await self._handshake(ws)
                    attempt = 0
                    logger.info("alpaca stream ready: %s %s", self.url, self.symbols)
                    async for event in self._stream(ws):
                        yield event
                        if self.max_events is not None and self.status.events_emitted >= self.max_events:
                            self._stopped = True
                            return
            except AlpacaFatalError as exc:
                self.status.last_error = str(exc)
                logger.error("alpaca fatal error, stopping feed: %s", exc)
                raise
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # réseau, protocole, heartbeat, erreur Alpaca récupérable
                self.status.last_error = f"{type(exc).__name__}: {exc}"
                logger.warning("alpaca stream interrupted: %s", self.status.last_error)
            finally:
                self._ws = None
                self.status.connected = False
                self.status.authenticated = False

            if self._stopped:
                return
            delay = self.backoff_delay(attempt)
            attempt += 1
            logger.info("reconnecting to alpaca in %.1fs (attempt %d)", delay, attempt)
            await self._sleep(delay)

    async def _recv_messages(self, ws: Any, timeout: float) -> list[dict[str, Any]]:
        raw = await asyncio.wait_for(ws.recv(), timeout)
        self.status.messages_received += 1
        self.status.last_message_at = self._clock()
        data = json.loads(raw)
        return data if isinstance(data, list) else [data]

    async def _expect_success(self, ws: Any, expected: str) -> None:
        for message in await self._recv_messages(ws, self.handshake_timeout):
            kind = message.get("T")
            if kind == "error":
                _raise_for_error(message)
            if kind == "success" and message.get("msg") == expected:
                return
        raise AlpacaError(None, f"unexpected handshake response, wanted {expected!r}")

    async def _handshake(self, ws: Any) -> None:
        await self._expect_success(ws, "connected")
        await ws.send(json.dumps({"action": "auth", "key": self._key_id, "secret": self._secret_key}))
        await self._expect_success(ws, "authenticated")
        self.status.authenticated = True
        subscription = {"action": "subscribe"}
        for channel, enabled in self.channels.items():
            if enabled:
                subscription[channel] = self.symbols
        await ws.send(json.dumps(subscription))

    async def _heartbeat(self, ws: Any) -> None:
        pong = await ws.ping()
        try:
            await asyncio.wait_for(pong, self.heartbeat_timeout)
        except asyncio.TimeoutError:
            raise HeartbeatTimeout(f"no pong within {self.heartbeat_timeout}s") from None

    async def _stream(self, ws: Any) -> AsyncIterator[MarketEvent]:
        while not self._stopped:
            try:
                messages = await self._recv_messages(ws, self.heartbeat_interval)
            except asyncio.TimeoutError:
                await self._heartbeat(ws)
                continue
            for message in messages:
                event = self.parse_message(message)
                if event is not None:
                    self.status.events_emitted += 1
                    self.status.last_latency = event.received_at - event.timestamp
                    yield event

    # ----------------------------------------------------------------- parsing

    def parse_message(self, message: dict[str, Any]) -> MarketEvent | None:
        """Convertit un message Alpaca en événement (None pour les messages de contrôle)."""
        kind = message.get("T")
        if kind == "error":
            _raise_for_error(message)
        if kind in ("success", "subscription"):
            logger.info("alpaca %s: %s", kind, {k: v for k, v in message.items() if k != "T"})
            return None
        try:
            if kind == "t":
                return self._parse_trade(message)
            if kind == "q":
                return self._parse_quote(message)
            if kind in ("b", "u"):
                return self._parse_bar(message, correction=kind == "u")
        except (KeyError, TypeError, ValueError) as exc:
            self.status.parse_errors += 1
            logger.warning("invalid alpaca message %r: %s", message, exc)
            return None
        logger.debug("ignored alpaca message type %r", kind)
        return None

    def _parse_trade(self, m: dict[str, Any]) -> TradeEvent:
        return TradeEvent(
            timestamp=parse_rfc3339(m["t"]),
            received_at=self._clock(),
            symbol=m["S"],
            source=self.SOURCE,
            price=float(m["p"]),
            size=float(m.get("s", 0)),
            payload={
                "trade_id": m.get("i"),
                "exchange": m.get("x"),
                "conditions": tuple(m.get("c") or ()),
                "tape": m.get("z"),
            },
        )

    def _parse_quote(self, m: dict[str, Any]) -> QuoteEvent:
        return QuoteEvent(
            timestamp=parse_rfc3339(m["t"]),
            received_at=self._clock(),
            symbol=m["S"],
            source=self.SOURCE,
            bid=float(m["bp"]),
            ask=float(m["ap"]),
            bid_size=float(m.get("bs", 0)),
            ask_size=float(m.get("as", 0)),
            payload={
                "bid_exchange": m.get("bx"),
                "ask_exchange": m.get("ax"),
                "conditions": tuple(m.get("c") or ()),
                "tape": m.get("z"),
            },
        )

    def _parse_bar(self, m: dict[str, Any], correction: bool) -> BarEvent:
        start = parse_rfc3339(m["t"])
        return BarEvent(
            timestamp=start,
            end=start + ONE_MINUTE,
            received_at=self._clock(),
            symbol=m["S"],
            source=self.SOURCE,
            timeframe="1m",
            open=float(m["o"]),
            high=float(m["h"]),
            low=float(m["l"]),
            close=float(m["c"]),
            volume=float(m.get("v", 0)),
            trade_count=int(m.get("n", 0)),
            payload={"vwap": m.get("vw"), "correction": correction},
        )


def _default_connect(url: str) -> Any:
    import websockets

    # Heartbeat géré par le feed lui-même (ping sur inactivité).
    return websockets.connect(url, ping_interval=None, max_size=2**23)
