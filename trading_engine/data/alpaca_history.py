"""Historique Alpaca pour initialiser le moteur (README §45 : historical seed).

Sans historique, il faut plusieurs séances avant la première covariance, et
des semaines avant les régimes MT/LT ou le momentum 20-252 jours. Au
démarrage, les barres passées sont chargées par l'API REST de données :

    GET https://data.alpaca.markets/v2/stocks/bars
        ?symbols=SPY,QQQ&timeframe=1Hour&start=...&end=...&feed=iex
        &adjustment=all&limit=10000&page_token=...

Chaque barre devient un `BarEvent` marqué `payload.warmup = True` : elle
initialise features, modèles et volumes, mais ne déclenche aucune décision.
Elles sont enregistrées dans le journal, donc le replay repart du même état.
Seules les barres **complètes** (fin <= maintenant) sont chargées.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterable, Mapping

from trading_engine.data.alpaca_feed import KEY_ENV, SECRET_ENV
from trading_engine.data.events import BarEvent
from trading_engine.timeutils import floor_time, parse_rfc3339, parse_timeframe

logger = logging.getLogger(__name__)

DATA_URL = "https://data.alpaca.markets/v2/stocks/bars"
ALPACA_TIMEFRAMES = {"1m": "1Min", "5m": "5Min", "15m": "15Min", "30m": "30Min", "1h": "1Hour", "1d": "1Day"}


def _http_get(url: str, headers: Mapping[str, str]) -> Mapping[str, Any]:
    req = urllib.request.Request(url, headers=dict(headers))
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


class AlpacaHistoricalClient:
    def __init__(
        self,
        *,
        key_id: str,
        secret_key: str,
        data_feed: str = "iex",
        http_get: Callable[[str, Mapping[str, str]], Mapping[str, Any]] = _http_get,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        if not key_id or not secret_key:
            raise ValueError("Alpaca API key id and secret are required")
        self._headers = {"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret_key,
                         "Accept": "application/json"}
        self.data_feed = data_feed
        self._get = http_get
        self._clock = clock

    def __repr__(self) -> str:  # ne jamais afficher les clés
        return f"AlpacaHistoricalClient(feed={self.data_feed!r})"

    @classmethod
    def from_env(cls, **kwargs: Any) -> "AlpacaHistoricalClient":
        key, secret = os.environ.get(KEY_ENV), os.environ.get(SECRET_ENV)
        if not key or not secret:
            raise RuntimeError(f"Alpaca credentials missing: set {KEY_ENV} and {SECRET_ENV}")
        return cls(key_id=key, secret_key=secret, **kwargs)

    def bars(self, symbols: Iterable[str], timeframe: str, start: datetime, end: datetime) -> list[BarEvent]:
        if timeframe not in ALPACA_TIMEFRAMES:
            raise ValueError(f"unsupported timeframe for Alpaca history: {timeframe!r}")
        step = parse_timeframe(timeframe)
        symbols = sorted(set(symbols))
        received = self._clock()
        params = {
            "symbols": ",".join(symbols), "timeframe": ALPACA_TIMEFRAMES[timeframe],
            "start": start.isoformat().replace("+00:00", "Z"), "end": end.isoformat().replace("+00:00", "Z"),
            "feed": self.data_feed, "adjustment": "all", "limit": "10000",
        }
        out: list[BarEvent] = []
        token = None
        while True:
            query = dict(params, **({"page_token": token} if token else {}))
            data = self._get(f"{DATA_URL}?{urllib.parse.urlencode(query)}", self._headers)
            for sym, rows in (data.get("bars") or {}).items():
                for row in rows or []:
                    begin = parse_rfc3339(row["t"])
                    if begin + step > received:
                        continue                      # barre encore ouverte
                    out.append(BarEvent(
                        timestamp=begin, end=begin + step, received_at=received, symbol=sym,
                        source="alpaca_history", timeframe=timeframe, open=float(row["o"]),
                        high=float(row["h"]), low=float(row["l"]), close=float(row["c"]),
                        volume=float(row.get("v", 0)), trade_count=int(row.get("n", 0)),
                        payload={"warmup": True, "vwap": row.get("vw")},
                    ))
            token = data.get("next_page_token")
            if not token:
                break
        return out

    async def warmup_bars(self, symbols: Iterable[str], lookbacks: Mapping[str, float]) -> list[BarEvent]:
        """Barres de démarrage pour chaque timeframe (`lookbacks` : jours calendaires),
        triées par fin de barre puis timeframe puis symbole."""
        now = self._clock()
        symbols = list(symbols)
        bars: list[BarEvent] = []
        for timeframe, days in sorted(lookbacks.items()):
            end = floor_time(now, parse_timeframe(timeframe))
            start = end - timedelta(days=float(days))
            try:
                chunk = await asyncio.to_thread(self.bars, symbols, timeframe, start, end)
            except Exception as exc:          # sans historique, le moteur démarre à froid
                logger.warning("Alpaca history %s failed: %s", timeframe, exc)
                continue
            logger.info("warm-up: %d bars %s", len(chunk), timeframe)
            bars.extend(chunk)
        bars.sort(key=lambda b: (b.end, parse_timeframe(b.timeframe), b.symbol))
        return bars
