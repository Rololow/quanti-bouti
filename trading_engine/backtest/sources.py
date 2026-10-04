"""Historique quotidien gratuit (Yahoo Finance, Stooq) pour le backtest long.

Sources non officielles, sans clé, sujettes à limitation : à utiliser pour la
recherche, pas en production. Les données téléchargées ne sont pas
redistribuées (pas commitées) : elles sont écrites en CSV local puis importées
par `import_daily_csv`.

- Yahoo : API « chart » JSON ; `adjclose` = dividendes et splits réinvestis.
- Stooq : CSV `Date,Open,High,Low,Close,Volume`.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable

from trading_engine.data.calendar import NY

logger = logging.getLogger(__name__)

YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
STOOQ_URL = "https://stooq.com/q/d/l/"
HEADERS = {"User-Agent": "Mozilla/5.0 (quanti-bouti backtest)", "Accept": "application/json,text/csv"}
FIELDS = ("Date", "Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividend")


def _get(url: str) -> str:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def yahoo_daily(symbol: str, start: date, end: date, *, http_get: Callable[[str], str] = _get) -> list[dict]:
    """Barres quotidiennes Yahoo (dates New York), y compris `Adj Close`."""
    epoch = lambda d: int(datetime.combine(d, datetime.min.time(), timezone.utc).timestamp())  # noqa: E731
    query = urllib.parse.urlencode({"period1": epoch(start), "period2": epoch(end + timedelta(days=1)),
                                    "interval": "1d", "events": "div,splits", "includeAdjustedClose": "true"})
    data = json.loads(http_get(f"{YAHOO_URL.format(symbol=urllib.parse.quote(symbol))}?{query}"))
    chart = data.get("chart") or {}
    if chart.get("error"):
        raise LookupError(f"Yahoo {symbol}: {chart['error']}")
    result = (chart.get("result") or [None])[0]
    if not result or not result.get("timestamp"):
        raise LookupError(f"Yahoo {symbol}: no data between {start} and {end}")
    quote = result["indicators"]["quote"][0]
    adj = ((result["indicators"].get("adjclose") or [{}])[0]).get("adjclose") or quote["close"]
    to_day = lambda ts: datetime.fromtimestamp(int(ts), timezone.utc).astimezone(NY).date()  # noqa: E731
    dividends: dict[date, float] = {}
    for ev in ((result.get("events") or {}).get("dividends") or {}).values():
        day = to_day(ev["date"])
        dividends[day] = dividends.get(day, 0.0) + float(ev["amount"])
    rows = []
    for i, ts in enumerate(result["timestamp"]):
        values = [quote["open"][i], quote["high"][i], quote["low"][i], quote["close"][i], adj[i]]
        if any(v is None for v in values):
            continue                                    # jour sans cotation (null)
        day = to_day(ts)
        rows.append(dict(zip(FIELDS, [day.isoformat(), *values, quote["volume"][i] or 0, dividends.get(day, 0.0)])))
    return rows


def stooq_daily(symbol: str, start: date, end: date, *, http_get: Callable[[str], str] = _get) -> list[dict]:
    query = urllib.parse.urlencode({"s": f"{symbol.lower()}.us", "i": "d",
                                    "d1": start.strftime("%Y%m%d"), "d2": end.strftime("%Y%m%d")})
    text = http_get(f"{STOOQ_URL}?{query}")
    rows = [r for r in csv.DictReader(io.StringIO(text)) if r.get("Date") and r.get("Close")]
    if not rows:
        raise LookupError(f"Stooq {symbol}: no data ({text[:80]!r})")
    # Pas de dividendes chez Stooq : ni net ni brut, prix seuls (voir README).
    return [{"Date": r["Date"], "Open": r["Open"], "High": r["High"], "Low": r["Low"], "Close": r["Close"],
             "Adj Close": r["Close"], "Volume": r.get("Volume") or 0, "Dividend": ""} for r in rows]


SOURCES = {"yahoo": yahoo_daily, "stooq": stooq_daily}


def yahoo_short_rates(start: date, end: date, *, http_get: Callable[[str], str] = _get) -> dict[date, float]:
    """Taux T-bill 3 mois (^IRX, en %) -> {jour: taux annuel décimal}."""
    rows = yahoo_daily("^IRX", start, end, http_get=http_get)
    return {date.fromisoformat(r["Date"]): float(r["Close"]) / 100.0 for r in rows}


def download_csvs(symbols: Iterable[str], start: date, end: date, directory: str | Path, *,
                  source: str = "yahoo", retries: int = 3, pause: float = 2.0,
                  http_get: Callable[[str], str] = _get, sleep: Callable[[float], None] = time.sleep) -> dict[str, Path]:
    """Un CSV par symbole dans `directory` ; retourne {symbole: chemin}."""
    fetch = SOURCES[source]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    out = {}
    for sym in symbols:
        for attempt in range(1, retries + 1):
            try:
                rows = fetch(sym, start, end, http_get=http_get)
                break
            except Exception as exc:
                if attempt == retries:
                    raise
                logger.warning("%s %s attempt %d failed: %s", source, sym, attempt, exc)
                sleep(pause * attempt)
        path = directory / f"{sym}.csv"
        with open(path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        out[sym] = path
        logger.info("%s %s: %d rows", source, sym, len(rows))
        sleep(pause)                                     # politesse envers la source
    return out
