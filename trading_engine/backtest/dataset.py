"""Données historiques pour le backtest.

- `download_bars` : barres Alpaca (REST) sur une longue période, limitées aux
  séances régulières (calendrier NYSE), écrites en JSONL (`BarEvent`), plus
  les taux BCE EUR/USD de la période (`<fichier>.fx.json`) ;
- `BarDatasetFeed` : rejoue ces barres dans le moteur **inchangé**, sous forme
  de trades synthétiques (convention OHLC : ouverture, puis le plus bas et le
  plus haut dans l'ordre le plus probable, puis la clôture). Barres, risque,
  décisions, ordres et fills passent par exactement le même code qu'en live.

Limites (assumées) : le chemin intra-barre est une approximation ; le volume
d'une barre est réparti sur ses trades synthétiques ; pas de cotations (le
spread est le spread par défaut du modèle de coûts).
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import AsyncIterator, Callable, Iterable, Iterator

from trading_engine.data.alpaca_history import AlpacaHistoricalClient
from trading_engine.data.calendar import NY, MarketCalendar
from trading_engine.data.events import BarEvent, MarketEvent, TradeEvent
from trading_engine.data.market_feed import MarketFeed
from trading_engine.storage.event_log import EventLogWriter, read_events
from trading_engine.tax.fx import fetch_ecb_rates

logger = logging.getLogger(__name__)

# Position des trades synthétiques dans la barre (fraction de sa durée).
TRADE_OFFSETS = (0.0, 0.25, 0.5, 0.9)


def fx_path(dataset: str | Path) -> Path:
    dataset = Path(dataset)
    return dataset.with_name(dataset.name + ".fx.json")


def in_session(bar: BarEvent, calendar: MarketCalendar) -> bool:
    session = calendar.session(bar.timestamp.astimezone(NY).date())
    return session is not None and session.open <= bar.timestamp and bar.end <= session.close


def _midnight(day: date) -> datetime:
    return datetime.combine(day, time(0), NY).astimezone(timezone.utc)


def download_bars(
    client: AlpacaHistoricalClient,
    symbols: Iterable[str],
    start: date,
    end: date,
    path: str | Path,
    *,
    timeframe: str = "30m",
    chunk_days: int = 90,
    with_fx: bool = True,
    fx_get: Callable[..., object] | None = None,
) -> dict:
    """Télécharge et écrit le dataset. Retourne un résumé (barres par symbole…)."""
    symbols = sorted(set(symbols))
    calendar = MarketCalendar.from_rules(start, end)
    bars: list[BarEvent] = []
    dropped = 0
    cursor = start
    while cursor <= end:
        stop = min(end, cursor + timedelta(days=chunk_days - 1))
        chunk = client.bars(symbols, timeframe, _midnight(cursor), _midnight(stop + timedelta(days=1)))
        for bar in chunk:
            if in_session(bar, calendar):
                # Données historiques : pas un warm-up (le moteur trade dessus).
                bars.append(BarEvent(**{**_fields(bar), "source": "alpaca_dataset",
                                        "received_at": bar.end, "payload": {"vwap": bar.payload.get("vwap")}}))
            else:
                dropped += 1
        logger.info("dataset %s..%s: %d bars", cursor, stop, len(chunk))
        cursor = stop + timedelta(days=1)
    bars.sort(key=lambda b: (b.timestamp, b.symbol))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with EventLogWriter(path) as writer:
        for bar in bars:
            writer.write(bar)
    summary = {
        "path": str(path), "timeframe": timeframe, "start": start.isoformat(), "end": end.isoformat(),
        "bars": len(bars), "dropped_outside_session": dropped,
        "per_symbol": {s: sum(1 for b in bars if b.symbol == s) for s in symbols},
        "fx": None,
    }
    if with_fx:
        try:
            kwargs = {} if fx_get is None else {"http_get": fx_get}
            rates = fetch_ecb_rates("USD", start - timedelta(days=10), end, **kwargs)
            with open(fx_path(path), "w", encoding="utf-8") as fh:
                json.dump(rates.to_payload(), fh)
            summary["fx"] = {"path": str(fx_path(path)), "rates": len(rates)}
        except Exception as exc:          # le backtest utilisera alors le taux fixe
            logger.warning("ECB rates not downloaded: %s", exc)
    return summary


def _fields(bar: BarEvent) -> dict:
    return {f: getattr(bar, f) for f in ("timestamp", "end", "received_at", "symbol", "source", "timeframe",
                                         "open", "high", "low", "close", "volume", "trade_count", "payload")}


def synthetic_trades(bar: BarEvent) -> list[TradeEvent]:
    """Trades d'une barre : O, puis L/H (ou H/L si la barre baisse), puis C."""
    duration = bar.end - bar.timestamp
    middle = (bar.low, bar.high) if bar.close >= bar.open else (bar.high, bar.low)
    prices = (bar.open, *middle, bar.close)
    size = max(bar.volume, 0.0) / len(prices)
    return [
        TradeEvent(timestamp=bar.timestamp + duration * f, received_at=bar.timestamp + duration * f,
                   symbol=bar.symbol, source="dataset", price=p, size=size)
        for f, p in zip(TRADE_OFFSETS, prices)
    ]


class BarDatasetFeed(MarketFeed):
    """Dataset de barres -> trades synthétiques, dans l'ordre chronologique global."""

    def __init__(self, path: str | Path, *, start: datetime | None = None, end: datetime | None = None,
                 max_events: int | None = None, yield_every: int = 2000) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"dataset not found: {self.path}")
        self.start, self.end = start, end
        self.max_events = max_events
        self.yield_every = max(1, yield_every)
        self.bars = 0

    def _groups(self) -> Iterator[list[BarEvent]]:
        group: list[BarEvent] = []
        for ev in read_events(self.path):
            if not isinstance(ev, BarEvent):
                continue
            if (self.start and ev.timestamp < self.start) or (self.end and ev.timestamp >= self.end):
                continue
            if group and ev.timestamp != group[0].timestamp:
                yield group
                group = []
            group.append(ev)
        if group:
            yield group

    async def __aiter__(self) -> AsyncIterator[MarketEvent]:
        n = 0
        for group in self._groups():
            self.bars += len(group)
            trades = sorted((t for bar in group for t in synthetic_trades(bar)),
                            key=lambda t: (t.timestamp, t.symbol))
            for trade in trades:
                if self.max_events is not None and n >= self.max_events:
                    return
                n += 1
                yield trade
                if n % self.yield_every == 0:
                    await asyncio.sleep(0)


def read_closes(path: str | Path) -> dict[str, list[tuple[datetime, float]]]:
    """Clôtures par symbole (pour les benchmarks)."""
    out: dict[str, list[tuple[datetime, float]]] = {}
    for ev in read_events(path):
        if isinstance(ev, BarEvent):
            out.setdefault(ev.symbol, []).append((ev.end, ev.close))
    return out


def write_synthetic_dataset(
    path: str | Path,
    symbols: Iterable[str],
    start: date,
    end: date,
    *,
    timeframe_minutes: int = 30,
    annual_vol: float = 0.18,
    drift: float = 0.05,
    seed: int = 0,
) -> int:
    """Dataset synthétique (séances NYSE réelles, prix log-normaux) pour les
    tests et pour vérifier la chaîne sans clé API. Aucun pouvoir prédictif."""
    import math
    import random

    rng = random.Random(seed)
    calendar = MarketCalendar.from_rules(start, end)
    step = timedelta(minutes=timeframe_minutes)
    per_year = 252 * 6.5 * 60 / timeframe_minutes
    sigma = annual_vol / math.sqrt(per_year)
    mu = drift / per_year - sigma * sigma / 2
    prices = {s: 100.0 * (1 + i) for i, s in enumerate(sorted(set(symbols)))}
    n = 0
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with EventLogWriter(path) as writer:
        for session in calendar.sessions:
            t = session.open
            while t + step <= session.close:
                for sym in sorted(prices):
                    o = prices[sym]
                    c = o * math.exp(mu + sigma * rng.gauss(0, 1))
                    h = max(o, c) * math.exp(abs(rng.gauss(0, sigma / 2)))
                    low = min(o, c) * math.exp(-abs(rng.gauss(0, sigma / 2)))
                    writer.write(BarEvent(timestamp=t, end=t + step, received_at=t + step, symbol=sym,
                                          source="synthetic", timeframe=f"{timeframe_minutes}m", open=o,
                                          high=h, low=low, close=c, volume=float(int(rng.uniform(5e4, 2e5))),
                                          trade_count=100))
                    prices[sym] = c
                    n += 1
                t += step
    return n


def import_daily_csv(files: dict[str, str | Path], path: str | Path, *, start: date | None = None,
                     end: date | None = None) -> dict:
    """CSV quotidiens (un par symbole) -> dataset de barres 1d sur les séances NYSE.

    Colonnes reconnues (casse ignorée) : Date, Open, High, Low, Close, Volume
    et, si présente, « Adj Close » : OHLC sont alors multipliés par
    Adj Close / Close (dividendes réinvestis, comme `adjustment=all`).
    Format Stooq (`Date,Open,High,Low,Close,Volume`) et Yahoo acceptés.
    Une barre couvre la séance (ouverture -> clôture, 13h00 les veilles de fête).
    """
    import csv

    bars: list[BarEvent] = []
    skipped: dict[str, int] = {}
    for sym, file in sorted(files.items()):
        with open(file, encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            cols = {c.strip().lower(): c for c in (reader.fieldnames or [])}
            missing = [c for c in ("date", "open", "high", "low", "close") if c not in cols]
            if missing:
                raise ValueError(f"{file}: missing columns {missing}")
            rows = list(reader)
        days = [date.fromisoformat(r[cols["date"]].strip()[:10]) for r in rows]
        if not days:
            continue
        calendar = MarketCalendar.from_rules(min(days), max(days))
        for r, day in zip(rows, days):
            if (start and day < start) or (end and day > end):
                continue
            try:
                o, h, low, c = (float(r[cols[k]]) for k in ("open", "high", "low", "close"))
                v = float(r[cols["volume"]]) if "volume" in cols and r[cols["volume"]] not in ("", None) else 0.0
                adj = float(r[cols["adj close"]]) if "adj close" in cols and r[cols["adj close"]] else c
            except ValueError:
                skipped[sym] = skipped.get(sym, 0) + 1         # « null », ligne incomplète
                continue
            session = calendar.session(day)
            if session is None or min(o, h, low, c) <= 0 or c <= 0:
                skipped[sym] = skipped.get(sym, 0) + 1
                continue
            k = adj / c
            o, h, low, c = o * k, max(o, h, low, c) * k, min(o, h, low, c) * k, adj
            bars.append(BarEvent(timestamp=session.open, end=session.close, received_at=session.close, symbol=sym,
                                 source="csv_dataset", timeframe="1d", open=o, high=h, low=low, close=c,
                                 volume=max(v, 0.0)))
    bars.sort(key=lambda b: (b.timestamp, b.symbol))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with EventLogWriter(path) as writer:
        for bar in bars:
            writer.write(bar)
    return {"path": str(path), "bars": len(bars), "skipped": skipped,
            "per_symbol": {s: sum(1 for b in bars if b.symbol == s) for s in files},
            "start": bars[0].timestamp.date().isoformat() if bars else None,
            "end": bars[-1].timestamp.date().isoformat() if bars else None}
