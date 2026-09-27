"""Journal des événements bruts (README §41-43).

Format JSONL : un événement par ligne, dans l'ordre de réception.

    {"type": "TradeEvent", "timestamp": "...", "received_at": "...", "symbol": "SPY", ...}

Le journal est la source du replay : relire le fichier redonne exactement la
même séquence d'événements, donc le même état du moteur.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime
from pathlib import Path
from typing import IO, Any, Iterator

from trading_engine.data.events import (
    AlertEvent,
    BarEvent,
    DecisionEvent,
    Event,
    FundamentalEvent,
    MarketEvent,
    NewsAnalysisEvent,
    NewsEvent,
    PortfolioEvent,
    QuoteEvent,
    RiskEvent,
    TradeEvent,
)

EVENT_CLASSES: dict[str, type[Event]] = {
    cls.__name__: cls
    for cls in (
        MarketEvent, TradeEvent, QuoteEvent, BarEvent, NewsEvent, FundamentalEvent,
        PortfolioEvent, RiskEvent, DecisionEvent, AlertEvent, NewsAnalysisEvent,
    )
}


def _datetime_fields(cls: type[Event]) -> frozenset[str]:
    return frozenset(f.name for f in dataclasses.fields(cls) if "datetime" in str(f.type))


_DATETIME_FIELDS = {name: _datetime_fields(cls) for name, cls in EVENT_CLASSES.items()}


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (tuple, set, frozenset)):
        return list(value)
    if hasattr(value, "items"):  # MappingProxyType
        return dict(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def to_record(event: Event) -> dict[str, Any]:
    name = type(event).__name__
    if name not in EVENT_CLASSES:
        raise TypeError(f"unsupported event type {name}")
    record: dict[str, Any] = {"type": name}
    for f in dataclasses.fields(event):
        value = getattr(event, f.name)
        record[f.name] = dict(value) if f.name == "payload" else value
    return record


def from_record(record: dict[str, Any]) -> Event:
    record = dict(record)
    name = record.pop("type")
    cls = EVENT_CLASSES[name]
    for field_name in _DATETIME_FIELDS[name]:
        if record.get(field_name) is not None:
            record[field_name] = datetime.fromisoformat(record[field_name])
    return cls(**record)


def dumps(event: Event) -> str:
    return json.dumps(to_record(event), default=_json_default, separators=(",", ":"))


def loads(line: str) -> Event:
    return from_record(json.loads(line))


class EventLogWriter:
    """Écrit les événements en append ; `flush_every` borne la perte en cas de crash."""

    def __init__(self, path: str | Path, flush_every: int = 100) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: IO[str] | None = open(self.path, "a", encoding="utf-8")
        self.flush_every = max(1, flush_every)
        self.count = 0

    def write(self, event: Event) -> None:
        if self._fh is None:
            raise ValueError("event log is closed")
        self._fh.write(dumps(event) + "\n")
        self.count += 1
        if self.count % self.flush_every == 0:
            self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "EventLogWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_events(path: str | Path) -> Iterator[Event]:
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield loads(line)
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{lineno}: invalid event record: {exc}") from exc
