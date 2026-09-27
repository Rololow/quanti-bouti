"""Événements génériques du moteur (README §4.2).

Chaque événement porte au minimum `timestamp`, `symbol`, `event_type`, `source`
et `payload`. On distingue aussi `received_at` (quand notre système l'a reçu)
du `timestamp` (quand l'événement a eu lieu) pour respecter la contrainte
d'information timing (README §21).

Les événements sont immuables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from trading_engine.timeutils import ensure_utc, utcnow


class EventType(str, Enum):
    MARKET = "market"
    TRADE = "trade"
    QUOTE = "quote"
    BAR = "bar"
    NEWS = "news"
    FUNDAMENTAL = "fundamental"
    PORTFOLIO = "portfolio"
    RISK = "risk"
    DECISION = "decision"
    ALERT = "alert"


@dataclass(frozen=True, kw_only=True)
class Event:
    timestamp: datetime
    symbol: str | None
    source: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    received_at: datetime = field(default_factory=utcnow)

    event_type = EventType.MARKET

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", ensure_utc(self.timestamp))
        object.__setattr__(self, "received_at", ensure_utc(self.received_at))
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))


@dataclass(frozen=True, kw_only=True)
class MarketEvent(Event):
    event_type = EventType.MARKET


@dataclass(frozen=True, kw_only=True)
class TradeEvent(MarketEvent):
    price: float
    size: float = 0.0

    event_type = EventType.TRADE

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.price <= 0:
            raise ValueError(f"trade price must be positive, got {self.price}")


@dataclass(frozen=True, kw_only=True)
class QuoteEvent(MarketEvent):
    bid: float
    ask: float
    bid_size: float = 0.0
    ask_size: float = 0.0

    event_type = EventType.QUOTE

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass(frozen=True, kw_only=True)
class BarEvent(MarketEvent):
    """Barre OHLCV. `timestamp` = début de la barre, `end` = fin (exclue)."""

    timeframe: str
    end: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    trade_count: int = 0

    event_type = EventType.BAR


@dataclass(frozen=True, kw_only=True)
class NewsEvent(Event):
    headline: str = ""

    event_type = EventType.NEWS


@dataclass(frozen=True, kw_only=True)
class FundamentalEvent(Event):
    """Donnée fondamentale datée (README §20) : jamais utilisable avant `available_at`."""

    name: str
    value: float
    period: str
    available_at: datetime

    event_type = EventType.FUNDAMENTAL

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "available_at", ensure_utc(self.available_at))

    def is_available(self, at: datetime) -> bool:
        return self.available_at <= ensure_utc(at)


@dataclass(frozen=True, kw_only=True)
class PortfolioEvent(Event):
    event_type = EventType.PORTFOLIO


@dataclass(frozen=True, kw_only=True)
class RiskEvent(Event):
    event_type = EventType.RISK


@dataclass(frozen=True, kw_only=True)
class DecisionEvent(Event):
    event_type = EventType.DECISION


@dataclass(frozen=True, kw_only=True)
class AlertEvent(Event):
    kind: str
    severity: str = "info"            # info | warning | critical
    message: str = ""

    event_type = EventType.ALERT
