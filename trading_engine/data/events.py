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
    NEWS_ANALYSIS = "news_analysis"
    ORDER_UPDATE = "order_update"
    CALENDAR = "calendar"
    FX = "fx"
    TAX_LEDGER = "tax_ledger"
    RATES = "rates"
    INTEREST_INDEX = "interest_index"


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
    """News brute. `timestamp` = publication, `received_at` = réception.
    Une news qui cite plusieurs symboles donne un événement par symbole,
    reliés par le même `news_id`."""

    headline: str = ""
    summary: str = ""
    url: str = ""
    news_id: str = ""
    provider: str = ""            # agence / éditeur (Benzinga, Reuters…)

    event_type = EventType.NEWS


@dataclass(frozen=True, kw_only=True)
class FundamentalEvent(Event):
    """Donnée fondamentale datée (README §20) : jamais utilisable avant `available_at`."""

    name: str                     # ex. revenue, eps, guidance_eps
    value: float
    period: str                   # période concernée, ex. 2026Q1
    available_at: datetime        # publication : jamais utilisable avant
    estimate: float | None = None # consensus attendu (earnings surprise)
    unit: str = ""

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
class CalendarEvent(Event):
    """Séances de marché (payload : `MarketCalendar.to_payload()`)."""

    event_type = EventType.CALENDAR


@dataclass(frozen=True, kw_only=True)
class FxEvent(Event):
    """Taux de change (payload : `FxRates.to_payload()`)."""

    event_type = EventType.FX


@dataclass(frozen=True, kw_only=True)
class RatesEvent(Event):
    """Taux courts pour le financement (payload : `RateSeries.to_payload()`)."""

    event_type = EventType.RATES


@dataclass(frozen=True, kw_only=True)
class InterestIndexEvent(Event):
    """Intérêts accumulés par part des fonds obligataires (base Reynders) :
    payload `{symbole: [[jour, cumul], ...]}`."""

    event_type = EventType.INTEREST_INDEX


@dataclass(frozen=True, kw_only=True)
class TaxLedgerEvent(Event):
    """État du registre fiscal au démarrage (payload : `source`, `ledger`)."""

    event_type = EventType.TAX_LEDGER


@dataclass(frozen=True, kw_only=True)
class RiskEvent(Event):
    event_type = EventType.RISK


@dataclass(frozen=True, kw_only=True)
class DecisionEvent(Event):
    event_type = EventType.DECISION


@dataclass(frozen=True, kw_only=True)
class NewsAnalysisEvent(Event):
    """Sortie structurée (validée) de l'IA pour une news (README §23-26).

    `timestamp` = publication de la news ; `received_at` = moment où l'analyse
    est disponible (après la latence du modèle) : elle n'est jamais utilisée
    avant. L'analyse elle-même est dans `payload`. Enregistrée dans le
    journal : le replay la relit sans rappeler le modèle.
    """

    news_id: str
    headline: str = ""
    model: str = ""

    event_type = EventType.NEWS_ANALYSIS


@dataclass(frozen=True, kw_only=True)
class OrderUpdateEvent(Event):
    """Mise à jour d'un ordre chez le broker (Alpaca `trade_updates`).

    `update` : new | fill | partial_fill | canceled | expired | rejected | ...
    `fill_qty` / `fill_price` : exécution de cet événement (fill, partial_fill) ;
    `filled_qty` / `filled_avg_price` : cumul de l'ordre. Enregistrée dans le
    journal : le replay rejoue les fills sans broker.
    """

    client_order_id: str
    broker_order_id: str = ""
    update: str = ""
    side: str = ""
    fill_qty: float = 0.0
    fill_price: float | None = None
    filled_qty: float = 0.0
    filled_avg_price: float | None = None

    event_type = EventType.ORDER_UPDATE

    @property
    def terminal(self) -> bool:
        return self.update in ("fill", "canceled", "expired", "rejected", "done_for_day")


@dataclass(frozen=True, kw_only=True)
class AlertEvent(Event):
    kind: str
    severity: str = "info"            # info | warning | critical
    message: str = ""

    event_type = EventType.ALERT
