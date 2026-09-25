"""État de marché courant par symbole (README §35 — MarketState)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime

from trading_engine.data.events import BarEvent, Event, QuoteEvent, TradeEvent


@dataclass(frozen=True)
class MarketState:
    symbol: str
    price: float
    timestamp: datetime
    bid: float | None = None
    ask: float | None = None

    @property
    def spread(self) -> float | None:
        if self.bid is None or self.ask is None:
            return None
        return self.ask - self.bid


class MarketStateStore:
    """Conserve le dernier `MarketState` (immuable) de chaque symbole.

    Les événements plus anciens que l'état courant sont ignorés : un message
    arrivé en retard ne doit pas écraser une information plus récente.
    """

    def __init__(self) -> None:
        self._states: dict[str, MarketState] = {}

    def get(self, symbol: str) -> MarketState | None:
        return self._states.get(symbol)

    def prices(self) -> dict[str, float]:
        return {sym: st.price for sym, st in self._states.items()}

    def __contains__(self, symbol: str) -> bool:
        return symbol in self._states

    def update(self, event: Event) -> MarketState | None:
        symbol = event.symbol
        if symbol is None:
            return None
        current = self._states.get(symbol)
        if current is not None and event.timestamp < current.timestamp:
            return current

        if isinstance(event, TradeEvent):
            new = (
                MarketState(symbol, event.price, event.timestamp)
                if current is None
                else replace(current, price=event.price, timestamp=event.timestamp)
            )
        elif isinstance(event, QuoteEvent):
            if current is None:
                new = MarketState(symbol, event.mid, event.timestamp, event.bid, event.ask)
            else:
                new = replace(current, bid=event.bid, ask=event.ask, timestamp=event.timestamp)
        elif isinstance(event, BarEvent):
            if current is not None and event.end <= current.timestamp:
                return current
            new = (
                MarketState(symbol, event.close, event.end)
                if current is None
                else replace(current, price=event.close, timestamp=event.end)
            )
        else:
            return current

        self._states[symbol] = new
        return new
