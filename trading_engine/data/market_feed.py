"""Flux de marché.

`MarketFeed` est l'interface commune ; `SimulatedMarketFeed` génère un flux
déterministe (marche aléatoire log-normale) pour développer et tester le
moteur sans clé API. Le flux Alpaca WebSocket arrivera en Phase 2 derrière la
même interface.
"""

from __future__ import annotations

import abc
import asyncio
import math
import random
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator, Mapping

from trading_engine.data.events import MarketEvent, QuoteEvent, TradeEvent


class MarketFeed(abc.ABC):
    @abc.abstractmethod
    def __aiter__(self) -> AsyncIterator[MarketEvent]: ...


class SimulatedMarketFeed(MarketFeed):
    SOURCE = "simulated"

    def __init__(
        self,
        initial_prices: Mapping[str, float],
        *,
        seed: int | None = None,
        start: datetime | None = None,
        tick_seconds: float = 1.0,
        annual_vol: float = 0.20,
        max_events: int | None = None,
        quote_probability: float = 0.3,
        realtime: bool = False,
    ) -> None:
        if not initial_prices:
            raise ValueError("at least one symbol is required")
        self.prices = dict(initial_prices)
        self.rng = random.Random(seed)
        self.clock = start or datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc)
        self.step = timedelta(seconds=tick_seconds)
        self.max_events = max_events
        self.quote_probability = quote_probability
        self.realtime = realtime
        # Volatilité par tick (temps continu 24/7 simplifié).
        seconds_per_year = 365 * 24 * 3600
        self.tick_vol = annual_vol * math.sqrt(tick_seconds / seconds_per_year)

    async def __aiter__(self) -> AsyncIterator[MarketEvent]:
        symbols = sorted(self.prices)
        n = 0
        while self.max_events is None or n < self.max_events:
            self.clock += self.step
            symbol = self.rng.choice(symbols)
            shock = self.rng.gauss(-0.5 * self.tick_vol**2, self.tick_vol)
            price = round(self.prices[symbol] * math.exp(shock), 4)
            self.prices[symbol] = price

            if self.rng.random() < self.quote_probability:
                half_spread = max(0.01, price * 0.0001)
                event: MarketEvent = QuoteEvent(
                    timestamp=self.clock,
                    received_at=self.clock,
                    symbol=symbol,
                    source=self.SOURCE,
                    bid=round(price - half_spread, 4),
                    ask=round(price + half_spread, 4),
                    bid_size=self.rng.randint(1, 50) * 100,
                    ask_size=self.rng.randint(1, 50) * 100,
                )
            else:
                event = TradeEvent(
                    timestamp=self.clock,
                    received_at=self.clock,
                    symbol=symbol,
                    source=self.SOURCE,
                    price=price,
                    size=self.rng.randint(1, 20) * 10,
                )
            yield event
            n += 1
            if self.realtime:
                await asyncio.sleep(self.step.total_seconds())
            else:
                await asyncio.sleep(0)
