"""Tests de perturbation (README §50.6) : rejouer un flux avec de petites
perturbations des données pour vérifier que les décisions n'en dépendent pas
(||ΔS|| << ||δ||). Utilisé dans la suite de tests et pour analyser un journal
enregistré (même moteur, source perturbée)."""

from __future__ import annotations

import dataclasses
import random
from typing import AsyncIterator

from trading_engine.data.events import BarEvent, MarketEvent, QuoteEvent, TradeEvent
from trading_engine.data.market_feed import MarketFeed


class PerturbedFeed(MarketFeed):
    """Enveloppe un flux : prix × (1 + ε·N(0,1)), tailles × (1 + ε_size·N(0,1))."""

    def __init__(self, feed: MarketFeed, *, price_noise: float = 1e-5,
                 size_noise: float = 0.0, seed: int = 0) -> None:
        self.feed = feed
        self.price_noise = price_noise
        self.size_noise = size_noise
        self.rng = random.Random(seed)

    def _p(self, value: float) -> float:
        return value * (1.0 + self.price_noise * self.rng.gauss(0.0, 1.0))

    def _s(self, value: float) -> float:
        return max(0.0, value * (1.0 + self.size_noise * self.rng.gauss(0.0, 1.0)))

    def perturb(self, event: MarketEvent) -> MarketEvent:
        if isinstance(event, TradeEvent):
            return dataclasses.replace(event, price=self._p(event.price), size=self._s(event.size))
        if isinstance(event, QuoteEvent):
            mid_shock = self._p(1.0)
            return dataclasses.replace(event, bid=event.bid * mid_shock, ask=event.ask * mid_shock)
        if isinstance(event, BarEvent):
            k = self._p(1.0)
            return dataclasses.replace(event, open=event.open * k, high=event.high * k,
                                       low=event.low * k, close=event.close * k)
        return event

    async def __aiter__(self) -> AsyncIterator[MarketEvent]:
        async for event in self.feed:
            yield self.perturb(event)
