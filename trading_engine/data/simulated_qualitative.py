"""Flux simulé de résultats trimestriels, guidance et news (démo et tests).

- earnings : EPS réalisé vs consensus (surprise aléatoire), chiffre d'affaires ;
  parfois une nouvelle guidance ;
- news : titres générés, avec des **reprises quasi identiques** par d'autres
  éditeurs (pour exercer la déduplication : un événement ≠ trois articles).

Déterministe pour une graine donnée.
"""

from __future__ import annotations

import asyncio
import math
import random
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator, Sequence

from trading_engine.data.events import Event, FundamentalEvent, NewsEvent
from trading_engine.data.market_feed import MarketFeed

TEMPLATES = {
    1: ["{s} beats expectations and raises outlook", "{s} reports record revenue",
        "{s} announces major new contract"],
    -1: ["{s} misses estimates, cuts guidance", "{s} faces regulatory probe",
         "{s} warns of weaker demand"],
}
REPHRASE = [("raises outlook", "lifts forecast"), ("reports record revenue", "posts record sales"),
            ("cuts guidance", "lowers outlook"), ("faces regulatory probe", "under regulatory investigation"),
            ("beats expectations", "tops estimates"), ("warns of weaker demand", "sees softer demand")]
PROVIDERS = ["Reuters", "Bloomberg", "Benzinga", "MarketWatch"]


class SimulatedQualitativeFeed(MarketFeed):
    def __init__(
        self,
        symbols: Sequence[str],
        *,
        start: datetime | None = None,
        until: datetime | None = None,
        seed: int | None = None,
        earnings_every: timedelta = timedelta(hours=24),
        news_every: timedelta = timedelta(hours=3),
        duplicate_probability: float = 0.5,
    ) -> None:
        self.symbols = sorted(symbols)
        self.start = start or datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc)
        self.until = until
        self.rng = random.Random(seed)
        self.earnings_every = earnings_every
        self.news_every = news_every
        self.duplicate_probability = duplicate_probability
        self._base: dict[str, tuple[float, float]] = {}

    def _earnings(self, sym: str, when: datetime, quarter: int) -> list[Event]:
        # Fondamentaux persistants par symbole : EPS et chiffre d'affaires
        # suivent une marche aléatoire (croissance ~2 %/trimestre), le consensus
        # est proche de la tendance et le réalisé s'en écarte un peu.
        base_eps, base_rev = self._base.setdefault(
            sym, (self.rng.uniform(1.0, 3.0), self.rng.uniform(1e9, 5e9)))
        growth = math.exp(self.rng.gauss(0.02, 0.04))
        base_eps, base_rev = base_eps * growth, base_rev * growth
        self._base[sym] = (base_eps, base_rev)
        estimate = round(base_eps * (1 + self.rng.gauss(0.0, 0.02)), 2)
        actual = round(estimate * (1 + self.rng.gauss(0.0, 0.06)), 2)
        period = f"2026Q{quarter}"
        events: list[Event] = [
            FundamentalEvent(timestamp=when, received_at=when, symbol=sym, source="simulated",
                             name="eps", value=actual, estimate=estimate, period=period,
                             available_at=when, unit="USD/share"),
            FundamentalEvent(timestamp=when, received_at=when, symbol=sym, source="simulated",
                             name="revenue", value=round(base_rev * (1 + self.rng.gauss(0.0, 0.02))),
                             period=period, available_at=when, unit="USD"),
        ]
        if self.rng.random() < 0.5:
            events.append(FundamentalEvent(
                timestamp=when, received_at=when, symbol=sym, source="simulated", name="guidance_eps",
                value=round(base_eps * 4 * (1 + self.rng.gauss(0.02, 0.03)), 2),
                period="2026FY", available_at=when, unit="USD/share"))
        return events

    def _news(self, sym: str, when: datetime, n: int) -> list[Event]:
        tone = self.rng.choice([1, -1])
        headline = self.rng.choice(TEMPLATES[tone]).format(s=sym)
        provider = self.rng.choice(PROVIDERS)
        out: list[Event] = [NewsEvent(timestamp=when, received_at=when, symbol=sym, source="simulated",
                                      headline=headline, news_id=f"n{n}", provider=provider,
                                      payload={"tone": tone})]
        k = 1
        while self.rng.random() < self.duplicate_probability and k < 4:
            text = headline
            for a, b in REPHRASE:
                if a in text and self.rng.random() < 0.7:
                    text = text.replace(a, b)
            later = when + timedelta(minutes=self.rng.randint(1, 30))
            out.append(NewsEvent(timestamp=later, received_at=later, symbol=sym, source="simulated",
                                 headline=text, news_id=f"n{n}-{k}", provider=PROVIDERS[k % len(PROVIDERS)],
                                 payload={"tone": tone, "duplicate_of": f"n{n}"}))
            k += 1
        return out

    async def __aiter__(self) -> AsyncIterator[Event]:
        pending: list[Event] = []
        next_earnings = self.start + self.earnings_every / 2
        next_news = self.start + self.news_every / 2
        quarter, n = 1, 0
        while True:
            if next_earnings <= next_news:
                when = next_earnings
                for sym in self.symbols:
                    pending.extend(self._earnings(sym, when, quarter))
                quarter = quarter % 4 + 1
                next_earnings += self.earnings_every
            else:
                when = next_news
                pending.extend(self._news(self.rng.choice(self.symbols), when, n))
                n += 1
                next_news += self.news_every
            if self.until is not None and when > self.until:
                break
            # émet tout ce qui est reçu avant le prochain point de génération
            horizon = min(next_earnings, next_news)
            pending.sort(key=lambda e: e.received_at)
            while pending and pending[0].received_at < horizon:
                ev = pending.pop(0)
                if self.until is not None and ev.received_at > self.until:
                    return
                yield ev
            await asyncio.sleep(0)
