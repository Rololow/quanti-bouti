"""Combinaison de plusieurs sources d'événements en un seul flux.

- `TimeMergedFeed` : sources finies ou déterministes (simulation, fichiers,
  replay) : fusion par `received_at` (ordre de réception), stable et
  reproductible ;
- `ConcurrentFeed` : sources live (WebSocket marché + news + poller
  fondamentaux) : chaque source tourne en tâche, les événements sont émis
  dans l'ordre d'arrivée. C'est cet ordre que le journal enregistre, donc
  le replay reste identique.
"""

from __future__ import annotations

import asyncio
import heapq
from typing import AsyncIterator, Sequence

from trading_engine.data.events import Event
from trading_engine.data.market_feed import MarketFeed


class TimeMergedFeed(MarketFeed):
    def __init__(self, feeds: Sequence[MarketFeed], *, stop_with_first: bool = True) -> None:
        """`stop_with_first` : s'arrête quand la première source (le marché) est
        épuisée, même si des sources auxiliaires sont infinies."""
        if not feeds:
            raise ValueError("at least one feed is required")
        self.feeds = list(feeds)
        self.stop_with_first = stop_with_first

    async def __aiter__(self) -> AsyncIterator[Event]:
        iterators = [feed.__aiter__() for feed in self.feeds]
        heap: list[tuple] = []

        async def advance(i: int) -> None:
            try:
                ev = await iterators[i].__anext__()
            except StopAsyncIteration:
                return
            heapq.heappush(heap, (ev.received_at, ev.timestamp, i, id(ev), ev))

        for i in range(len(iterators)):
            await advance(i)
        primary_done = False
        while heap:
            _, _, i, _, ev = heapq.heappop(heap)
            yield ev
            before = len(heap)
            await advance(i)
            if i == 0 and len(heap) == before:
                primary_done = True
            if primary_done and self.stop_with_first:
                return


class ConcurrentFeed(MarketFeed):
    def __init__(self, feeds: Sequence[MarketFeed], queue_size: int = 10_000) -> None:
        self.feeds = list(feeds)
        self.queue_size = queue_size

    async def __aiter__(self) -> AsyncIterator[Event]:
        queue: asyncio.Queue = asyncio.Queue(self.queue_size)
        done = object()

        async def pump(feed: MarketFeed) -> None:
            try:
                async for ev in feed:
                    await queue.put(ev)
            finally:
                await queue.put(done)

        tasks = [asyncio.create_task(pump(f)) for f in self.feeds]
        remaining = len(tasks)
        try:
            while remaining:
                item = await queue.get()
                if item is done:
                    remaining -= 1
                    continue
                yield item
        finally:
            for t in tasks:
                t.cancel()
