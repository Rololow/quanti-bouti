"""Replay d'un journal d'événements (README §43 — event replay).

Même interface que les flux live : le moteur ne sait pas s'il tourne en live
ou en replay. Les événements sont rejoués dans l'ordre du journal, c'est-à-dire
l'ordre de réception d'origine.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import AsyncIterator

from trading_engine.data.events import MarketEvent
from trading_engine.data.market_feed import MarketFeed
from trading_engine.storage.event_log import read_events

logger = logging.getLogger(__name__)


class ReplayFeed(MarketFeed):
    def __init__(self, path: str | Path, *, max_events: int | None = None, yield_every: int = 1000) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"event log not found: {self.path}")
        self.max_events = max_events
        self.yield_every = max(1, yield_every)
        self.out_of_order = 0

    async def __aiter__(self) -> AsyncIterator[MarketEvent]:
        last_received = None
        market_events = 0
        for n, event in enumerate(read_events(self.path)):
            # max_events compte les événements de marché, comme les flux live
            # (les news / fondamentaux du journal ne consomment pas la limite).
            if isinstance(event, MarketEvent):
                if self.max_events is not None and market_events >= self.max_events:
                    return
                market_events += 1
            if last_received is not None and event.received_at < last_received:
                self.out_of_order += 1
            last_received = event.received_at
            yield event
            if n % self.yield_every == 0:
                await asyncio.sleep(0)  # laisse tourner les autres tâches asyncio
        if self.out_of_order:
            logger.warning("%d events out of receive order in %s", self.out_of_order, self.path)
