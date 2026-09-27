"""Événements lus depuis un fichier JSONL (même format que le journal) :
fondamentaux historiques, news archivées, calendrier d'earnings…

Les événements sont émis dans l'ordre de `received_at`. Une base historique
peut contenir une valeur « reçue » avant sa publication (`available_at`) :
le moteur ne l'utilisera qu'à partir de `available_at` (README §44).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncIterator

from trading_engine.data.events import Event
from trading_engine.data.market_feed import MarketFeed
from trading_engine.storage.event_log import read_events


class EventFileFeed(MarketFeed):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"event file not found: {self.path}")

    async def __aiter__(self) -> AsyncIterator[Event]:
        events = sorted(read_events(self.path), key=lambda e: e.received_at)
        for i, ev in enumerate(events):
            yield ev
            if i % 1000 == 0:
                await asyncio.sleep(0)
