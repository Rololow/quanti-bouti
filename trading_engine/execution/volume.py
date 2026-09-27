"""Suivi du volume de marché pour estimer la participation (README §32)."""

from __future__ import annotations

from datetime import timedelta

from trading_engine.timeutils import TRADING_HOURS_PER_DAY, parse_timeframe


class VolumeTracker:
    """Volume EWMA par barre `timeframe` -> débit de marché (titres / seconde)."""

    def __init__(self, timeframe: str = "5m", lam: float = 0.95) -> None:
        self.timeframe = timeframe
        self.step = parse_timeframe(timeframe)
        self.lam = lam
        self._volume: dict[str, float] = {}

    def update(self, symbol: str, volume: float) -> None:
        if volume < 0:
            return
        prev = self._volume.get(symbol)
        self._volume[symbol] = volume if prev is None else self.lam * prev + (1 - self.lam) * volume

    def rate(self, symbol: str) -> float | None:
        vol = self._volume.get(symbol)
        return None if vol is None else vol / self.step.total_seconds()

    def expected_volume(self, symbol: str, duration: timedelta) -> float | None:
        rate = self.rate(symbol)
        return None if rate is None else rate * duration.total_seconds()

    def adv(self, symbol: str) -> float | None:
        """Volume journalier moyen estimé (séance de 6h30)."""
        return self.expected_volume(symbol, timedelta(hours=TRADING_HOURS_PER_DAY))
