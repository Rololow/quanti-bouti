"""Agrégation des trades en barres OHLCV multi-horizon (README §6-7).

Une barre est émise lorsque le premier trade de l'intervalle suivant arrive
(ou via `flush`). Ainsi les modèles lents (1h, 1d) ne sont mis à jour qu'à la
clôture de leur barre, jamais à chaque tick.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from trading_engine.data.events import BarEvent, TradeEvent
from trading_engine.timeutils import floor_time, parse_timeframe


@dataclass
class _PartialBar:
    start: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    trade_count: int

    def add(self, price: float, size: float) -> None:
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.volume += size
        self.trade_count += 1


class BarBuilder:
    SOURCE = "bar_builder"

    def __init__(self, timeframes: tuple[str, ...] | list[str]) -> None:
        self.timeframes: dict[str, timedelta] = {tf: parse_timeframe(tf) for tf in timeframes}
        self._partial: dict[tuple[str, str], _PartialBar] = {}

    def on_trade(self, trade: TradeEvent) -> list[BarEvent]:
        """Intègre un trade et retourne les barres clôturées par celui-ci."""
        closed: list[BarEvent] = []
        for tf, step in self.timeframes.items():
            key = (trade.symbol, tf)
            start = floor_time(trade.timestamp, step)
            bar = self._partial.get(key)
            if bar is not None and start < bar.start:
                continue  # trade en retard sur une barre déjà clôturée
            if bar is not None and start > bar.start:
                closed.append(self._emit(trade.symbol, tf, step, bar))
                bar = None
            if bar is None:
                self._partial[key] = _PartialBar(
                    start, trade.price, trade.price, trade.price, trade.price, trade.size, 1
                )
            else:
                bar.add(trade.price, trade.size)
        return closed

    def flush(self, now: datetime | None = None) -> list[BarEvent]:
        """Émet les barres en cours dont la fin est <= `now` (toutes si None)."""
        closed: list[BarEvent] = []
        for (symbol, tf), bar in list(self._partial.items()):
            step = self.timeframes[tf]
            if now is None or bar.start + step <= now:
                closed.append(self._emit(symbol, tf, step, bar))
                del self._partial[(symbol, tf)]
        return closed

    def _emit(self, symbol: str, tf: str, step: timedelta, bar: _PartialBar) -> BarEvent:
        end = bar.start + step
        return BarEvent(
            timestamp=bar.start,
            end=end,
            received_at=end,
            symbol=symbol,
            source=self.SOURCE,
            timeframe=tf,
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            volume=bar.volume,
            trade_count=bar.trade_count,
        )
