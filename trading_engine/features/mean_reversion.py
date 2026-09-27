"""Features de mean reversion, surtout utiles sur les horizons courts (README §12)."""

from __future__ import annotations

import math
from datetime import date, datetime

from trading_engine.features.returns import PriceHistory


def zscore(history: PriceHistory, window: int) -> float | None:
    """(p_t - moyenne) / écart-type sur les `window` derniers prix."""
    prices = history.window(window)
    if prices is None or window < 2:
        return None
    mean = math.fsum(prices) / window
    var = math.fsum((p - mean) ** 2 for p in prices) / window
    if var <= 0:
        return 0.0
    return (prices[-1] - mean) / math.sqrt(var)


def ma_distance(history: PriceHistory, window: int) -> float | None:
    """Distance relative du prix à sa moyenne mobile : p / MA - 1."""
    prices = history.window(window)
    if prices is None:
        return None
    return prices[-1] / (math.fsum(prices) / window) - 1.0


def reversal(history: PriceHistory, n: int = 1) -> float | None:
    """Short-term reversal : opposé du rendement récent."""
    ret = history.log_return(n)
    return None if ret is None else -ret


class SessionVWAP:
    """VWAP de la séance, remis à zéro à chaque changement de jour UTC."""

    def __init__(self) -> None:
        self.session: date | None = None
        self.notional = 0.0
        self.volume = 0.0

    def update(self, timestamp: datetime, price: float, size: float) -> float | None:
        if self.session != timestamp.date():
            self.session = timestamp.date()
            self.notional = 0.0
            self.volume = 0.0
        if size > 0:
            self.notional += price * size
            self.volume += size
        return self.value

    @property
    def value(self) -> float | None:
        return self.notional / self.volume if self.volume > 0 else None

    def distance(self, price: float) -> float | None:
        vwap = self.value
        return None if vwap is None else price / vwap - 1.0
