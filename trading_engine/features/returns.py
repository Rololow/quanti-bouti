"""Historique de prix et rendements logarithmiques (README §11)."""

from __future__ import annotations

import math
from collections import deque


class PriceHistory:
    """Fenêtre glissante des derniers prix d'une série (symbole × timeframe)."""

    def __init__(self, maxlen: int) -> None:
        if maxlen < 2:
            raise ValueError(f"maxlen must be >= 2, got {maxlen}")
        self.prices: deque[float] = deque(maxlen=maxlen)

    def __len__(self) -> int:
        return len(self.prices)

    @property
    def last(self) -> float | None:
        return self.prices[-1] if self.prices else None

    def update(self, price: float) -> float | None:
        """Ajoute un prix ; retourne le dernier log-rendement (None au premier prix)."""
        if price <= 0:
            raise ValueError(f"price must be positive, got {price}")
        self.prices.append(price)
        return self.log_return(1)

    def log_return(self, n: int) -> float | None:
        """log(p_t / p_{t-n}) ; None si l'historique est trop court."""
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if len(self.prices) <= n:
            return None
        return math.log(self.prices[-1] / self.prices[-1 - n])

    def window(self, n: int) -> list[float] | None:
        """Les `n` derniers prix, ou None s'il y en a moins."""
        if len(self.prices) < n:
            return None
        return list(self.prices)[-n:]
