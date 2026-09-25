"""Volatilité EWMA online (README §10).

    sigma_t^2 = lambda * sigma_{t-1}^2 + (1 - lambda) * r_t^2

Aucun retraining, état compact (dernier prix + variance), mise à jour O(1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class EWMAVolatility:
    lam: float = 0.94
    variance: float | None = None
    last_price: float | None = None
    n_updates: int = 0

    def __post_init__(self) -> None:
        if not 0.0 < self.lam < 1.0:
            raise ValueError(f"lambda must be in (0, 1), got {self.lam}")

    def seed(self, prices: list[float]) -> None:
        """Initialise l'état depuis un petit historique (README §42)."""
        for price in prices:
            self.update(price)

    def update(self, price: float) -> float | None:
        """Ajoute un prix ; retourne la volatilité par période (None si indéfinie)."""
        if price <= 0:
            raise ValueError(f"price must be positive, got {price}")
        if self.last_price is not None:
            r = math.log(price / self.last_price)
            if self.variance is None:
                self.variance = r * r
            else:
                self.variance = self.lam * self.variance + (1.0 - self.lam) * r * r
            self.n_updates += 1
        self.last_price = price
        return self.volatility

    @property
    def volatility(self) -> float | None:
        return None if self.variance is None else math.sqrt(self.variance)

    def annualized(self, periods_per_year: float) -> float | None:
        vol = self.volatility
        return None if vol is None else vol * math.sqrt(periods_per_year)


class VolatilityBook:
    """Un estimateur EWMA par (symbole, horizon)."""

    def __init__(self, lam: float = 0.94) -> None:
        self.lam = lam
        self._models: dict[tuple[str, str], EWMAVolatility] = {}

    def model(self, symbol: str, horizon: str) -> EWMAVolatility:
        key = (symbol, horizon)
        if key not in self._models:
            self._models[key] = EWMAVolatility(self.lam)
        return self._models[key]

    def update(self, symbol: str, horizon: str, price: float) -> float | None:
        return self.model(symbol, horizon).update(price)

    def get(self, symbol: str, horizon: str) -> float | None:
        model = self._models.get((symbol, horizon))
        return None if model is None else model.volatility
