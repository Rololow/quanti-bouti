"""Financement du cash et du levier.

- cash positif : rémunéré au taux court (T-bill 3 mois, ^IRX), moins un écart ;
- cash négatif (levier : emprunt sur marge, ou financement implicite des
  futures) : coûte le taux court plus `borrow_spread`.

Les intérêts courent par jour calendaire (base 365), au taux du dernier jour
ouvré. La série de taux entre dans le moteur comme un `RatesEvent` journalisé
(backtest : fichier téléchargé avec le dataset ; live : taux fixe configuré).

Approximation du levier par futures : un future sur indice ou sur Treasuries
porte un coût de financement implicite proche du taux court ; le moteur le
représente par une position au comptant financée à T-bill + écart. Les appels
de marge, le roll des contrats et leur fiscalité propre ne sont pas modélisés.
"""

from __future__ import annotations

import bisect
from datetime import date
from typing import Mapping


class RateSeries:
    """Taux annuels (0.045 = 4,5 %) datés ; lookup au dernier taux connu."""

    def __init__(self, rates: Mapping[date, float], source: str = "unknown") -> None:
        if any(v != v or v < -0.05 or v > 0.5 for v in rates.values()):
            raise ValueError("implausible short rate in series")
        self._days = sorted(rates)
        self._values = [float(rates[d]) for d in self._days]
        self.source = source

    def __len__(self) -> int:
        return len(self._days)

    def rate(self, day: date) -> float:
        if not self._days:
            return 0.0
        i = bisect.bisect_right(self._days, day) - 1
        return self._values[max(i, 0)]

    def to_payload(self) -> dict:
        return {"source": self.source, "rates": [[d.isoformat(), v] for d, v in zip(self._days, self._values)]}

    @classmethod
    def from_payload(cls, payload: Mapping) -> "RateSeries":
        return cls({date.fromisoformat(d): float(v) for d, v in payload.get("rates", [])},
                   payload.get("source", "unknown"))

    @classmethod
    def fixed(cls, rate: float, day: date) -> "RateSeries":
        return cls({day: rate}, source="fixed")


def accrue(cash: float, rate: float, days: int, *, borrow_spread: float, credit_spread: float,
           credit_cash: bool = True) -> float:
    """Intérêts (signés) sur `days` jours : > 0 gagnés, < 0 payés."""
    if days <= 0 or cash == 0:
        return 0.0
    if cash > 0:
        return cash * max(0.0, rate - credit_spread) * days / 365.0 if credit_cash else 0.0
    return cash * (rate + borrow_spread) * days / 365.0
