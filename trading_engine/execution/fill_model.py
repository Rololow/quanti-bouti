"""Probabilité d'exécution d'un ordre limite (README §32 : P_fill(p, q, Δt)).

- prix : un ordre passif placé à une distance d du mid n'est exécuté que si
  le prix l'atteint pendant Δt. Pour une marche aléatoire de volatilité
  σ_Δt, le principe de réflexion donne
      P(touch) = 2 · (1 - Φ(d / σ_Δt))          (d > 0)
  Un ordre qui croise le spread (d <= 0) est exécutable immédiatement ;
- quantité : même touché, on ne peut raisonnablement prendre qu'une part du
  volume de marché : fraction = min(1, max_participation · V_marché / Q) ;
- calibration : un facteur appris par le feedback corrige le modèle (fills
  observés / prédits).
"""

from __future__ import annotations

import math


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


class FillModel:
    def __init__(self, max_participation: float = 0.10) -> None:
        self.max_participation = max_participation
        self.calibration = 1.0          # appris : fill observé / fill prédit

    def touch_probability(self, distance: float, sigma_horizon: float) -> float:
        """distance : écart relatif entre le prix limite et le prix exécutable
        (> 0 = il faut que le marché vienne à nous)."""
        if distance <= 0:
            return 1.0
        if sigma_horizon <= 0:
            return 0.0
        return min(1.0, 2.0 * (1.0 - _phi(distance / sigma_horizon)))

    def expected_fill(
        self,
        quantity: float,
        distance: float,
        sigma_horizon: float,
        expected_market_volume: float | None,
    ) -> float:
        p_touch = self.touch_probability(distance, sigma_horizon)
        capacity = 1.0
        if expected_market_volume is not None and quantity:
            capacity = min(1.0, self.max_participation * expected_market_volume / abs(quantity))
        return max(0.0, min(1.0, p_touch * capacity * self.calibration))
