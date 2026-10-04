"""Surveillance de la qualité et de la stabilité d'un modèle online (README §49.1).

Pour chaque prévision dont le résultat est connu :

- skill      : 1 - MSE(modèle) / MSE(prévision nulle), en EWMA. <= 0 veut dire
               que le modèle ne bat pas « prédire zéro » ;
- coverage   : fraction des résultats dans ±1 écart-type prédictif (≈ 0.68 si
               l'incertitude est bien calibrée) ;
- dérive     : test de Page-Hinkley sur l'erreur standardisée au carré
               (espérance 1 si le modèle est calibré) ; une hausse durable
               signale un changement de régime / concept drift.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PageHinkley:
    """Détecte une hausse durable de la moyenne d'une série."""

    delta: float = 0.5        # tolérance autour de la moyenne
    threshold: float = 50.0   # seuil de détection
    mean: float = 0.0
    n: int = 0
    cumulative: float = 0.0
    minimum: float = 0.0

    def update(self, value: float) -> bool:
        self.n += 1
        self.mean += (value - self.mean) / self.n
        self.cumulative += value - self.mean - self.delta
        self.minimum = min(self.minimum, self.cumulative)
        if self.cumulative - self.minimum > self.threshold:
            self.reset()
            return True
        return False

    def reset(self) -> None:
        self.mean = 0.0
        self.n = 0
        self.cumulative = 0.0
        self.minimum = 0.0


class ModelMonitor:
    def __init__(self, lam: float = 0.98, drift: PageHinkley | None = None) -> None:
        self.lam = lam
        self.mse_model: float | None = None
        self.mse_zero: float | None = None
        self.coverage: float | None = None
        self.drift = drift or PageHinkley()
        self.n = 0
        self.drift_count = 0

    def _ewma(self, current: float | None, value: float) -> float:
        return value if current is None else self.lam * current + (1 - self.lam) * value

    def update(self, realized: float, mean: float, std: float) -> bool:
        """Enregistre un résultat ; retourne True si une dérive est détectée."""
        err = realized - mean
        self.mse_model = self._ewma(self.mse_model, err * err)
        self.mse_zero = self._ewma(self.mse_zero, realized * realized)
        self.coverage = self._ewma(self.coverage, 1.0 if abs(err) <= std else 0.0)
        self.n += 1
        drifted = self.drift.update((err / std) ** 2)
        if drifted:
            self.drift_count += 1
        return drifted

    @property
    def skill(self) -> float | None:
        if not self.mse_zero:
            return None
        return 1.0 - self.mse_model / self.mse_zero
