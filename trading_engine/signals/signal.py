"""Format commun des signaux (README §49.9-49.10, §50.5) :
prédiction + incertitude + fiabilité.

Un signal est une **prévision de rendement avec son incertitude**, pas un
score sans unité : `mean` et `std` sont des log-rendements sur `horizon`,
directement comparables aux coûts d'exécution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Signal:
    symbol: str
    horizon: str          # ex. "30m", "1d"
    mean: float           # rendement log attendu sur l'horizon
    std: float            # écart-type prédictif (incertitude totale)
    n_obs: int            # nombre d'observations ayant servi à l'apprendre
    timestamp: datetime
    source: str
    # Fiabilité récente du modèle dans le contexte courant (README §50.5) :
    # 0 = ne bat pas « prédire zéro », None = pas encore mesurée.
    reliability: float | None = None
    # Écart-type des prévisions des modèles d'un ensemble (déjà inclus dans std).
    disagreement: float | None = None

    def __post_init__(self) -> None:
        if not self.std > 0:
            raise ValueError(f"signal std must be positive, got {self.std}")

    @property
    def t_stat(self) -> float:
        """Rendement attendu en unités d'incertitude."""
        return self.mean / self.std

    @property
    def prob_positive(self) -> float:
        """P(r > 0) sous hypothèse gaussienne."""
        return 0.5 * (1.0 + math.erf(self.t_stat / math.sqrt(2.0)))

    def net_of(self, cost: float) -> float:
        """Rendement attendu net d'un coût (même unité que `mean`)."""
        return abs(self.mean) - cost
