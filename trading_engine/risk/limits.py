"""Limites de risque surveillées (README §15).

Le Risk Engine est indépendant des modèles de signal : il vérifie l'état
réel du portefeuille et signale les dépassements. Faire respecter les
contraintes sur la cible est le rôle du Constraint Engine (allocation).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskLimits:
    max_weight: float = 0.40
    max_gross_leverage: float = 1.0
    max_portfolio_vol: float = 0.20          # annualisée
    max_drawdown: float = 0.15               # en valeur absolue (15 %)
    max_risk_contribution: float = 0.50      # part relative du risque total
    max_position_drawdown: float = 0.25
    max_drift: float = 0.05                  # |poids - cible|


@dataclass(frozen=True)
class RiskBreach:
    kind: str                 # ex. "WEIGHT", "PORTFOLIO_VOL", "DRAWDOWN"
    symbol: str | None
    value: float
    limit: float

    @property
    def severity(self) -> float:
        """Dépassement relatif : 0.2 = 20 % au-dessus de la limite."""
        return abs(self.value) / self.limit - 1.0 if self.limit else float("inf")

    def __str__(self) -> str:
        target = f" {self.symbol}" if self.symbol else ""
        return f"{self.kind}{target}: {self.value:.3f} > {self.limit:.3f}"
