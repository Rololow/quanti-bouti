"""Risque par position (README §15)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PositionRisk:
    symbol: str
    weight: float
    target_weight: float
    volatility: float | None           # annualisée
    marginal_risk: float | None        # MRC_i
    risk_contribution: float | None    # RC_i (en volatilité)
    risk_share: float | None           # RC_i relative (somme = 1)
    drawdown: float                    # du prix depuis son plus haut

    @property
    def drift(self) -> float:
        return self.weight - self.target_weight
