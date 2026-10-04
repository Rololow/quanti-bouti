"""Volatility targeting (README §18) : remet un portefeuille à l'échelle de la
volatilité cible, sans dépasser un budget d'exposition brute."""

from __future__ import annotations

import numpy as np

from trading_engine.risk.portfolio_risk import portfolio_vol


def volatility_target(
    weights: np.ndarray, cov: np.ndarray, target_vol: float, *, max_gross: float = 1.0
) -> tuple[np.ndarray, float]:
    """Retourne (poids mis à l'échelle, facteur d'échelle)."""
    vol = portfolio_vol(weights, cov)
    gross = float(np.abs(weights).sum())
    if vol <= 0 or gross <= 0:
        return weights.copy(), 1.0
    scale = min(target_vol / vol, max_gross / gross)
    return weights * scale, scale
