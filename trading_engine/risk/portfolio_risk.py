"""Risque au niveau portefeuille (README §15-16, §49.12).

    sigma_p      = sqrt(w' Sigma w)
    MRC_i        = (Sigma w)_i / sigma_p
    RC_i         = w_i * MRC_i                     (somme = sigma_p)
    RC_i^rel     = w_i (Sigma w)_i / (w' Sigma w)  (somme = 1)

Une position de 10 % peut porter bien plus de 10 % du risque : on regarde
les contributions, pas seulement les poids nominaux.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections import deque
from datetime import date, datetime

import numpy as np


@dataclass(frozen=True)
class RiskDecomposition:
    portfolio_vol: float
    marginal: np.ndarray       # MRC_i
    contribution: np.ndarray   # RC_i
    relative: np.ndarray       # RC_i^rel


def portfolio_vol(weights: np.ndarray, cov: np.ndarray) -> float:
    return math.sqrt(max(float(weights @ cov @ weights), 0.0))


def risk_contributions(weights: np.ndarray, cov: np.ndarray) -> RiskDecomposition:
    weights = np.asarray(weights, dtype=float)
    sigma_w = cov @ weights
    variance = float(weights @ sigma_w)
    n = len(weights)
    if variance <= 0:
        zeros = np.zeros(n)
        return RiskDecomposition(0.0, zeros, zeros, zeros)
    vol = math.sqrt(variance)
    marginal = sigma_w / vol
    return RiskDecomposition(vol, marginal, weights * marginal, weights * sigma_w / variance)


def herfindahl(weights: np.ndarray) -> float:
    """Concentration des poids (somme des carrés des poids normalisés)."""
    w = np.abs(np.asarray(weights, dtype=float))
    total = w.sum()
    return float(np.sum((w / total) ** 2)) if total > 0 else 0.0


def effective_n(shares: np.ndarray) -> float:
    """Nombre effectif de paris : 1 / somme(parts^2)."""
    hhi = herfindahl(shares)
    return 1.0 / hhi if hhi > 0 else 0.0


def diversification_ratio(weights: np.ndarray, cov: np.ndarray) -> float | None:
    """sum(|w_i| sigma_i) / sigma_p ; 1 = aucune diversification."""
    vol = portfolio_vol(weights, cov)
    if vol <= 0:
        return None
    return float(np.abs(weights) @ np.sqrt(np.clip(np.diag(cov), 0.0, None))) / vol


class DrawdownTracker:
    """Drawdown courant et maximal d'une série de valeurs (portefeuille ou prix)."""

    def __init__(self) -> None:
        self.peak: float | None = None
        self.drawdown = 0.0
        self.max_drawdown = 0.0
        self.peak_time: datetime | None = None

    def update(self, value: float, timestamp: datetime | None = None) -> float:
        if value <= 0:
            return self.drawdown
        if self.peak is None or value > self.peak:
            self.peak = value
            self.peak_time = timestamp
        self.drawdown = value / self.peak - 1.0
        self.max_drawdown = min(self.max_drawdown, self.drawdown)
        return self.drawdown


class RollingDrawdown:
    """Drawdown par rapport au plus haut des `window_days` derniers jours
    calendaires (plus haut de chaque jour gardé).

    Le contrôle du drawdown sur le plus haut historique enferme le portefeuille
    en cash après une perte proche de la limite : l'exposition ne revient qu'avec
    un nouveau record, que le cash ne peut pas produire. Un plus haut glissant
    oublie l'ancien sommet et laisse l'exposition se reconstruire."""

    def __init__(self, window_days: int) -> None:
        if window_days < 1:
            raise ValueError(f"window_days must be >= 1, got {window_days}")
        self.window_days = window_days
        self._days: deque[tuple[date, float]] = deque()
        self.peak: float | None = None
        self.drawdown = 0.0

    def update(self, value: float, day: date) -> float:
        if value <= 0:
            return self.drawdown
        if self._days and self._days[-1][0] == day:
            if value > self._days[-1][1]:
                self._days[-1] = (day, value)
        else:
            self._days.append((day, value))
        expired = False
        while (day - self._days[0][0]).days >= self.window_days:
            self._days.popleft()
            expired = True
        if expired or self.peak is None or value > self.peak:
            self.peak = max(v for _, v in self._days)
        self.drawdown = value / self.peak - 1.0
        return self.drawdown
