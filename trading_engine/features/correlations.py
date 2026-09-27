"""Covariance / corrélation online entre actifs (README §13).

    Sigma_t = lambda * Sigma_{t-1} + (1 - lambda) * r_t r_t^T

Les rendements des différents actifs doivent être synchronisés : on les
calcule sur des barres de même début (`BarReturnAligner`). Un actif sans barre
sur un intervalle a un rendement nul (prix inchangé).
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Mapping

import numpy as np

from trading_engine.data.events import BarEvent


class BarReturnAligner:
    """Regroupe les barres d'un timeframe par intervalle et émet un vecteur de rendements.

    Un intervalle est finalisé quand arrive une barre d'un intervalle plus
    récent ; les barres en retard sur un intervalle déjà finalisé sont ignorées.
    """

    def __init__(self) -> None:
        self.last_close: dict[str, float] = {}
        self.bucket_start: datetime | None = None
        self.bucket_closes: dict[str, float] = {}

    def on_bar(self, bar: BarEvent) -> dict[str, float] | None:
        if self.bucket_start is not None and bar.timestamp < self.bucket_start:
            return None
        out = None
        if self.bucket_start is not None and bar.timestamp > self.bucket_start:
            out = self.finalize()
        self.bucket_start = bar.timestamp
        self.bucket_closes[bar.symbol] = bar.close
        return out

    def finalize(self) -> dict[str, float] | None:
        returns = {
            sym: math.log(self.bucket_closes[sym] / prev) if sym in self.bucket_closes else 0.0
            for sym, prev in self.last_close.items()
        }
        self.last_close.update(self.bucket_closes)
        self.bucket_closes = {}
        return returns or None


class EWMACovariance:
    def __init__(self, lam: float = 0.94) -> None:
        if not 0.0 < lam < 1.0:
            raise ValueError(f"lambda must be in (0, 1), got {lam}")
        self.lam = lam
        self.symbols: list[str] = []
        self._index: dict[str, int] = {}
        self._cov = np.zeros((0, 0))
        self.n_updates = 0

    def _add_symbols(self, new: list[str], r: np.ndarray) -> None:
        old_n = len(self.symbols)
        for sym in new:
            self._index[sym] = len(self.symbols)
            self.symbols.append(sym)
        cov = np.zeros((len(self.symbols), len(self.symbols)))
        cov[:old_n, :old_n] = self._cov
        # Nouveaux actifs : initialisés avec leur première observation.
        idx = np.arange(old_n, len(self.symbols))
        cov[idx, :] = np.outer(r[idx], r)
        cov[:, idx] = np.outer(r, r[idx])
        self._cov = cov

    def update(self, returns: Mapping[str, float]) -> None:
        new = sorted(sym for sym in returns if sym not in self._index)
        n = len(self.symbols) + len(new)
        r = np.zeros(n)
        for sym, i in self._index.items():
            r[i] = returns.get(sym, 0.0)
        for k, sym in enumerate(new):
            r[len(self.symbols) + k] = returns[sym]
        if new:
            first = self.n_updates == 0
            self._add_symbols(new, r)
            if first:
                self.n_updates = 1
                return
        self._cov = self.lam * self._cov + (1.0 - self.lam) * np.outer(r, r)
        self.n_updates += 1

    def covariance(self) -> tuple[list[str], np.ndarray]:
        return list(self.symbols), self._cov.copy()

    def volatilities(self) -> dict[str, float]:
        return {sym: math.sqrt(max(self._cov[i, i], 0.0)) for sym, i in self._index.items()}

    def correlation(self) -> tuple[list[str], np.ndarray]:
        std = np.sqrt(np.clip(np.diag(self._cov), 0.0, None))
        denom = np.outer(std, std)
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.where(denom > 0, self._cov / denom, 0.0)
        np.fill_diagonal(corr, 1.0)
        return list(self.symbols), np.clip(corr, -1.0, 1.0)

    def pair(self, a: str, b: str) -> float | None:
        if a not in self._index or b not in self._index:
            return None
        _, corr = self.correlation()
        return float(corr[self._index[a], self._index[b]])
