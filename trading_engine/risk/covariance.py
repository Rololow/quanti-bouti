"""Matrice de covariance annualisée pour le risque et l'allocation (README §13, §16).

Source : covariance EWMA du `FeatureEngine` sur un timeframe de barres.
Traitements pour la stabilité numérique :

- shrinkage vers la diagonale : Sigma = (1 - s) Sigma + s diag(Sigma), ce qui
  réduit le bruit d'estimation des corrélations ;
- projection sur les matrices semi-définies positives (valeurs propres >= 0) ;
- actif sans historique de covariance mais avec une volatilité connue :
  variance seule, corrélations nulles.
"""

from __future__ import annotations

import math

import numpy as np

from trading_engine.timeutils import periods_per_year
from trading_engine.features.feature_engine import FeatureEngine


def nearest_psd(cov: np.ndarray, floor: float = 0.0) -> np.ndarray:
    cov = 0.5 * (cov + cov.T)
    values, vectors = np.linalg.eigh(cov)
    values = np.maximum(values, floor)
    return (vectors * values) @ vectors.T


def shrink_to_diagonal(cov: np.ndarray, intensity: float) -> np.ndarray:
    if not 0.0 <= intensity <= 1.0:
        raise ValueError(f"shrinkage must be in [0, 1], got {intensity}")
    return (1.0 - intensity) * cov + intensity * np.diag(np.diag(cov))


def annualized_covariance(
    features: FeatureEngine,
    symbols: list[str],
    timeframe: str,
    *,
    shrinkage: float = 0.1,
    min_updates: int = 2,
) -> np.ndarray | None:
    """Covariance annualisée alignée sur `symbols` ; None si trop peu de données."""
    scale = periods_per_year(timeframe)
    n = len(symbols)
    cov = np.zeros((n, n))
    known = np.zeros(n, dtype=bool)

    # Trop peu d'observations : pas de covariance du tout (une allocation
    # calculée sur 2 barres donnerait des poids extrêmes).
    if timeframe not in features.covariance_timeframes or features.correlation_updates(timeframe) < min_updates:
        return None
    cov_symbols, raw = features.covariance(timeframe)
    index = {sym: i for i, sym in enumerate(cov_symbols)}
    idx = [index.get(sym) for sym in symbols]
    for a, ia in enumerate(idx):
        if ia is None:
            continue
        known[a] = True
        for b, ib in enumerate(idx):
            if ib is not None:
                cov[a, b] = raw[ia, ib] * scale

    for a, sym in enumerate(symbols):
        if not known[a]:
            vol = features.volatility.get(sym, timeframe)
            if vol is not None:
                cov[a, a] = vol * vol * scale
                known[a] = True
    if not known.any():
        return None
    return nearest_psd(shrink_to_diagonal(cov, shrinkage))


def annualized_vol(variance: float) -> float:
    return math.sqrt(max(variance, 0.0))
