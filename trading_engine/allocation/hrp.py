"""Hierarchical Risk Parity (López de Prado, 2016) — README §18.

1. distance de corrélation d_ij = sqrt((1 - rho_ij) / 2) ;
2. clustering hiérarchique (single linkage) -> ordre quasi-diagonal ;
3. bissection récursive : le poids est réparti entre les deux moitiés en
   proportion inverse de leur variance (portefeuille inverse-variance).

Pas d'inversion de matrice : robuste quand la covariance est mal estimée.
"""

from __future__ import annotations

import numpy as np


def correlation_distance(cov: np.ndarray) -> np.ndarray:
    std = np.sqrt(np.clip(np.diag(cov), 1e-300, None))
    corr = np.clip(cov / np.outer(std, std), -1.0, 1.0)
    return np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, None))


def quasi_diagonal_order(dist: np.ndarray) -> list[int]:
    """Ordre des feuilles d'un clustering single-linkage (agglomératif, O(n^3))."""
    n = dist.shape[0]
    clusters: dict[int, list[int]] = {i: [i] for i in range(n)}
    d = dist.astype(float).copy()
    np.fill_diagonal(d, np.inf)
    alive = list(range(n))
    while len(alive) > 1:
        sub = d[np.ix_(alive, alive)]
        a_pos, b_pos = np.unravel_index(np.argmin(sub), sub.shape)
        a, b = alive[a_pos], alive[b_pos]
        if a > b:
            a, b = b, a
        clusters[a] = clusters[a] + clusters.pop(b)
        # single linkage : distance au nouveau cluster = min des distances
        d[a, :] = np.minimum(d[a, :], d[b, :])
        d[:, a] = d[a, :]
        d[a, a] = np.inf
        alive.remove(b)
    return clusters[alive[0]]


def _cluster_variance(cov: np.ndarray, items: list[int]) -> float:
    sub = cov[np.ix_(items, items)]
    inv = 1.0 / np.diag(sub)
    w = inv / inv.sum()
    return float(w @ sub @ w)


def hrp_weights(cov: np.ndarray) -> np.ndarray:
    n = cov.shape[0]
    weights = np.zeros(n)
    active = np.flatnonzero(np.diag(cov) > 0)
    if active.size == 0:
        return weights
    sub = cov[np.ix_(active, active)]
    order = quasi_diagonal_order(correlation_distance(sub))
    w = np.ones(len(active))
    stack = [order]
    while stack:
        items = stack.pop()
        if len(items) < 2:
            continue
        half = len(items) // 2
        left, right = items[:half], items[half:]
        var_l, var_r = _cluster_variance(sub, left), _cluster_variance(sub, right)
        alpha = 1.0 - var_l / (var_l + var_r)
        w[left] *= alpha
        w[right] *= 1.0 - alpha
        stack.extend([left, right])
    weights[active] = w / w.sum()
    return weights
