"""Risk parity / risk budgeting (README §18).

Cherche w >= 0 tel que chaque actif porte sa part de risque b_i :

    w_i (Sigma w)_i / (w' Sigma w) = b_i

Résolu par descente de coordonnées cyclique (Griveau-Billion, Richard,
Roncalli 2013) sur le problème convexe min 0.5 w'Sigma w - sum b_i log w_i,
puis normalisé à sum(w) = 1. Les actifs de budget nul ont un poids nul.
"""

from __future__ import annotations

import math

import numpy as np


def risk_budget_weights(
    cov: np.ndarray,
    budgets: np.ndarray | None = None,
    *,
    max_iter: int = 500,
    tol: float = 1e-10,
) -> np.ndarray:
    n = cov.shape[0]
    b = np.full(n, 1.0 / n) if budgets is None else np.asarray(budgets, dtype=float)
    if np.any(b < 0) or b.sum() <= 0:
        raise ValueError("risk budgets must be non-negative with a positive sum")
    b = b / b.sum()
    active = (b > 0) & (np.diag(cov) > 0)
    w = np.zeros(n)
    if not active.any():
        return w
    idx = np.flatnonzero(active)
    sub = cov[np.ix_(idx, idx)]
    bs = b[idx]
    x = 1.0 / np.sqrt(np.diag(sub))
    x /= x.sum()
    for _ in range(max_iter):
        previous = x.copy()
        for i in range(len(idx)):
            s_ii = sub[i, i]
            cross = float(sub[i] @ x - s_ii * x[i])
            x[i] = (-cross + math.sqrt(cross * cross + 4.0 * s_ii * bs[i])) / (2.0 * s_ii)
        if np.max(np.abs(x - previous)) < tol * max(1.0, np.max(np.abs(x))):
            break
    w[idx] = x / x.sum()
    return w


def inverse_volatility_weights(cov: np.ndarray) -> np.ndarray:
    vol = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    inv = np.where(vol > 0, 1.0 / np.where(vol > 0, vol, 1.0), 0.0)
    total = inv.sum()
    return inv / total if total > 0 else inv
