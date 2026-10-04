"""HMM gaussien (covariances diagonales) : Baum-Welch et filtrage online (README §9).

Le HMM fournit une **estimation probabiliste** d'un état latent,
P(S_t | x_{1:t}) — pas une vérité sur le marché (README §49.3).

- `fit` : Baum-Welch (forward-backward normalisé) sur une fenêtre ;
- `filter_step` : mise à jour online de P(S_t | x_{1:t}) en O(K^2).

Régularisation : variance plancher, et pseudo-comptes « collants » sur la
diagonale de la matrice de transition pour éviter les régimes qui
clignotent d'une barre à l'autre.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

LOG_2PI = math.log(2.0 * math.pi)


@dataclass(frozen=True)
class HMMParams:
    startprob: np.ndarray   # (K,)
    transmat: np.ndarray    # (K, K)
    means: np.ndarray       # (K, D)
    variances: np.ndarray   # (K, D)

    @property
    def n_states(self) -> int:
        return len(self.startprob)

    def permute(self, order: np.ndarray) -> "HMMParams":
        return HMMParams(
            self.startprob[order],
            self.transmat[np.ix_(order, order)],
            self.means[order],
            self.variances[order],
        )


def log_emissions(X: np.ndarray, params: HMMParams) -> np.ndarray:
    """log N(x_t | mu_k, diag(var_k)) pour chaque t, k -> (T, K)."""
    X = np.atleast_2d(X)
    var = params.variances
    diff = X[:, None, :] - params.means[None, :, :]
    return -0.5 * (np.sum(np.log(var) + LOG_2PI, axis=1)[None, :] + np.sum(diff**2 / var[None], axis=2))


def _forward_backward(X: np.ndarray, params: HMMParams):
    logB = log_emissions(X, params)
    shift = logB.max(axis=1, keepdims=True)
    B = np.exp(logB - shift)
    T, K = B.shape
    A = params.transmat

    alpha = np.empty((T, K))
    scale = np.empty(T)
    a = params.startprob * B[0]
    scale[0] = a.sum()
    alpha[0] = a / scale[0]
    for t in range(1, T):
        a = (alpha[t - 1] @ A) * B[t]
        scale[t] = a.sum()
        alpha[t] = a / scale[t]

    beta = np.empty((T, K))
    beta[-1] = 1.0
    for t in range(T - 2, -1, -1):
        beta[t] = A @ (B[t + 1] * beta[t + 1]) / scale[t + 1]

    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)

    # xi = Σ_t alpha_t ⊗ (B_{t+1} β_{t+1} / c_{t+1}) ∘ A, en un produit matriciel
    xi = (alpha[:-1].T @ (B[1:] * beta[1:] / scale[1:, None])) * A

    step_ll = np.log(scale) + shift[:, 0]     # log p(x_t | x_{1:t-1})
    return gamma, xi, alpha, float(step_ll.sum()), step_ll


def initial_params(X: np.ndarray, n_states: int, sort_dim: int = -1, stay: float = 0.9) -> HMMParams:
    """Initialisation par quantiles de la dimension `sort_dim` (ex. volatilité)."""
    order = np.argsort(X[:, sort_dim], kind="stable")
    chunks = np.array_split(X[order], n_states)
    means = np.array([c.mean(axis=0) for c in chunks])
    global_var = X.var(axis=0) + 1e-12
    variances = np.array([np.maximum(c.var(axis=0), 1e-3 * global_var) for c in chunks])
    K = n_states
    transmat = np.full((K, K), (1 - stay) / max(K - 1, 1))
    np.fill_diagonal(transmat, stay if K > 1 else 1.0)
    return HMMParams(np.full(K, 1.0 / K), transmat, means, variances)


def fit(
    X: np.ndarray,
    n_states: int,
    *,
    init: HMMParams | None = None,
    n_iter: int = 25,
    tol: float = 1e-4,
    sticky: float = 5.0,
    min_var_ratio: float = 1e-3,
    sort_dim: int = -1,
) -> tuple[HMMParams, float]:
    """Baum-Welch ; retourne (params, log-vraisemblance)."""
    X = np.asarray(X, dtype=float)
    if len(X) < 2 * n_states:
        raise ValueError(f"not enough observations ({len(X)}) for {n_states} states")
    params = init if init is not None else initial_params(X, n_states, sort_dim)
    var_floor = min_var_ratio * (X.var(axis=0) + 1e-12)
    prev = -np.inf
    loglik = prev
    for _ in range(n_iter):
        gamma, xi, _, loglik, _ = _forward_backward(X, params)
        weights = gamma.sum(axis=0) + 1e-12
        means = (gamma.T @ X) / weights[:, None]
        variances = (gamma.T @ X**2) / weights[:, None] - means**2
        variances = np.maximum(variances, var_floor)
        counts = xi + sticky * np.eye(n_states) + 1e-6
        transmat = counts / counts.sum(axis=1, keepdims=True)
        startprob = gamma[0] + 1e-6
        params = HMMParams(startprob / startprob.sum(), transmat, means, variances)
        if loglik - prev < tol * abs(loglik):
            break
        prev = loglik
    return params, loglik


def filter_probs(X: np.ndarray, params: HMMParams) -> np.ndarray:
    """P(S_T | x_{1:T}) sur une séquence (filtrage, pas de lissage)."""
    _, _, alpha, _, _ = _forward_backward(np.asarray(X, dtype=float), params)
    return alpha[-1]


def step_logliks(X: np.ndarray, params: HMMParams) -> np.ndarray:
    """log p(x_t | x_{1:t-1}) pour chaque t : vraisemblance prédictive pas à pas."""
    return _forward_backward(np.asarray(X, dtype=float), params)[4]


def filter_step(prev: np.ndarray, x: np.ndarray, params: HMMParams) -> np.ndarray:
    """Une étape de filtrage online : P(S_t | x_{1:t}) à partir de P(S_{t-1} | x_{1:t-1})."""
    return filter_step_ll(prev, x, params)[0]


def filter_step_ll(prev: np.ndarray, x: np.ndarray, params: HMMParams) -> tuple[np.ndarray, float]:
    """Filtrage online + log p(x_t | x_{1:t-1}) (compatibilité des données avec le modèle)."""
    logb = log_emissions(np.asarray(x, dtype=float)[None, :], params)[0]
    shift = logb.max()
    post = (prev @ params.transmat) * np.exp(logb - shift)
    total = post.sum()
    if not np.isfinite(total) or total <= 0:
        return np.full_like(prev, 1.0 / len(prev)), -np.inf
    return post / total, float(np.log(total) + shift)
