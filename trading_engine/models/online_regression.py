"""Régression linéaire online bayésienne (RLS avec oubli) — README §28, §49.1.

Garde-fous contre la non-stationnarité et l'apprentissage du bruit :

- forgetting factor `lam` : les observations anciennes pèsent lam^age ;
- régularisation ridge (précision a priori) et plafond de covariance, pour
  éviter l'explosion de P quand les features ne varient pas (wind-up) ;
- taille d'échantillon minimale avant toute prédiction ;
- prédiction probabiliste : moyenne et écart-type prédictif
  s^2 (1 + x' P x), qui combine bruit résiduel et incertitude des coefficients.
"""

from __future__ import annotations

import math

import numpy as np


class OnlineLinearRegression:
    def __init__(
        self,
        n_features: int,
        *,
        lam: float = 0.995,
        ridge: float = 1.0,
        min_samples: int = 50,
        fit_intercept: bool = True,
        noise_lam: float = 0.99,
    ) -> None:
        if not 0.0 < lam <= 1.0:
            raise ValueError(f"lam must be in (0, 1], got {lam}")
        if ridge <= 0:
            raise ValueError(f"ridge must be positive, got {ridge}")
        self.fit_intercept = fit_intercept
        self.dim = n_features + (1 if fit_intercept else 0)
        self.lam = lam
        self.ridge = ridge
        self.min_samples = min_samples
        self.noise_lam = noise_lam
        self.coef = np.zeros(self.dim)
        self.P = np.eye(self.dim) / ridge
        self._max_trace = self.dim / ridge
        self.noise_var: float | None = None
        self.n_updates = 0
        self.last_coef_change = 0.0

    def _design(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.concatenate(([1.0], x)) if self.fit_intercept else x

    @property
    def ready(self) -> bool:
        return self.n_updates >= self.min_samples and self.noise_var is not None

    def update(self, x: np.ndarray, y: float) -> float:
        """Intègre (x, y) ; retourne l'erreur a priori y - x'w."""
        z = self._design(x)
        Pz = self.P @ z
        gain = Pz / (self.lam + z @ Pz)
        err = float(y - z @ self.coef)
        delta = gain * err
        self.coef = self.coef + delta
        self.P = (self.P - np.outer(gain, Pz)) / self.lam
        self.P = 0.5 * (self.P + self.P.T)  # symétrie numérique
        trace = float(np.trace(self.P))
        if trace > self._max_trace:
            self.P *= self._max_trace / trace
        self.noise_var = (
            err * err if self.noise_var is None
            else self.noise_lam * self.noise_var + (1 - self.noise_lam) * err * err
        )
        self.last_coef_change = float(np.linalg.norm(delta))
        self.n_updates += 1
        return err

    def predict(self, x: np.ndarray) -> tuple[float, float] | None:
        """(moyenne, écart-type prédictif), ou None tant que le modèle n'est pas prêt."""
        if not self.ready:
            return None
        z = self._design(x)
        mean = float(z @ self.coef)
        var = self.noise_var * (1.0 + float(z @ self.P @ z))
        return mean, math.sqrt(max(var, 1e-18))

    def inflate_uncertainty(self, factor: float) -> None:
        """Augmente l'incertitude des coefficients (après une dérive détectée)
        pour ré-apprendre plus vite, dans la limite du plafond."""
        self.P *= factor
        trace = float(np.trace(self.P))
        if trace > self._max_trace:
            self.P *= self._max_trace / trace
