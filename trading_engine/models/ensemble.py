"""Ensemble de modèles prédictifs (README §50.3).

L'accord entre modèles ne vaut que si leurs **erreurs** sont indépendantes.
On suit donc la covariance EWMA des erreurs hors échantillon des modèles
(sur les résultats où tous avaient une prévision) et :

- pondération : w ∝ Sigma_e^{-1} 1 (combinaison de variance minimale),
  ramenée vers l'équipondération (shrinkage), sans poids négatif ;
- nombre effectif de modèles indépendants : N_eff = 1' R^{-1} 1 (R =
  corrélation des erreurs) ; deux modèles corrélés à 0.9 ≈ 1,05 modèle ;
- incertitude combinée : sqrt(w' S R S w) avec S = diag(std prédictifs),
  puis le **désaccord** des moyennes l'élargit :
      sigma_eff^2 = sigma_comb^2 + sum_i w_i (mu_i - mu)^2
- fiabilité combinée : moyenne pondérée des fiabilités de contexte.

Tant qu'il y a trop peu de résultats communs, les erreurs sont supposées
parfaitement corrélées (hypothèse prudente : aucun bénéfice de diversification).
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from trading_engine.signals.signal import Signal


class ModelEnsemble:
    def __init__(
        self,
        names: list[str],
        *,
        lam: float = 0.99,
        shrinkage: float = 0.5,
        min_samples: int = 30,
    ) -> None:
        if not names:
            raise ValueError("at least one model is required")
        self.names = list(names)
        self.lam = lam
        self.shrinkage = shrinkage
        self.min_samples = min_samples
        k = len(names)
        self._cov = np.zeros((k, k))
        self.n = 0

    def record(self, errors: Mapping[str, float]) -> None:
        """Erreurs hors échantillon des modèles pour un même résultat réalisé."""
        if any(name not in errors for name in self.names):
            return
        e = np.array([errors[name] for name in self.names])
        if not np.all(np.isfinite(e)):
            return
        outer = np.outer(e, e)
        self._cov = outer if self.n == 0 else self.lam * self._cov + (1 - self.lam) * outer
        self.n += 1

    @property
    def ready(self) -> bool:
        return self.n >= self.min_samples

    def correlation(self) -> np.ndarray:
        k = len(self.names)
        if not self.ready:
            return np.ones((k, k))            # prudence : aucune diversification supposée
        std = np.sqrt(np.clip(np.diag(self._cov), 1e-300, None))
        corr = np.clip(self._cov / np.outer(std, std), -1.0, 1.0)
        np.fill_diagonal(corr, 1.0)
        return corr

    def effective_models(self) -> float:
        corr = self.correlation()
        ones = np.ones(len(self.names))
        n_eff = float(ones @ np.linalg.pinv(corr) @ ones)
        return float(np.clip(n_eff, 1.0, len(self.names)))

    def weights(self) -> np.ndarray:
        k = len(self.names)
        equal = np.full(k, 1.0 / k)
        if not self.ready:
            return equal
        cov = (1 - self.shrinkage) * self._cov + self.shrinkage * np.diag(np.diag(self._cov))
        raw = np.linalg.pinv(cov) @ np.ones(k)
        raw = np.clip(raw, 0.0, None)
        if raw.sum() <= 0:
            return equal
        w = raw / raw.sum()
        return (1 - self.shrinkage) * w + self.shrinkage * equal

    def combine(self, signals: Mapping[str, Signal | None], source: str = "ensemble") -> Signal | None:
        available = [i for i, name in enumerate(self.names) if signals.get(name) is not None]
        if not available:
            return None
        sigs = [signals[self.names[i]] for i in available]
        w = self.weights()[available]
        w = w / w.sum()
        corr = self.correlation()[np.ix_(available, available)]
        means = np.array([s.mean for s in sigs])
        stds = np.array([s.std for s in sigs])
        mean = float(w @ means)
        model_var = float((w * stds) @ corr @ (w * stds))
        disagreement = float(w @ (means - mean) ** 2)
        rels = [s.reliability for s in sigs]
        reliability = (
            None if any(r is None for r in rels) else float(w @ np.array(rels, dtype=float))
        )
        return Signal(
            symbol=sigs[0].symbol,
            horizon=sigs[0].horizon,
            mean=mean,
            std=math.sqrt(model_var + disagreement),
            n_obs=min(s.n_obs for s in sigs),
            timestamp=max(s.timestamp for s in sigs),
            source=source,
            reliability=reliability,
            disagreement=math.sqrt(disagreement),
        )
