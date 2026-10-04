"""Détection de régime online par HMM (README §9, §49.3, §50.4).

Cycle de vie d'un `RegimeModel` (un par symbole et par horizon HF / MT / LT) :

1. warm-up : accumulation de `min_samples` observations (historical seed) ;
2. premier ajustement Baum-Welch, puis filtrage online à chaque barre ;
3. ré-ajustement tous les `refit_every` barres sur une fenêtre glissante de
   `window` observations (oubli), en partant des paramètres courants.

Les états sont triés par volatilité croissante après chaque ajustement : le
label d'un état garde donc le même sens d'un ajustement à l'autre. La sortie
est une distribution de probabilités, jamais un seul régime.

MODEL_DEGRADED : on suit la vraisemblance prédictive hors échantillon
log p(x_t | x_{1:t-1}) de chaque nouvelle observation. Sa moyenne récente
(EWMA rapide) est comparée à sa moyenne longue (EWMA lente) ; l'écart est
normalisé par sa propre variance mesurée empiriquement, ce qui tient compte
de l'autocorrélation des observations (la référence dans l'échantillon
d'ajustement, elle, serait optimiste). Un écart inférieur à `-degraded_z`
signifie que les données ne ressemblent plus à ce que le modèle voit
d'habitude : l'état est signalé dégradé au lieu d'afficher une confiance élevée.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from trading_engine.models import hmm

DEFAULT_LABELS = {
    2: ("LOW_VOL", "HIGH_VOL"),
    3: ("LOW_VOL", "MID_VOL", "HIGH_VOL"),
    4: ("LOW_VOL", "MID_VOL", "HIGH_VOL", "EXTREME_VOL"),
}


@dataclass(frozen=True)
class RegimeState:
    horizon: str
    probabilities: dict[str, float]
    timestamp: datetime
    n_obs: int
    degraded: bool = False
    fit_zscore: float | None = None   # vraisemblance récente vs référence (négatif = moins bonne)

    @property
    def most_likely(self) -> str:
        return max(self.probabilities, key=self.probabilities.get)

    @property
    def confidence(self) -> float:
        return self.probabilities[self.most_likely]


class RegimeModel:
    # Observation : [rendement de la barre, log(volatilité EWMA)] ; la dernière
    # dimension (volatilité) sert à ordonner les états.
    VOL_DIM = -1

    def __init__(
        self,
        horizon: str,
        *,
        n_states: int = 3,
        min_samples: int = 100,
        window: int = 500,
        refit_every: int = 50,
        n_iter: int = 25,
        labels: tuple[str, ...] | None = None,
        degraded_z: float = 4.0,
        fast_lambda: float = 0.9,
        slow_lambda: float = 0.995,
        quality_warmup: int = 100,
    ) -> None:
        if min_samples < 2 * n_states or window < min_samples:
            raise ValueError("need window >= min_samples >= 2 * n_states")
        self.horizon = horizon
        self.n_states = n_states
        self.min_samples = min_samples
        self.refit_every = refit_every
        self.n_iter = n_iter
        self.labels = labels or DEFAULT_LABELS.get(n_states) or tuple(
            f"REGIME_{k}" for k in range(n_states)
        )
        if len(self.labels) != n_states:
            raise ValueError("one label per state is required")
        self.buffer: deque[np.ndarray] = deque(maxlen=window)
        self.params: hmm.HMMParams | None = None
        self.probs: np.ndarray | None = None
        self.n_obs = 0
        self.fit_count = 0
        self.switches = 0          # changements de régime dominant (stabilité)
        self.loglik: float | None = None
        self._since_fit = 0
        self._last_label: str | None = None
        self.degraded_z = degraded_z
        self.fast_lambda = fast_lambda
        self.slow_lambda = slow_lambda
        self.quality_warmup = quality_warmup
        self._ll_fast: float | None = None
        self._ll_slow: float | None = None
        self._ll_dev_var = 0.0
        self._ll_n = 0
        self.degraded = False
        self.degraded_count = 0

    def _fit(self) -> None:
        X = np.array(self.buffer)
        params, self.loglik = hmm.fit(
            X, self.n_states, init=self.params, n_iter=self.n_iter, sort_dim=self.VOL_DIM
        )
        order = np.argsort(params.means[:, self.VOL_DIM], kind="stable")
        self.params = params.permute(order)
        self.probs = hmm.filter_probs(X, self.params)
        self.fit_count += 1
        self._since_fit = 0

    def update(self, x: np.ndarray, timestamp: datetime) -> RegimeState | None:
        x = np.asarray(x, dtype=float)
        if not np.all(np.isfinite(x)):
            return self.state(timestamp) if self.params is not None else None
        self.buffer.append(x)
        self.n_obs += 1

        if self.params is None:
            if len(self.buffer) < self.min_samples:
                return None
            self._fit()
        else:
            self.probs, ll = hmm.filter_step_ll(self.probs, x, self.params)
            self._update_fit_quality(ll)
            self._since_fit += 1
            if self._since_fit >= self.refit_every:
                self._fit()

        state = self.state(timestamp)
        if self._last_label is not None and state.most_likely != self._last_label:
            self.switches += 1
        self._last_label = state.most_likely
        return state

    def _update_fit_quality(self, ll: float) -> None:
        if self._ll_slow is None:
            self._ll_fast = self._ll_slow = ll if np.isfinite(ll) else 0.0
        sd = self._ll_dev_var ** 0.5
        if not np.isfinite(ll):
            ll = self._ll_slow - 10 * (sd or 1.0)
        elif sd > 0:
            ll = max(ll, self._ll_fast - 20 * sd)     # borne l'effet d'un point aberrant
        self._ll_fast = self.fast_lambda * self._ll_fast + (1 - self.fast_lambda) * ll
        self._ll_slow = self.slow_lambda * self._ll_slow + (1 - self.slow_lambda) * ll
        dev = self._ll_fast - self._ll_slow
        # Variance de l'écart : moyenne simple pendant le warm-up, puis EWMA
        # lente. L'écart est écrêté à ±degraded_z σ pour que la dégradation
        # qu'on cherche à détecter ne gonfle pas elle-même la variance.
        self._ll_n += 1
        if self._ll_n > self.quality_warmup and sd > 0:
            dev = max(-self.degraded_z * sd, min(dev, self.degraded_z * sd))
        weight = max(1.0 / self._ll_n, 1 - self.slow_lambda)
        self._ll_dev_var = (1 - weight) * self._ll_dev_var + weight * dev * dev

        was = self.degraded
        z = self.fit_zscore
        self.degraded = z is not None and z < -self.degraded_z
        if self.degraded and not was:
            self.degraded_count += 1

    @property
    def fit_zscore(self) -> float | None:
        """Écart (en écarts-types) entre la vraisemblance récente et sa moyenne longue.
        Négatif = les données collent moins bien au modèle que d'habitude."""
        if self._ll_n < self.quality_warmup or self._ll_dev_var <= 0:
            return None
        return (self._ll_fast - self._ll_slow) / self._ll_dev_var ** 0.5

    def state(self, timestamp: datetime) -> RegimeState:
        return RegimeState(
            horizon=self.horizon,
            probabilities={label: float(p) for label, p in zip(self.labels, self.probs)},
            timestamp=timestamp,
            n_obs=self.n_obs,
            degraded=self.degraded,
            fit_zscore=self.fit_zscore,
        )
