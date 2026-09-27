"""Détection de régime online par HMM (README §9, §49.3).

Cycle de vie d'un `RegimeModel` (un par symbole et par horizon HF / MT / LT) :

1. warm-up : accumulation de `min_samples` observations (historical seed) ;
2. premier ajustement Baum-Welch, puis filtrage online à chaque barre ;
3. ré-ajustement tous les `refit_every` barres sur une fenêtre glissante de
   `window` observations (oubli), en partant des paramètres courants.

Les états sont triés par volatilité croissante après chaque ajustement : le
label d'un état garde donc le même sens d'un ajustement à l'autre. La sortie
est une distribution de probabilités, jamais un seul régime.
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
            self.probs = hmm.filter_step(self.probs, x, self.params)
            self._since_fit += 1
            if self._since_fit >= self.refit_every:
                self._fit()

        state = self.state(timestamp)
        if self._last_label is not None and state.most_likely != self._last_label:
            self.switches += 1
        self._last_label = state.most_likely
        return state

    def state(self, timestamp: datetime) -> RegimeState:
        return RegimeState(
            horizon=self.horizon,
            probabilities={label: float(p) for label, p in zip(self.labels, self.probs)},
            timestamp=timestamp,
            n_obs=self.n_obs,
        )
