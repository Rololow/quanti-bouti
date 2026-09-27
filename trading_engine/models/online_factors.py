"""Modèle de facteurs online (README §28 : E[r_{t+h} | X_t]).

À chaque clôture de barre `timeframe` d'un symbole :

1. les features X_t sont standardisées (moyenne / variance EWMA) ;
2. l'échantillon (X_t, p_t) est mis en attente ;
3. `horizon_bars` barres plus tard, le rendement réalisé
   r = log(p_{t+h} / p_t) est connu : la régression est mise à jour.
   Aucune information future n'est donc utilisée (anti-look-ahead) ;
4. la prévision courante est émise sous forme de `Signal` (moyenne ± std).

Le modèle est commun à tous les symboles (plus de données, moins de bruit).
Un `ModelMonitor` mesure le skill hors échantillon ; une dérive détectée
augmente l'incertitude des coefficients pour ré-apprendre plus vite.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Mapping

import numpy as np

from trading_engine.models.monitoring import ModelMonitor
from trading_engine.models.online_regression import OnlineLinearRegression
from trading_engine.signals.signal import Signal
from trading_engine.timeutils import format_timeframe, parse_timeframe


class EWMAStandardizer:
    """Standardisation online des features, bornée à ±clip."""

    def __init__(self, n: int, lam: float = 0.99, clip: float = 5.0) -> None:
        self.lam = lam
        self.clip = clip
        self.mean = np.zeros(n)
        self.var = np.ones(n)
        self.n_updates = 0

    def update(self, x: np.ndarray) -> np.ndarray:
        if self.n_updates == 0:
            self.mean = x.copy()
        else:
            diff = x - self.mean
            self.mean = self.mean + (1 - self.lam) * diff
            self.var = self.lam * self.var + (1 - self.lam) * diff * diff
        self.n_updates += 1
        return self.transform(x)

    def transform(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mean) / np.sqrt(np.maximum(self.var, 1e-18))
        return np.clip(z, -self.clip, self.clip)


@dataclass
class _Pending:
    x: np.ndarray
    price: float
    prediction: tuple[float, float] | None


class OnlineFactorModel:
    def __init__(
        self,
        feature_names: Iterable[str],
        *,
        timeframe: str,
        horizon_bars: int,
        lam: float = 0.995,
        ridge: float = 10.0,
        min_samples: int = 50,
        drift_inflation: float = 10.0,
    ) -> None:
        if horizon_bars < 1:
            raise ValueError(f"horizon_bars must be >= 1, got {horizon_bars}")
        self.feature_names = tuple(feature_names)
        if not self.feature_names:
            raise ValueError("at least one feature is required")
        self.timeframe = timeframe
        self.horizon_bars = horizon_bars
        self.horizon = format_timeframe(parse_timeframe(timeframe) * horizon_bars)
        self.source = f"factor_{timeframe}"
        self.drift_inflation = drift_inflation
        self.regression = OnlineLinearRegression(
            len(self.feature_names), lam=lam, ridge=ridge, min_samples=min_samples
        )
        self.standardizer = EWMAStandardizer(len(self.feature_names))
        self.monitor = ModelMonitor()
        self._pending: dict[str, deque[_Pending]] = {}
        self._signals: dict[str, Signal] = {}

    def on_bar(
        self, symbol: str, close: float, features: Mapping[str, float], timestamp: datetime
    ) -> Signal | None:
        queue = self._pending.setdefault(symbol, deque())

        # 1. Résultats désormais connus : apprentissage.
        # Chaque barre ajoute une entrée à la file (None si features en warm-up),
        # donc la tête de file a exactement `horizon_bars` barres d'âge.
        queue.append(None)
        if len(queue) > self.horizon_bars:
            matured = queue.popleft()
            if matured is not None:
                realized = math.log(close / matured.price)
                if matured.prediction is not None:
                    mean, std = matured.prediction
                    if self.monitor.update(realized, mean, std):
                        self.regression.inflate_uncertainty(self.drift_inflation)
                self.regression.update(matured.x, realized)

        # 2. Nouvelle observation et prévision.
        values = [features.get(name) for name in self.feature_names]
        if any(v is None or not math.isfinite(v) for v in values):
            self._signals.pop(symbol, None)
            return None
        x = self.standardizer.update(np.array(values, dtype=float))
        prediction = self.regression.predict(x)
        queue[-1] = _Pending(x, close, prediction)

        if prediction is None:
            self._signals.pop(symbol, None)
            return None
        mean, std = prediction
        signal = Signal(
            symbol=symbol, horizon=self.horizon, mean=mean, std=std,
            n_obs=self.regression.n_updates, timestamp=timestamp, source=self.source,
        )
        self._signals[symbol] = signal
        return signal

    def signal(self, symbol: str) -> Signal | None:
        return self._signals.get(symbol)

    def coefficients(self) -> dict[str, float]:
        names = ("intercept",) + self.feature_names
        return dict(zip(names, self.regression.coef.tolist()))
