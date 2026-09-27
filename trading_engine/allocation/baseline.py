"""Stratégie de référence : momentum simple + volatility targeting (README §49.2).

Tout module plus sophistiqué doit battre cette baseline en replay pour être
conservé. Construction (long-only) :

1. score_i  = max(momentum_i, 0)                      (momentum normalisé)
2. raw_i    = score_i / sigma_i                       (pondération par le risque)
3. w_i      = budget * raw_i / sum(raw)               (budget investi)
4. si sigma_p = sqrt(w' Sigma w) annualisée > cible : w *= cible / sigma_p
5. plafond par position

Chaque poids est accompagné de son attribution (score, volatilité, facteur
de scaling) pour pouvoir expliquer tout changement de cible.
"""

from __future__ import annotations

import math
import numpy as np

from trading_engine.allocation.types import TargetAllocation
from trading_engine.features.feature_engine import FeatureEngine
from trading_engine.timeutils import periods_per_year

class BaselineAllocator:
    def __init__(
        self,
        *,
        momentum_horizon: str = "1h",
        vol_timeframe: str = "1h",
        target_vol: float = 0.10,
        budget: float = 1.0,
        max_weight: float = 0.40,
    ) -> None:
        if not 0 < budget <= 1:
            raise ValueError(f"budget must be in (0, 1], got {budget}")
        self.momentum_feature = f"mom_{momentum_horizon}"
        self.vol_timeframe = vol_timeframe
        self.target_vol = target_vol
        self.budget = budget
        self.max_weight = max_weight
        self._annualize = math.sqrt(periods_per_year(vol_timeframe))

    def allocate(self, features: FeatureEngine, symbols: list[str]) -> TargetAllocation:
        raw: dict[str, float] = {}
        attribution: dict[str, dict[str, float]] = {}
        for sym in symbols:
            snap = features.snapshot(sym)
            mom = snap.get(self.momentum_feature)
            vol = snap.get(f"vol_{self.vol_timeframe}")
            if mom is None or not vol:
                continue  # warm-up : pas d'opinion -> poids nul
            score = max(mom, 0.0)
            raw[sym] = score / vol
            attribution[sym] = {"momentum": mom, "score": score, "vol": vol * self._annualize}

        total = sum(raw.values())
        weights = {sym: 0.0 for sym in symbols}
        if total <= 0:
            return TargetAllocation(weights, attribution)
        for sym, value in raw.items():
            weights[sym] = self.budget * value / total

        port_vol = self._portfolio_vol(features, weights)
        scale = 1.0
        if port_vol is not None and port_vol > self.target_vol:
            scale = self.target_vol / port_vol
        for sym in weights:
            weights[sym] = min(weights[sym] * scale, self.max_weight)
            if sym in attribution:
                attribution[sym]["unscaled_weight"] = self.budget * raw[sym] / total
                attribution[sym]["vol_scale"] = scale
                attribution[sym]["weight"] = weights[sym]
        final_vol = self._portfolio_vol(features, weights)
        return TargetAllocation(weights, attribution, final_vol, scale)

    def _portfolio_vol(self, features: FeatureEngine, weights: dict[str, float]) -> float | None:
        if self.vol_timeframe not in features.covariance_timeframes:
            return None
        if features.correlation_updates(self.vol_timeframe) < 2:
            return None
        symbols, cov = features.covariance(self.vol_timeframe)
        w = np.array([weights.get(sym, 0.0) for sym in symbols])
        variance = float(w @ cov @ w)
        return math.sqrt(max(variance, 0.0)) * self._annualize
