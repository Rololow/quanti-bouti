"""Orchestration des modèles online (README §8-9, §28, §49.1).

- HMM de régime par symbole et par horizon (HMM-HF / HMM-MT / HMM-LT), chacun
  mis à jour uniquement à la clôture de sa propre barre ;
- modèle de facteurs online émettant des `Signal` (rendement attendu ± std).

Chaque famille de modèles peut être désactivée dans la configuration
(ablation, README §49.2).
"""

from __future__ import annotations

import math

import numpy as np

from trading_engine.config import ModelsConfig
from trading_engine.data.events import BarEvent
from trading_engine.features.feature_engine import FeatureEngine
from trading_engine.models.online_factors import OnlineFactorModel
from trading_engine.models.regime import RegimeModel, RegimeState
from trading_engine.signals.signal import Signal


class ModelEngine:
    def __init__(self, config: ModelsConfig, features: FeatureEngine) -> None:
        self.config = config
        self.features = features
        # horizon (HF/MT/LT) -> timeframe de barre
        self.regime_timeframes = dict(config.regimes.timeframes) if config.regimes.enabled else {}
        self._regime_models: dict[tuple[str, str], RegimeModel] = {}
        self._regimes: dict[str, dict[str, RegimeState]] = {}
        self.factor_model = (
            OnlineFactorModel(
                config.factor.features,
                timeframe=config.factor.timeframe,
                horizon_bars=config.factor.horizon_bars,
                lam=config.factor.forgetting,
                ridge=config.factor.ridge,
                min_samples=config.factor.min_samples,
            )
            if config.factor.enabled
            else None
        )

    def _regime_model(self, symbol: str, horizon: str) -> RegimeModel:
        key = (symbol, horizon)
        if key not in self._regime_models:
            cfg = self.config.regimes
            self._regime_models[key] = RegimeModel(
                horizon, n_states=cfg.n_states, min_samples=cfg.min_samples,
                window=cfg.window, refit_every=cfg.refit_every,
            )
        return self._regime_models[key]

    def on_bar(self, bar: BarEvent) -> None:
        """À appeler après `FeatureEngine.on_bar` pour la même barre."""
        if bar.payload.get("correction"):
            return
        snapshot = None
        for horizon, tf in self.regime_timeframes.items():
            if tf != bar.timeframe:
                continue
            snapshot = snapshot or self.features.snapshot(bar.symbol)
            ret, vol = snapshot.get(f"ret_{tf}"), snapshot.get(f"vol_{tf}")
            if ret is None or not vol or vol <= 0:
                continue
            state = self._regime_model(bar.symbol, horizon).update(
                np.array([ret, math.log(vol)]), bar.end
            )
            if state is not None:
                self._regimes.setdefault(bar.symbol, {})[horizon] = state

        if self.factor_model is not None and bar.timeframe == self.factor_model.timeframe:
            snapshot = snapshot or self.features.snapshot(bar.symbol)
            self.factor_model.on_bar(bar.symbol, bar.close, snapshot, bar.end)

    def regimes(self, symbol: str) -> dict[str, RegimeState]:
        return dict(self._regimes.get(symbol, {}))

    def regime_model(self, symbol: str, horizon: str) -> RegimeModel | None:
        return self._regime_models.get((symbol, horizon))

    def signals(self, symbol: str) -> list[Signal]:
        if self.factor_model is None:
            return []
        signal = self.factor_model.signal(symbol)
        return [] if signal is None else [signal]
