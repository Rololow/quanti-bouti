"""Orchestration des modèles online (README §8-9, §28, §49.1, §50.3-50.5).

- HMM de régime par symbole et par horizon (HMM-HF / HMM-MT / HMM-LT), chacun
  mis à jour uniquement à la clôture de sa propre barre. Le HMM ne vote pas :
  il fournit le **contexte** (régime) des autres modèles ;
- un ou plusieurs prédicteurs online (modèles de facteurs sur des jeux de
  features différents), chacun avec sa fiabilité par contexte ;
- un ensemble qui combine les prédicteurs selon la corrélation de leurs
  erreurs et élargit l'incertitude en cas de désaccord.

Chaque famille de modèles peut être désactivée dans la configuration
(ablation, README §49.2).
"""

from __future__ import annotations

import math

import numpy as np

from trading_engine.config import ModelsConfig
from trading_engine.data.events import BarEvent
from trading_engine.features.feature_engine import FeatureEngine
from trading_engine.models.ensemble import ModelEnsemble
from trading_engine.models.online_factors import OnlineFactorModel
from trading_engine.models.regime import RegimeModel, RegimeState
from trading_engine.safety.safety_engine import ModelHealth
from trading_engine.signals.signal import Signal

UNKNOWN_CONTEXT = "UNKNOWN"
DEGRADED_CONTEXT = "DEGRADED"


class ModelEngine:
    def __init__(self, config: ModelsConfig, features: FeatureEngine) -> None:
        self.config = config
        self.features = features
        # horizon (HF/MT/LT) -> timeframe de barre
        self.regime_timeframes = dict(config.regimes.timeframes) if config.regimes.enabled else {}
        self._regime_models: dict[tuple[str, str], RegimeModel] = {}
        self._regimes: dict[str, dict[str, RegimeState]] = {}

        self.predictors: dict[str, OnlineFactorModel] = {
            p.name: OnlineFactorModel(
                p.features, timeframe=p.timeframe, horizon_bars=p.horizon_bars, name=p.name,
                lam=p.forgetting, ridge=p.ridge, min_samples=p.min_samples,
            )
            for p in config.predictors if p.enabled
        }
        frames = {(m.timeframe, m.horizon_bars) for m in self.predictors.values()}
        if len(frames) > 1:
            raise ValueError("all predictors must share the same timeframe and horizon_bars")
        self.predictor_timeframe = next(iter(frames))[0] if frames else None
        self.ensemble = (
            ModelEnsemble(list(self.predictors), lam=config.ensemble.lam,
                          shrinkage=config.ensemble.shrinkage, min_samples=config.ensemble.min_samples)
            if self.predictors else None
        )
        self._signals: dict[str, Signal] = {}
        self._seen_drifts: dict[str, int] = {}
        self._seen_degraded: dict[tuple[str, str], int] = {}

    # ------------------------------------------------------------------ régimes

    def _regime_model(self, symbol: str, horizon: str) -> RegimeModel:
        key = (symbol, horizon)
        if key not in self._regime_models:
            cfg = self.config.regimes
            self._regime_models[key] = RegimeModel(
                horizon, n_states=cfg.n_states, min_samples=cfg.min_samples,
                window=cfg.window, refit_every=cfg.refit_every, degraded_z=cfg.degraded_z,
            )
        return self._regime_models[key]

    def context(self, symbol: str) -> str:
        """Contexte de marché d'un symbole pour la fiabilité des prédicteurs :
        régime le plus probable à l'horizon du prédicteur (ou le plus court)."""
        regimes = self._regimes.get(symbol, {})
        horizons = [h for h, tf in self.regime_timeframes.items() if tf == self.predictor_timeframe]
        horizons = horizons or list(self.regime_timeframes)
        for horizon in horizons:
            state = regimes.get(horizon)
            if state is not None:
                return DEGRADED_CONTEXT if state.degraded else state.most_likely
        return UNKNOWN_CONTEXT

    # ------------------------------------------------------------------ mises à jour

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

        if self.predictors and bar.timeframe == self.predictor_timeframe:
            snapshot = snapshot or self.features.snapshot(bar.symbol)
            context = self.context(bar.symbol)
            signals = {
                name: model.on_bar(bar.symbol, bar.close, snapshot, bar.end, context)
                for name, model in self.predictors.items()
            }
            outcomes = {name: m.last_outcome.get(bar.symbol) for name, m in self.predictors.items()}
            if all(o is not None for o in outcomes.values()):
                self.ensemble.record({name: o.error for name, o in outcomes.items()})
            combined = self.ensemble.combine(signals)
            if combined is None:
                self._signals.pop(bar.symbol, None)
            else:
                self._signals[bar.symbol] = combined

    # ------------------------------------------------------------------ lecture

    @property
    def factor_model(self) -> OnlineFactorModel | None:
        """Premier prédicteur (compatibilité)."""
        return next(iter(self.predictors.values()), None)

    def health(self) -> ModelHealth:
        """Dérives et dégradations détectées depuis le dernier appel, valeurs non finies."""
        health = ModelHealth()
        for name, model in self.predictors.items():
            count = model.monitor.drift_count
            health.drift_events += count - self._seen_drifts.get(name, 0)
            self._seen_drifts[name] = count
            reg = model.regression
            if not (np.all(np.isfinite(reg.coef)) and np.all(np.isfinite(reg.P))):
                health.non_finite = True
                health.details.append(f"predictor {name}")
        for (sym, horizon), model in sorted(self._regime_models.items()):
            if model.probs is not None and not np.all(np.isfinite(model.probs)):
                health.non_finite = True
                health.details.append(f"regime {sym} {horizon}")
            if model.degraded:
                health.degraded.append((sym, f"regime {horizon} z={model.fit_zscore:.1f}"))
        return health

    def regimes(self, symbol: str) -> dict[str, RegimeState]:
        return dict(self._regimes.get(symbol, {}))

    def regime_model(self, symbol: str, horizon: str) -> RegimeModel | None:
        return self._regime_models.get((symbol, horizon))

    def signals(self, symbol: str) -> list[Signal]:
        signal = self._signals.get(symbol)
        return [] if signal is None else [signal]

    def predictor_signals(self, symbol: str) -> dict[str, Signal | None]:
        return {name: m.signal(symbol) for name, m in self.predictors.items()}
