"""Feature Engine multi-horizon (README §6, §7, §10-13).

Chaque feature est mise à jour à son propre rythme :

- tick  : volatilité EWMA tick, VWAP de séance ;
- barre : rendements, volatilité par barre, momentum, mean reversion,
          covariance / corrélation entre actifs.

Aucun modèle lent n'est recalculé à chaque tick.

Noms des features (dictionnaire plat, prêt pour les modèles online) :

    ret_<tf>          dernier log-rendement de la barre <tf>
    vol_<tf>          volatilité EWMA par barre <tf> (vol_tick pour les trades)
    mom_<horizon>     momentum normalisé par la volatilité
    z_<tf>            z-score du prix sur `zscore_window` barres
    ma_dist_<tf>      distance à la moyenne mobile
    reversal_<tf>     opposé du dernier rendement
    vwap_dist         distance au VWAP de séance
    fund_* / qual_score   fondamentaux point-in-time (si branchés)
    news_*            activité news dédupliquée (si branchée)
    ai_*              score événementiel issu de l'extraction IA validée

Les fondamentaux et les news sont lus « as of » l'horloge de marché
(`now` = dernier timestamp de trade ou de fin de barre) : aucune information
n'est visible avant sa publication.
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable

import numpy as np

from trading_engine.data.events import BarEvent, TradeEvent
from trading_engine.features.correlations import BarReturnAligner, EWMACovariance
from trading_engine.features.mean_reversion import SessionVWAP, ma_distance, reversal, zscore
from trading_engine.features.momentum import momentum, resolve_horizon
from trading_engine.features.returns import PriceHistory
from trading_engine.features.volatility import VolatilityBook

TICK = "tick"

DEFAULT_MOMENTUM_HORIZONS = ("5m", "30m", "1h", "1d", "20d", "60d", "252d")


class FeatureEngine:
    def __init__(
        self,
        timeframes: Iterable[str],
        *,
        momentum_horizons: Iterable[str] = DEFAULT_MOMENTUM_HORIZONS,
        mean_reversion_timeframes: Iterable[str] = ("5m",),
        zscore_window: int = 20,
        correlation_timeframes: Iterable[str] | None = None,
        lam: float = 0.94,
        fundamentals=None,
        news=None,
        ai=None,
    ) -> None:
        self.ai = ai                          # NewsIntelligence | None
        self.fundamentals = fundamentals      # FundamentalFeatures | None
        self.news = news                      # NewsEngine | None
        self.now: datetime | None = None      # horloge de marché
        self.timeframes = tuple(timeframes)
        self.zscore_window = zscore_window
        self.mean_reversion_timeframes = tuple(
            tf for tf in mean_reversion_timeframes if tf in self.timeframes
        )
        self.volatility = VolatilityBook(lam)

        # horizon -> (timeframe, nombre de barres) ; horizons sans barre adaptée ignorés.
        self.momentum_specs: dict[str, tuple[str, int]] = {}
        for horizon in momentum_horizons:
            try:
                self.momentum_specs[horizon] = resolve_horizon(horizon, self.timeframes)
            except ValueError:
                continue

        # Taille d'historique nécessaire par timeframe.
        self._maxlen: dict[str, int] = {tf: 2 for tf in self.timeframes}
        for tf, n in self.momentum_specs.values():
            self._maxlen[tf] = max(self._maxlen[tf], n + 1)
        for tf in self.mean_reversion_timeframes:
            self._maxlen[tf] = max(self._maxlen[tf], zscore_window)

        self._history: dict[tuple[str, str], PriceHistory] = {}
        self._vwap: dict[str, SessionVWAP] = {}
        self._last_price: dict[str, float] = {}
        self._updated_at: dict[str, datetime] = {}

        corr_tfs = self.timeframes if correlation_timeframes is None else tuple(correlation_timeframes)
        self._aligners = {tf: BarReturnAligner() for tf in corr_tfs}
        self._covariances = {tf: EWMACovariance(lam) for tf in corr_tfs}

    # ---------------------------------------------------------------- updates

    def _tick_clock(self, t: datetime) -> None:
        if self.now is None or t > self.now:
            self.now = t

    def on_trade(self, trade: TradeEvent) -> None:
        self._tick_clock(trade.timestamp)
        sym = trade.symbol
        self._last_price[sym] = trade.price
        self._updated_at[sym] = trade.timestamp
        self.volatility.update(sym, TICK, trade.price)
        self._vwap.setdefault(sym, SessionVWAP()).update(trade.timestamp, trade.price, trade.size)

    def on_bar(self, bar: BarEvent) -> None:
        # Une barre corrigée (trades tardifs) remplace la précédente : ne pas
        # la compter une seconde fois.
        if bar.payload.get("correction"):
            return
        self._tick_clock(bar.end)
        tf = bar.timeframe
        key = (bar.symbol, tf)
        if key not in self._history:
            self._history[key] = PriceHistory(self._maxlen.get(tf, 2))
        self._history[key].update(bar.close)
        self.volatility.update(bar.symbol, tf, bar.close)
        self._updated_at[bar.symbol] = max(bar.end, self._updated_at.get(bar.symbol, bar.end))

        aligner = self._aligners.get(tf)
        if aligner is not None:
            returns = aligner.on_bar(bar)
            if returns:
                self._covariances[tf].update(returns)

    # ---------------------------------------------------------------- lecture

    def history(self, symbol: str, timeframe: str) -> PriceHistory | None:
        return self._history.get((symbol, timeframe))

    def symbols(self) -> list[str]:
        return sorted(set(self._last_price) | {sym for sym, _ in self._history})

    def snapshot(self, symbol: str) -> dict[str, float]:
        """Toutes les features disponibles pour `symbol` (les features en warm-up sont absentes)."""
        out: dict[str, float] = {}

        tick_vol = self.volatility.get(symbol, TICK)
        if tick_vol is not None:
            out["vol_tick"] = tick_vol
        vwap = self._vwap.get(symbol)
        price = self._last_price.get(symbol)
        if vwap is not None and price is not None:
            dist = vwap.distance(price)
            if dist is not None:
                out["vwap_dist"] = dist

        for tf in self.timeframes:
            hist = self._history.get((symbol, tf))
            if hist is None:
                continue
            ret = hist.log_return(1)
            if ret is not None:
                out[f"ret_{tf}"] = ret
            vol = self.volatility.get(symbol, tf)
            if vol is not None:
                out[f"vol_{tf}"] = vol

        for horizon, (tf, n) in self.momentum_specs.items():
            hist = self._history.get((symbol, tf))
            if hist is None:
                continue
            value = momentum(hist, self.volatility.get(symbol, tf), n)
            if value is not None:
                out[f"mom_{horizon}"] = value

        for tf in self.mean_reversion_timeframes:
            hist = self._history.get((symbol, tf))
            if hist is None:
                continue
            for name, value in (
                (f"z_{tf}", zscore(hist, self.zscore_window)),
                (f"ma_dist_{tf}", ma_distance(hist, self.zscore_window)),
                (f"reversal_{tf}", reversal(hist)),
            ):
                if value is not None:
                    out[name] = value

        if self.fundamentals is not None:
            out.update(self.fundamentals.snapshot(symbol, self.now))
        if self.news is not None:
            out.update(self.news.snapshot(symbol, self.now))
        if self.ai is not None:
            out.update(self.ai.snapshot(symbol, self.now))
        return out

    def correlation(self, timeframe: str) -> tuple[list[str], np.ndarray]:
        return self._covariances[timeframe].correlation()

    def covariance(self, timeframe: str) -> tuple[list[str], np.ndarray]:
        return self._covariances[timeframe].covariance()

    @property
    def covariance_timeframes(self) -> tuple[str, ...]:
        return tuple(self._covariances)

    def correlation_updates(self, timeframe: str) -> int:
        return self._covariances[timeframe].n_updates
