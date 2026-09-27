"""Alertes (README §35) : signaux d'attention pour l'humain, publiés une fois
quand une condition apparaît (pas à chaque évaluation tant qu'elle dure).

    POSITION_DRIFT        poids trop loin de la cible
    RISK_LIMIT            dépassement d'une limite de risque
    DRAWDOWN              drawdown du portefeuille au-delà de la limite
    VOLATILITY_SPIKE      volatilité court terme >> volatilité plus longue
    CORRELATION_SPIKE     corrélation moyenne en forte hausse
    REGIME_CHANGE         changement de régime dominant (HMM)
    SIGNAL_REVERSAL       le signal change de signe avec conviction
    SIGNAL_DISAGREEMENT   les modèles de l'ensemble divergent
    SAFETY_STATE          changement d'état du Safety Engine
    REBALANCE_PROPOSED    le Decision Engine propose un UREBALANCE
    NEW_EARNINGS          nouveaux résultats publiés (avec la surprise)
    GUIDANCE_CHANGE       nouvelle guidance, variation >= seuil

(IMPORTANT_NEWS viendra avec la phase IA : l'importance d'une news demande
une lecture structurée.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping

import numpy as np

from trading_engine.models.regime import RegimeState
from trading_engine.signals.signal import Signal
from trading_engine.timeutils import parse_timeframe


@dataclass(frozen=True)
class Alert:
    kind: str
    symbol: str | None
    timestamp: datetime | None
    message: str
    severity: str = "info"

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.symbol or 'PORTFOLIO'}"


@dataclass(frozen=True)
class AlertConfig:
    vol_fast: str = "5m"
    vol_slow: str = "1h"
    vol_spike_ratio: float = 2.0
    correlation_timeframe: str = "1h"
    correlation_jump: float = 0.3
    reversal_t: float = 0.5            # |t-stat| minimal du nouveau signal
    # Les régimes HF basculent souvent : visibles dans le suivi, pas en alerte.
    regime_horizons: tuple[str, ...] = ("MT", "LT")
    regime_min_confidence: float = 0.8
    guidance_change: float = 0.02      # variation relative minimale signalée
    disagreement_share: float = 0.5    # désaccord / incertitude totale


class AlertEngine:
    def __init__(self, config: AlertConfig | None = None) -> None:
        self.config = config or AlertConfig()
        self._active: set[str] = set()
        self._last_regime: dict[tuple[str, str], str] = {}
        self._last_sign: dict[str, int] = {}
        self._last_safety: str | None = None
        self._corr_baseline: float | None = None
        self._last_published: dict[tuple[str, str], datetime] = {}
        self._scale = math.sqrt(parse_timeframe(self.config.vol_slow) / parse_timeframe(self.config.vol_fast))

    def _state_alerts(self, conditions: list[Alert]) -> list[Alert]:
        """Conditions persistantes : alerte seulement quand elles apparaissent."""
        keys = {a.key for a in conditions}
        new = [a for a in conditions if a.key not in self._active]
        self._active = keys
        return new

    def evaluate(
        self,
        timestamp: datetime | None,
        *,
        breaches: list = (),
        volatilities: Mapping[str, tuple[float | None, float | None]] | None = None,
        correlation: np.ndarray | None = None,
        regimes: Mapping[str, Mapping[str, RegimeState]] | None = None,
        signals: Mapping[str, Signal | None] | None = None,
        safety_state: str | None = None,
        decision=None,
        earnings: Mapping[str, tuple[datetime, float | None]] | None = None,
        guidance: Mapping[str, tuple[datetime, float]] | None = None,
    ) -> list[Alert]:
        """`earnings` : symbole -> (publication, surprise) du dernier EPS publié ;
        `guidance` : symbole -> (publication, variation) de la dernière guidance."""
        cfg = self.config
        conditions: list[Alert] = []
        events: list[Alert] = []

        for b in breaches:
            kind = {"DRIFT": "POSITION_DRIFT", "DRAWDOWN": "DRAWDOWN"}.get(b.kind, "RISK_LIMIT")
            severity = "critical" if b.kind in ("DRAWDOWN", "LEVERAGE") else "warning"
            conditions.append(Alert(kind if kind != "RISK_LIMIT" else f"RISK_LIMIT_{b.kind}",
                                    b.symbol, timestamp, str(b), severity))

        for sym, (fast, slow) in sorted((volatilities or {}).items()):
            if fast and slow and fast * self._scale > cfg.vol_spike_ratio * slow:
                conditions.append(Alert("VOLATILITY_SPIKE", sym, timestamp,
                                        f"vol {cfg.vol_fast} ≈ {fast * self._scale / slow:.1f}× vol {cfg.vol_slow}",
                                        "warning"))

        if correlation is not None and correlation.shape[0] > 1:
            n = correlation.shape[0]
            mean_corr = float((correlation.sum() - n) / (n * (n - 1)))
            if self._corr_baseline is not None and mean_corr - self._corr_baseline > cfg.correlation_jump:
                conditions.append(Alert("CORRELATION_SPIKE", None, timestamp,
                                        f"corrélation moyenne {self._corr_baseline:+.2f} → {mean_corr:+.2f}",
                                        "warning"))
            self._corr_baseline = mean_corr if self._corr_baseline is None else (
                0.9 * self._corr_baseline + 0.1 * mean_corr)

        for sym, sig in sorted((signals or {}).items()):
            if sig is None:
                continue
            if sig.disagreement is not None and sig.disagreement > cfg.disagreement_share * sig.std:
                conditions.append(Alert("SIGNAL_DISAGREEMENT", sym, timestamp,
                                        f"désaccord {sig.disagreement * 1e4:.1f}bp / incertitude {sig.std * 1e4:.1f}bp"))
            sign = int(np.sign(sig.mean)) if abs(sig.t_stat) >= cfg.reversal_t else 0
            previous = self._last_sign.get(sym, 0)
            if sign and previous and sign != previous:
                events.append(Alert("SIGNAL_REVERSAL", sym, timestamp,
                                    f"signal {'haussier' if sign > 0 else 'baissier'} (t={sig.t_stat:+.2f})"))
            if sign:
                self._last_sign[sym] = sign

        for sym, by_horizon in sorted((regimes or {}).items()):
            for horizon, state in sorted(by_horizon.items()):
                if horizon not in cfg.regime_horizons:
                    continue
                if not state.degraded and state.confidence < cfg.regime_min_confidence:
                    continue      # état ambigu : on garde le dernier régime affirmé
                label = "DEGRADED" if state.degraded else state.most_likely
                previous = self._last_regime.get((sym, horizon))
                if previous is not None and previous != label:
                    events.append(Alert("REGIME_CHANGE", sym, timestamp,
                                        f"{horizon}: {previous} → {label} ({state.confidence:.0%})"))
                self._last_regime[(sym, horizon)] = label

        if safety_state is not None:
            if self._last_safety is not None and safety_state != self._last_safety:
                severity = "critical" if safety_state == "HALTED" else "warning"
                events.append(Alert("SAFETY_STATE", None, timestamp,
                                    f"{self._last_safety} → {safety_state}", severity))
            self._last_safety = safety_state

        for sym, (published, surprise) in sorted((earnings or {}).items()):
            key = ("earnings", sym)
            if self._last_published.get(key) != published:
                text = "résultats publiés" + (f", surprise {surprise:+.1%}" if surprise is not None else "")
                severity = "warning" if surprise is not None and abs(surprise) >= 0.1 else "info"
                events.append(Alert("NEW_EARNINGS", sym, timestamp, text, severity))
                self._last_published[key] = published

        for sym, (published, change) in sorted((guidance or {}).items()):
            key = ("guidance", sym)
            if self._last_published.get(key) != published:
                if abs(change) >= cfg.guidance_change:
                    events.append(Alert("GUIDANCE_CHANGE", sym, timestamp,
                                        f"guidance EPS {change:+.1%}", "warning"))
                self._last_published[key] = published

        if decision is not None and decision.action == "UREBALANCE":
            events.append(Alert("REBALANCE_PROPOSED", None, timestamp,
                                f"{decision.decision_id}: {decision.fraction:.0%} du chemin, "
                                f"net {decision.net_benefit:+.2f}"))

        return self._state_alerts(conditions) + events
