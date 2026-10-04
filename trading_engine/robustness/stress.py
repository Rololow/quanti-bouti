"""Stress tests de la cible (README §50.6).

Au moment d'un rebalancement, la cible est recalculée sous des perturbations
**plausibles des estimations** (pas des krachs) :

    vol_up:<SYM>     volatilité d'un actif ×1.5 (le pire des actifs est retenu)
    corr_up          corrélations rapprochées de 1 de moitié
    corr_down        corrélations divisées par deux
    signal_down      rendements attendus -1 σ de leur incertitude
    signal_up        rendements attendus +1 σ

On compare la **forme** de l'allocation (poids normalisés) : un simple
changement de levier par le volatility targeting n'est pas une instabilité.

    instabilité_s = sum |w_s / sum(w_s) - w / sum(w)|      (dans [0, 2])
    RobustnessScore = 1 - max_s instabilité_s / 2          (dans [0, 1])

Une cible qui change beaucoup sous des hypothèses proches dépend trop de
bruits d'estimation : on la traite comme incertaine (mouvement réduit).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Mapping

import numpy as np

from trading_engine.signals.signal import Signal

Allocate = Callable[[np.ndarray, Mapping[str, Signal | None]], Mapping[str, float]]


@dataclass(frozen=True)
class StressReport:
    score: float
    instability: dict[str, float]      # scénario -> instabilité de forme
    worst: str | None


def _shape(weights: Mapping[str, float], symbols: list[str]) -> np.ndarray:
    w = np.array([max(weights.get(s, 0.0), 0.0) for s in symbols])
    total = w.sum()
    return w / total if total > 0 else w


def _scale_corr(cov: np.ndarray, towards: float, fraction: float) -> np.ndarray:
    std = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    outer = np.outer(std, std)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(outer > 0, cov / outer, 0.0)
    new = corr + fraction * (towards - corr)
    np.fill_diagonal(new, 1.0)
    return new * outer


def _shift_signals(signals: Mapping[str, Signal | None], k: float) -> dict[str, Signal | None]:
    return {s: None if sig is None else replace(sig, mean=sig.mean + k * sig.std) for s, sig in signals.items()}


def stress_test(
    allocate: Allocate,
    symbols: list[str],
    cov: np.ndarray,
    signals: Mapping[str, Signal | None],
    *,
    vol_bump: float = 1.5,
    include_signals: bool = True,
) -> StressReport:
    base = _shape(allocate(cov, signals), symbols)
    if base.sum() <= 0:
        return StressReport(1.0, {}, None)   # rien d'investi : rien à déstabiliser

    scenarios: dict[str, tuple[np.ndarray, Mapping[str, Signal | None]]] = {
        "corr_up": (_scale_corr(cov, 1.0, 0.5), signals),
        "corr_down": (_scale_corr(cov, 0.0, 0.5), signals),
    }
    for i, sym in enumerate(symbols):
        if cov[i, i] <= 0:
            continue
        bumped = cov.copy()
        bumped[i, :] *= vol_bump
        bumped[:, i] *= vol_bump
        scenarios[f"vol_up:{sym}"] = (bumped, signals)
    if include_signals and any(sig is not None for sig in signals.values()):
        scenarios["signal_down"] = (cov, _shift_signals(signals, -1.0))
        scenarios["signal_up"] = (cov, _shift_signals(signals, +1.0))

    instability = {}
    for name, (c, sigs) in scenarios.items():
        shape = _shape(allocate(c, sigs), symbols)
        instability[name] = float(np.abs(shape - base).sum()) if shape.sum() > 0 else 2.0

    worst = max(instability, key=instability.get) if instability else None
    score = 1.0 - (instability[worst] / 2.0 if worst else 0.0)
    return StressReport(max(0.0, min(1.0, score)), instability, worst)
