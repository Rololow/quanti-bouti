"""Allocation Engine (README §17-18, §49.12-49.13).

Pipeline : Signal -> Risk budget -> Risk allocation -> Volatility targeting.

Méthodes :

- "risk_parity" : contributions au risque égales ;
- "hrp"         : hierarchical risk parity ;
- "signal"      : budgets de risque proportionnels au rendement attendu
                  positif des signaux, pondéré par la crédibilité du modèle
                  (son skill hors échantillon, borné à [0, 1]). Un modèle qui
                  ne bat pas « prédire zéro » n'obtient aucun budget.

Toutes les décisions sont portfolio-level (covariance complète), et chaque
poids est décomposé en étapes dont la somme donne le poids final.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from trading_engine.allocation.hrp import hrp_weights
from trading_engine.allocation.risk_parity import risk_budget_weights
from trading_engine.allocation.targeting import volatility_target
from trading_engine.signals.signal import Signal

METHODS = ("risk_parity", "hrp", "signal")


class RiskAllocator:
    def __init__(
        self,
        method: str = "risk_parity",
        *,
        target_vol: float = 0.10,
        max_gross: float = 1.0,
        min_skill: float = 0.0,
    ) -> None:
        if method not in METHODS:
            raise ValueError(f"unknown allocation method {method!r}, expected one of {METHODS}")
        self.method = method
        self.target_vol = target_vol
        self.max_gross = max_gross
        self.min_skill = min_skill

    def signal_budgets(
        self, symbols: list[str], signals: Mapping[str, Signal | None], skill: float | None
    ) -> np.ndarray:
        credibility = 0.0 if skill is None or skill <= self.min_skill else min(skill, 1.0)
        budgets = np.zeros(len(symbols))
        for i, sym in enumerate(symbols):
            sig = signals.get(sym)
            if sig is not None and sig.mean > 0:
                budgets[i] = sig.mean * credibility
        return budgets

    def allocate(
        self,
        symbols: list[str],
        cov: np.ndarray,
        signals: Mapping[str, Signal | None] | None = None,
        skill: float | None = None,
    ) -> tuple[dict[str, float], dict[str, dict[str, float]], float]:
        """Retourne (poids, attribution par étapes, facteur de vol targeting)."""
        steps: dict[str, np.ndarray] = {}
        if self.method == "hrp":
            base = hrp_weights(cov)
        else:
            base = risk_budget_weights(cov)
        steps["risk_allocation"] = base
        w = base

        if self.method == "signal":
            budgets = self.signal_budgets(symbols, signals or {}, skill)
            tilted = risk_budget_weights(cov, budgets) if budgets.sum() > 0 else np.zeros(len(symbols))
            steps["signal_tilt"] = tilted - w
            w = tilted

        scaled, scale = volatility_target(w, cov, self.target_vol, max_gross=self.max_gross)
        steps["vol_target"] = scaled - w

        weights = {sym: float(scaled[i]) for i, sym in enumerate(symbols)}
        attribution = {
            sym: {name: float(delta[i]) for name, delta in steps.items()}
            for i, sym in enumerate(symbols)
        }
        return weights, attribution, scale
