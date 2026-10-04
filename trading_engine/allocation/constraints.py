"""Constraint Engine (README §49.15) : rend la cible réalisable.

Placé entre l'allocation et la décision. Les contraintes sont appliquées
dans cet ordre ; l'excédent retiré va en cash (jamais redistribué, ce qui
ne peut qu'augmenter le risque) :

1. bornes par position          min_weight <= w_i <= max_weight
2. limites par secteur          sum_{i in s} w_i <= sector_limit_s
3. exposition brute / cash      sum |w_i| <= min(max_gross, 1 - min_cash)
4. volatilité du portefeuille   sqrt(w' Sigma w) <= max_portfolio_vol
5. turnover par rebalancement   sum |w_i - current_i| <= max_turnover
                                (on n'avance qu'une fraction du chemin)

Toutes ces contraintes définissent des ensembles convexes : si la position
courante et la cible sont réalisables, le pas partiel de l'étape 5 l'est aussi.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np

from trading_engine.risk.portfolio_risk import portfolio_vol


@dataclass(frozen=True)
class Constraints:
    max_weight: float = 0.40
    min_weight: float = 0.0              # 0 = long-only
    max_gross: float = 1.0
    min_cash: float = 0.0
    max_portfolio_vol: float | None = None
    max_turnover: float | None = None
    sectors: Mapping[str, str] = field(default_factory=dict)       # symbole -> secteur
    sector_limits: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ConstrainedTarget:
    weights: dict[str, float]
    adjustments: dict[str, float]    # poids final - poids demandé
    binding: tuple[str, ...]


class ConstraintEngine:
    def __init__(self, constraints: Constraints | None = None) -> None:
        self.constraints = constraints or Constraints()

    def apply(
        self,
        targets: Mapping[str, float],
        current: Mapping[str, float] | None = None,
        cov: np.ndarray | None = None,
        cov_symbols: list[str] | None = None,
    ) -> ConstrainedTarget:
        c = self.constraints
        current = dict(current or {})
        symbols = sorted(set(targets) | set(current))
        requested = np.array([targets.get(s, 0.0) for s in symbols])
        w = requested.copy()
        binding: list[str] = []

        # 1. bornes par position
        clipped = np.clip(w, c.min_weight, c.max_weight)
        for sym, before, after in zip(symbols, w, clipped):
            if not np.isclose(before, after):
                binding.append(f"{'MAX' if before > after else 'MIN'}_WEIGHT:{sym}")
        w = clipped

        # 2. secteurs
        for sector, limit in sorted(c.sector_limits.items()):
            members = [i for i, s in enumerate(symbols) if c.sectors.get(s) == sector]
            exposure = float(np.abs(w[members]).sum()) if members else 0.0
            if exposure > limit > 0:
                w[members] *= limit / exposure
                binding.append(f"SECTOR:{sector}")

        # 3. exposition brute / cash minimum
        gross_cap = gross_cap_of(c)
        gross = float(np.abs(w).sum())
        if gross > gross_cap > 0:
            w *= gross_cap / gross
            binding.append("GROSS_EXPOSURE" if c.max_gross <= 1.0 - c.min_cash else "MIN_CASH")

        # 4. volatilité du portefeuille
        if c.max_portfolio_vol is not None and cov is not None:
            index = {s: i for i, s in enumerate(cov_symbols or [])}
            idx = [index.get(s) for s in symbols]
            known = [k for k, i in enumerate(idx) if i is not None]
            if known:
                sub = cov[np.ix_([idx[k] for k in known], [idx[k] for k in known])]
                vol = portfolio_vol(w[known], sub)
                if vol > c.max_portfolio_vol:
                    w *= c.max_portfolio_vol / vol
                    binding.append("PORTFOLIO_VOL")

        # 5. turnover
        if c.max_turnover is not None:
            cur = np.array([current.get(s, 0.0) for s in symbols])
            turnover = float(np.abs(w - cur).sum())
            if turnover > c.max_turnover:
                w = cur + (w - cur) * (c.max_turnover / turnover)
                binding.append("TURNOVER")

        weights = {s: float(v) for s, v in zip(symbols, w)}
        adjustments = {s: float(v - r) for s, v, r in zip(symbols, w, requested)}
        return ConstrainedTarget(weights, adjustments, tuple(binding))


def gross_cap_of(c: "Constraints") -> float:
    """Exposition brute maximale. Sans levier (max_gross <= 1), le cash minimum
    s'en déduit ; avec levier, max_gross fait foi (le cash peut être négatif :
    emprunt / marge, financé)."""
    return c.max_gross if c.max_gross > 1.0 else min(c.max_gross, 1.0 - c.min_cash)


def fit_gross_after_freeze(
    weights: Mapping[str, float],
    current: Mapping[str, float],
    frozen: Iterable[str],
    gross_cap: float,
) -> dict[str, float]:
    """Respecte `sum |w| <= gross_cap` quand des symboles sont gelés à leur
    poids courant (Safety DEGRADED) : les autres ne font qu'une partie de leur
    chemin vers la cible. Si le portefeuille courant dépasse déjà la limite,
    les symboles non gelés sont réduits proportionnellement.
    """
    frozen = set(frozen)
    out = dict(weights)
    if sum(abs(v) for v in out.values()) <= gross_cap + 1e-12:
        return out
    movable = [s for s in out if s not in frozen]

    def blended(k: float) -> dict[str, float]:
        return {s: (current.get(s, 0.0) + k * (out[s] - current.get(s, 0.0)) if s in movable else out[s])
                for s in out}

    def gross(w: Mapping[str, float]) -> float:
        return sum(abs(v) for v in w.values())

    if gross(blended(0.0)) <= gross_cap:
        lo, hi = 0.0, 1.0                     # plus grande fraction du chemin qui respecte la limite
        for _ in range(50):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if gross(blended(mid)) <= gross_cap else (lo, mid)
        return blended(lo)
    base = blended(0.0)
    fixed = sum(abs(base[s]) for s in base if s not in movable)
    free = sum(abs(base[s]) for s in movable)
    scale = max(0.0, gross_cap - fixed) / free if free > 0 else 0.0
    return {s: base[s] * scale if s in movable else base[s] for s in base}
