"""Types communs de l'allocation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TargetAllocation:
    """Poids cibles et leur explication.

    `attribution[symbol]` décompose le poids final en étapes successives
    (ex. risk_allocation, signal_tilt, vol_target, constraints) dont la
    somme est égale au poids final (README §49.13).
    """

    weights: dict[str, float]
    attribution: dict[str, dict[str, float]] = field(default_factory=dict)
    portfolio_vol: float | None = None   # volatilité annualisée ex-ante
    vol_scale: float = 1.0
    binding: tuple[str, ...] = ()        # contraintes actives
