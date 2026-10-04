"""Coûts d'exécution et impact de marché (README §32).

    C(q) = C_spread + C_slippage + C_impact + C_fees

- spread : un ordre qui croise paie la moitié du spread par rapport au mid ;
  un ordre passif (au bid / à l'ask) n'en paie pas (mais risque de ne pas
  être exécuté : voir `fill_model`) ;
- impact : loi en racine carrée, classique et robuste
      impact = η · σ_jour · sqrt(Q / ADV)            (fraction du prix)
  η est appris par la boucle d'exécution (`feedback`) ; au départ c'est une
  hypothèse (η ≈ 0.5), d'où une confiance d'exécution faible ;
- frais : commission par ordre + par titre (Alpaca : 0).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CostConfig:
    default_spread_bps: float = 5.0
    slippage_bps: float = 1.0
    impact_eta: float = 0.5              # prior, recalibré par le feedback
    commission_per_order: float = 0.0
    commission_per_share: float = 0.0
    default_daily_vol: float = 0.015     # si la volatilité est inconnue
    default_adv: float = 1_000_000.0     # si le volume est inconnu (titres / jour)


@dataclass(frozen=True)
class CostEstimate:
    spread: float
    slippage: float
    impact: float
    fees: float
    impact_fraction: float
    participation: float | None

    @property
    def total(self) -> float:
        return self.spread + self.slippage + self.impact + self.fees


class ExecutionCostModel:
    def __init__(self, config: CostConfig | None = None) -> None:
        self.config = config or CostConfig()
        self.eta = self.config.impact_eta

    def impact_fraction(self, quantity: float, daily_vol: float | None, adv: float | None) -> float:
        cfg = self.config
        sigma = daily_vol if daily_vol and daily_vol > 0 else cfg.default_daily_vol
        volume = adv if adv and adv > 0 else cfg.default_adv
        return self.eta * sigma * math.sqrt(abs(quantity) / volume)

    def estimate(
        self,
        quantity: float,
        price: float,
        *,
        rel_spread: float | None = None,
        crossing: float = 1.0,            # part du demi-spread payée (0 passif, 1 croise)
        daily_vol: float | None = None,
        adv: float | None = None,
        expected_market_volume: float | None = None,
    ) -> CostEstimate:
        cfg = self.config
        notional = abs(quantity) * price
        spread = rel_spread if rel_spread is not None else cfg.default_spread_bps * 1e-4
        impact_frac = self.impact_fraction(quantity, daily_vol, adv)
        participation = (
            abs(quantity) / expected_market_volume
            if expected_market_volume and expected_market_volume > 0 else None
        )
        return CostEstimate(
            spread=notional * spread / 2.0 * max(0.0, min(crossing, 1.0)),
            slippage=notional * cfg.slippage_bps * 1e-4,
            impact=notional * impact_frac,
            fees=(cfg.commission_per_order + cfg.commission_per_share * abs(quantity)) if quantity else 0.0,
            impact_fraction=impact_frac,
            participation=participation,
        )
