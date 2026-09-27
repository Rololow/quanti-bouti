"""Hard controls (README §50.2) : veto final sur les cibles et les ordres.

Distincts du Constraint Engine (qui façonne la cible) : ce sont des limites
fixes, configurées à part, que les modèles ne voient ni ne modifient. Même si
un modèle devient fou (`target = 0.99`), rien ne passe au-delà.

Même logique que les contrôles pré-trade d'accès au marché : bloquer les
ordres erronés ou hors limites prédéfinies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import Mapping

from trading_engine.execution.orders import OrderProposal
from trading_engine.safety.safety_engine import SafetyState

# Tolérance d'arrondi : un ordre dimensionné exactement à une limite ne doit
# pas être rejeté à cause de l'arithmétique flottante (0.2 + 0.1 > 0.3).
EPS = 1e-9

__all__ = ["HardControls", "HardLimits", "ControlResult", "OrderProposal"]


@dataclass(frozen=True)
class HardLimits:
    max_target_weight: float = 0.50
    max_target_gross: float = 1.0
    max_order_notional: float = 50_000.0
    max_order_weight: float = 0.20        # notionnel / valeur du portefeuille
    max_participation: float = 0.10       # quantité / volume récent
    price_collar: float = 0.05            # |limite / dernier prix - 1|
    max_orders_per_day: int = 50
    max_daily_turnover: float = 1.0       # somme des notionnels / valeur, par jour


@dataclass(frozen=True)
class ControlResult:
    approved: bool
    violations: tuple[str, ...] = ()


class HardControls:
    def __init__(self, limits: HardLimits | None = None) -> None:
        self.limits = limits or HardLimits()
        self._orders: dict[date, int] = {}
        self._turnover: dict[date, float] = {}

    def validate_targets(self, weights: Mapping[str, float]) -> ControlResult:
        lim = self.limits
        violations = []
        for sym, w in sorted(weights.items()):
            if not math.isfinite(w):
                violations.append(f"NON_FINITE_TARGET {sym}")
            elif abs(w) > lim.max_target_weight + EPS:
                violations.append(f"TARGET_WEIGHT {sym} {w:.3f} > {lim.max_target_weight:.3f}")
        gross = sum(abs(w) for w in weights.values() if math.isfinite(w))
        if gross > lim.max_target_gross + EPS:
            violations.append(f"TARGET_GROSS {gross:.3f} > {lim.max_target_gross:.3f}")
        return ControlResult(not violations, tuple(violations))

    def max_order_quantity(
        self, price: float, portfolio_value: float, recent_volume: float | None,
        when: datetime | None = None,
    ) -> float:
        """Plus grande quantité compatible avec les limites de taille, de
        participation et le budget de turnover restant du jour (pour
        dimensionner les ordres avant contrôle)."""
        lim = self.limits
        caps = [lim.max_order_notional / price]
        if portfolio_value > 0:
            caps.append(lim.max_order_weight * portfolio_value / price)
            if when is not None:
                remaining = lim.max_daily_turnover - self._turnover.get(when.date(), 0.0)
                caps.append(max(0.0, remaining) * portfolio_value / price)
        if recent_volume is not None and recent_volume > 0:
            caps.append(lim.max_participation * recent_volume)
        return max(0.0, min(caps))

    def check_order(
        self,
        order: OrderProposal,
        *,
        portfolio_value: float,
        last_price: float | None,
        recent_volume: float | None,
        safety_state: SafetyState,
    ) -> ControlResult:
        lim = self.limits
        v: list[str] = []
        day = order.timestamp.date()
        if safety_state is SafetyState.HALTED:
            v.append("SAFETY_HALTED")
        if not (math.isfinite(order.quantity) and math.isfinite(order.limit_price)) or order.limit_price <= 0:
            v.append("INVALID_ORDER")
        else:
            if order.notional > lim.max_order_notional * (1 + EPS):
                v.append(f"ORDER_NOTIONAL {order.notional:,.0f} > {lim.max_order_notional:,.0f}")
            if portfolio_value > 0 and order.notional / portfolio_value > lim.max_order_weight + EPS:
                v.append(f"ORDER_WEIGHT {order.notional / portfolio_value:.3f} > {lim.max_order_weight:.3f}")
            if last_price is None:
                v.append("NO_REFERENCE_PRICE")
            elif abs(order.limit_price / last_price - 1.0) > lim.price_collar:
                v.append(f"PRICE_COLLAR {order.limit_price} vs {last_price}")
            if recent_volume is not None and recent_volume > 0 and abs(order.quantity) / recent_volume > lim.max_participation + EPS:
                v.append(f"PARTICIPATION {abs(order.quantity) / recent_volume:.3f} > {lim.max_participation:.3f}")
            turnover = self._turnover.get(day, 0.0) + (order.notional / portfolio_value if portfolio_value > 0 else math.inf)
            if turnover > lim.max_daily_turnover + EPS:
                v.append(f"DAILY_TURNOVER {turnover:.3f} > {lim.max_daily_turnover:.3f}")
        if self._orders.get(day, 0) >= lim.max_orders_per_day:
            v.append(f"ORDERS_PER_DAY {lim.max_orders_per_day}")

        if not v:
            self._orders[day] = self._orders.get(day, 0) + 1
            self._turnover[day] = self._turnover.get(day, 0.0) + order.notional / portfolio_value
        return ControlResult(not v, tuple(v))
