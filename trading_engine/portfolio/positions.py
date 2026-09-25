"""Positions (README §14, §35)."""

from __future__ import annotations

from dataclasses import dataclass, replace

_EPS = 1e-12


@dataclass(frozen=True)
class Position:
    symbol: str
    quantity: float
    avg_price: float
    realized_pnl: float = 0.0

    def apply_fill(self, quantity: float, price: float) -> "Position":
        """Retourne la position après exécution de `quantity` (signée) à `price`.

        - même sens : prix moyen pondéré ;
        - réduction : prix moyen inchangé, PnL réalisé ;
        - retournement : PnL réalisé sur l'ancienne jambe, nouveau prix moyen.
        """
        if price <= 0:
            raise ValueError(f"fill price must be positive, got {price}")
        if abs(quantity) < _EPS:
            return self

        old_q = self.quantity
        new_q = old_q + quantity

        if abs(old_q) < _EPS or old_q * quantity > 0:
            avg = (old_q * self.avg_price + quantity * price) / new_q
            return replace(self, quantity=new_q, avg_price=avg)

        closed = min(abs(quantity), abs(old_q))
        sign = 1.0 if old_q > 0 else -1.0
        realized = self.realized_pnl + sign * closed * (price - self.avg_price)

        if abs(new_q) < _EPS:
            return replace(self, quantity=0.0, avg_price=0.0, realized_pnl=realized)
        if old_q * new_q > 0:
            return replace(self, quantity=new_q, realized_pnl=realized)
        return replace(self, quantity=new_q, avg_price=price, realized_pnl=realized)


@dataclass(frozen=True)
class PositionState:
    """Photographie complète d'une position à un instant donné.

    Les champs de signal / régime / risque sont remplis par les moteurs des
    phases suivantes ; ils restent `None` tant qu'ils ne sont pas calculés.
    """

    symbol: str
    quantity: float
    price: float
    avg_price: float

    market_value: float
    pnl: float
    realized_pnl: float

    weight: float
    target_weight: float

    volatility: float | None = None
    risk_contribution: float | None = None

    signal_hf: float | None = None
    signal_st: float | None = None
    signal_mt: float | None = None
    signal_lt: float | None = None

    regime: str | None = None
    regime_probability: float | None = None

    @property
    def drift(self) -> float:
        return self.weight - self.target_weight
