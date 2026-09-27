"""Ordres proposés, plans d'exécution et fills (README §32-33).

Un `OrderProposal` est une recommandation technique : il ne devient un
ordre réel que par une décision humaine / broker (ici : paper trading).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass(frozen=True)
class OrderProposal:
    symbol: str
    quantity: float                        # signée : > 0 achat, < 0 vente
    limit_price: float
    timestamp: datetime
    decision_id: str | None = None
    # choix de l'optimiseur
    aggressiveness: float = 1.0            # 0 = passif (bid/ask), 1 = croise le spread
    duration: timedelta = timedelta(minutes=30)
    # attentes (modèles d'exécution)
    arrival_mid: float | None = None       # mid au moment de la proposition
    expected_fill: float | None = None     # fraction attendue exécutée pendant `duration`
    expected_cost: float | None = None     # spread + impact + frais attendus (devise)
    expected_impact: float | None = None   # impact attendu (fraction du prix)
    participation: float | None = None     # quantité / volume de marché attendu
    urgency: float = 0.0

    @property
    def side(self) -> str:
        return "buy" if self.quantity > 0 else "sell"

    @property
    def notional(self) -> float:
        return abs(self.quantity) * self.limit_price

    @property
    def expires_at(self) -> datetime:
        return self.timestamp + self.duration


@dataclass(frozen=True)
class Fill:
    symbol: str
    quantity: float                        # signée
    price: float
    timestamp: datetime
    decision_id: str | None = None


@dataclass(frozen=True)
class RejectedOrder:
    order: OrderProposal
    violations: tuple[str, ...]


@dataclass(frozen=True)
class ExecutionPlan:
    decision_id: str
    timestamp: datetime | None
    orders: tuple[OrderProposal, ...]
    rejected: tuple[RejectedOrder, ...] = ()
    expected_cost: float = 0.0
    execution_confidence: float | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)
