"""Paper broker : exécute les OrderProposal sur le flux de marché, sans
jamais rien envoyer à un broker réel (README §3 : signal ≠ ordre).

Règles (volontairement prudentes) :

- un ordre d'achat est exécutable sur un trade au prix <= limite (vente :
  >= limite) ; il est exécuté **au prix limite** ;
- quantité par trade <= `fill_share` × taille du trade imprimé ;
- un nouvel ordre sur un symbole remplace l'ordre en cours ;
- un ordre non exécuté à `expires_at` est annulé (le reste n'est pas fait).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from trading_engine.data.events import TradeEvent
from trading_engine.execution.orders import Fill, OrderProposal


@dataclass
class WorkingOrder:
    order: OrderProposal
    filled: float = 0.0
    notional: float = 0.0

    @property
    def remaining(self) -> float:
        return abs(self.order.quantity) - abs(self.filled)

    @property
    def average_price(self) -> float | None:
        return self.notional / abs(self.filled) if self.filled else None


@dataclass
class BrokerUpdate:
    fills: list[Fill] = field(default_factory=list)
    closed: list[WorkingOrder] = field(default_factory=list)


class PaperBroker:
    def __init__(self, fill_share: float = 1.0) -> None:
        self.fill_share = fill_share
        self.working: dict[str, WorkingOrder] = {}

    def submit(self, order: OrderProposal) -> WorkingOrder | None:
        """Soumet un ordre ; retourne l'ordre remplacé s'il y en avait un."""
        replaced = self.working.pop(order.symbol, None)
        self.working[order.symbol] = WorkingOrder(order)
        return replaced

    def cancel_all(self) -> list[WorkingOrder]:
        closed = list(self.working.values())
        self.working.clear()
        return closed

    def on_trade(self, trade: TradeEvent) -> BrokerUpdate:
        update = BrokerUpdate()
        for sym in sorted(self.working):
            wo = self.working[sym]
            if trade.timestamp >= wo.order.expires_at:
                update.closed.append(self.working.pop(sym))

        wo = self.working.get(trade.symbol)
        if wo is None or trade.timestamp < wo.order.timestamp:
            return update
        order = wo.order
        executable = trade.price <= order.limit_price if order.quantity > 0 else trade.price >= order.limit_price
        if not executable or trade.size <= 0:
            return update
        qty = min(wo.remaining, self.fill_share * trade.size)
        if qty <= 0:
            return update
        signed = qty if order.quantity > 0 else -qty
        wo.filled += signed
        wo.notional += qty * order.limit_price
        update.fills.append(Fill(order.symbol, signed, order.limit_price, trade.timestamp, order.decision_id))
        if wo.remaining <= 1e-9:
            update.closed.append(self.working.pop(trade.symbol))
        return update
