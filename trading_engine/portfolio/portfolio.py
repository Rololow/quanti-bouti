"""Portefeuille et son état agrégé (README §14, §17)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping

from trading_engine.portfolio.positions import Position, PositionState


@dataclass(frozen=True)
class PortfolioState:
    timestamp: datetime | None
    cash: float
    market_value: float
    total_value: float
    unrealized_pnl: float
    realized_pnl: float
    gross_exposure: float
    positions: tuple[PositionState, ...]

    @property
    def leverage(self) -> float:
        return self.gross_exposure / self.total_value if self.total_value else 0.0

    def position(self, symbol: str) -> PositionState | None:
        return next((p for p in self.positions if p.symbol == symbol), None)


class Portfolio:
    def __init__(
        self,
        cash: float = 0.0,
        positions: Mapping[str, Position] | None = None,
        target_weights: Mapping[str, float] | None = None,
    ) -> None:
        self.cash = cash
        self.positions: dict[str, Position] = dict(positions or {})
        self.target_weights: dict[str, float] = dict(target_weights or {})
        self.prices: dict[str, float] = {}
        self.last_update: datetime | None = None

    def update_price(self, symbol: str, price: float, timestamp: datetime | None = None) -> None:
        self.prices[symbol] = price
        if timestamp is not None and (self.last_update is None or timestamp > self.last_update):
            self.last_update = timestamp

    def apply_fill(self, symbol: str, quantity: float, price: float) -> Position:
        current = self.positions.get(symbol, Position(symbol, 0.0, 0.0))
        updated = current.apply_fill(quantity, price)
        self.positions[symbol] = updated
        self.cash -= quantity * price
        return updated

    def set_target_weights(self, weights: Mapping[str, float]) -> None:
        self.target_weights = dict(weights)

    def _price(self, pos: Position) -> float:
        # Tant qu'aucun prix n'est reçu, on valorise au prix moyen.
        return self.prices.get(pos.symbol, pos.avg_price)

    def snapshot(
        self,
        volatilities: Mapping[str, float | None] | None = None,
    ) -> PortfolioState:
        volatilities = volatilities or {}
        values = {sym: pos.quantity * self._price(pos) for sym, pos in self.positions.items()}
        market_value = sum(values.values())
        total_value = self.cash + market_value

        symbols = sorted(set(self.positions) | set(self.target_weights))
        states = []
        for sym in symbols:
            pos = self.positions.get(sym, Position(sym, 0.0, 0.0))
            price = self.prices.get(sym, pos.avg_price)
            mv = values.get(sym, 0.0)
            states.append(
                PositionState(
                    symbol=sym,
                    quantity=pos.quantity,
                    price=price,
                    avg_price=pos.avg_price,
                    market_value=mv,
                    pnl=pos.quantity * (price - pos.avg_price),
                    realized_pnl=pos.realized_pnl,
                    weight=mv / total_value if total_value else 0.0,
                    target_weight=self.target_weights.get(sym, 0.0),
                    volatility=volatilities.get(sym),
                )
            )

        return PortfolioState(
            timestamp=self.last_update,
            cash=self.cash,
            market_value=market_value,
            total_value=total_value,
            unrealized_pnl=sum(s.pnl for s in states),
            realized_pnl=sum(s.realized_pnl for s in states),
            gross_exposure=sum(abs(v) for v in values.values()),
            positions=tuple(states),
        )
