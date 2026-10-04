"""Optimiseur d'ordres (README §33) : comment exécuter un UREBALANCE.

Pour chaque symbole à traiter, on évalue une grille (agressivité a, durée Δt) :

    U(o) = P_fill(o) · (V(Δt) - C_exec(o)) - C_risk(o)

- V(Δt)   : valeur de l'exécution, part du bénéfice de la décision attribuée
            au symbole ; la part « alpha » (urgence) décroit avec la
            demi-vie du signal, la part « risque » ne décroit pas ;
- C_exec  : spread payé selon a, slippage, impact (racine carrée), frais ;
- C_risk  : risque de timing sur la partie non exécutée,
            λ · |notionnel| · σ_Δt · (1 - P_fill).

Un signal HF qui disparaît en 20 minutes est donc exécuté plus
agressivement qu'un écart de risque durable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta

from trading_engine.execution.cost_model import ExecutionCostModel
from trading_engine.execution.fill_model import FillModel
from trading_engine.execution.order_pricer import distance_to_execution, limit_price
from trading_engine.execution.orders import OrderProposal
from trading_engine.execution.volume import VolumeTracker
from trading_engine.timeutils import TRADING_HOURS_PER_DAY, parse_timeframe

TRADING_DAY = timedelta(hours=TRADING_HOURS_PER_DAY)


@dataclass(frozen=True)
class OptimizerConfig:
    aggressiveness: tuple[float, ...] = (0.0, 0.5, 1.0)
    durations: tuple[str, ...] = ("5m", "30m", "1h")
    alpha_half_life: str = "1h"
    timing_risk_aversion: float = 1.0
    allow_fractional: bool = False
    tick: float = 0.01


@dataclass(frozen=True)
class Candidate:
    aggressiveness: float
    duration: timedelta
    limit: float
    quantity: float
    expected_fill: float
    cost: float
    impact_fraction: float
    participation: float | None
    value: float
    timing_risk: float
    utility: float


class OrderOptimizer:
    def __init__(
        self,
        config: OptimizerConfig,
        cost_model: ExecutionCostModel,
        fill_model: FillModel,
        volume: VolumeTracker,
    ) -> None:
        self.config = config
        self.cost_model = cost_model
        self.fill_model = fill_model
        self.volume = volume
        self._half_life = parse_timeframe(config.alpha_half_life)
        self._durations = [parse_timeframe(d) for d in config.durations]

    def _quantity(self, notional: float, price: float) -> float:
        q = notional / price
        if self.config.allow_fractional:
            return q
        return float(math.trunc(q))            # vers zéro : jamais plus que demandé

    def candidates(
        self,
        symbol: str,
        notional: float,
        value: float,
        urgency: float,
        *,
        bid: float,
        ask: float,
        daily_vol: float | None,
    ) -> list[Candidate]:
        side = "buy" if notional > 0 else "sell"
        mid = (bid + ask) / 2
        rel_spread = (ask - bid) / mid
        sigma_day = daily_vol if daily_vol and daily_vol > 0 else self.cost_model.config.default_daily_vol
        adv = self.volume.adv(symbol)
        out = []
        for duration in self._durations:
            sigma_h = sigma_day * math.sqrt(duration / TRADING_DAY)
            market_volume = self.volume.expected_volume(symbol, duration)
            decay = math.exp(-math.log(2) * (duration / self._half_life))
            v = value * ((1 - urgency) + urgency * decay)
            for a in self.config.aggressiveness:
                limit = limit_price(side, bid, ask, a, self.config.tick)
                qty = self._quantity(abs(notional), limit) * (1 if side == "buy" else -1)
                if qty == 0:
                    continue
                dist = distance_to_execution(side, limit, bid, ask)
                fill = self.fill_model.expected_fill(qty, dist, sigma_h, market_volume)
                cost = self.cost_model.estimate(
                    qty, mid, rel_spread=rel_spread, crossing=a, daily_vol=sigma_day, adv=adv,
                    expected_market_volume=market_volume,
                )
                risk = self.config.timing_risk_aversion * abs(notional) * sigma_h * (1 - fill)
                utility = fill * (v - cost.total) - risk
                out.append(Candidate(a, duration, limit, qty, fill, cost.total, cost.impact_fraction,
                                     cost.participation, v, risk, utility))
        return out

    def optimize(
        self,
        symbol: str,
        notional: float,
        value: float,
        urgency: float,
        *,
        bid: float,
        ask: float,
        daily_vol: float | None,
        timestamp: datetime,
        decision_id: str | None = None,
    ) -> OrderProposal | None:
        cands = self.candidates(symbol, notional, value, urgency, bid=bid, ask=ask, daily_vol=daily_vol)
        if not cands:
            return None
        best = max(cands, key=lambda c: (c.utility, -c.aggressiveness))
        return OrderProposal(
            symbol=symbol, quantity=best.quantity, limit_price=round(best.limit, 10),
            timestamp=timestamp, decision_id=decision_id, aggressiveness=best.aggressiveness,
            duration=best.duration, arrival_mid=(bid + ask) / 2, expected_fill=best.expected_fill,
            expected_cost=best.cost, expected_impact=best.impact_fraction,
            participation=best.participation, urgency=urgency,
        )
