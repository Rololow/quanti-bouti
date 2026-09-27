"""Boucle d'apprentissage de l'exécution (README §34, §49.8).

À la clôture de chaque ordre (entièrement exécuté ou expiré) :

- fill : calibration = EWMA(fill réalisé) / EWMA(fill prédit), bornée ;
- coût : implementation shortfall réalisé (prix moyen vs mid à l'arrivée)
  comparé au coût prédit ; η (impact) est recalibré si `learn_impact` ;
- confiance d'exécution : croît avec le nombre d'ordres observés et
  décroît avec l'erreur de prédiction :
      conf = n / (n + n0) · max(0, 1 - erreur relative moyenne)
  Elle vaut 0 au départ : les modèles d'exécution ne reposent alors que sur
  des hypothèses, et la décision doit le savoir (MODEL ≠ EXECUTION CONFIDENCE).

En paper trading, les fills sont simulés : ils renseignent la probabilité
d'exécution mais pas l'impact réel ; `learn_impact` est donc désactivé par
défaut en paper.
"""

from __future__ import annotations

from dataclasses import dataclass

from trading_engine.execution.cost_model import ExecutionCostModel
from trading_engine.execution.fill_model import FillModel
from trading_engine.execution.orders import OrderProposal


@dataclass(frozen=True)
class ExecutionRecord:
    order: OrderProposal
    filled_quantity: float
    average_price: float | None
    fill_ratio: float
    shortfall: float | None          # coût réalisé vs mid d'arrivée (fraction, > 0 = coût)
    predicted_cost: float | None     # fraction


class ExecutionFeedback:
    def __init__(
        self,
        fill_model: FillModel,
        cost_model: ExecutionCostModel,
        *,
        lam: float = 0.9,
        prior_orders: int = 20,
        learn_impact: bool = False,
    ) -> None:
        self.fill_model = fill_model
        self.cost_model = cost_model
        self.lam = lam
        self.prior_orders = prior_orders
        self.learn_impact = learn_impact
        self.records: list[ExecutionRecord] = []
        self._fill_num = self._fill_den = 0.0
        self._cost_num = self._cost_den = 0.0
        self._error: float | None = None

    def _ewma(self, current: float, value: float) -> float:
        return self.lam * current + (1 - self.lam) * value

    def on_order_closed(self, order: OrderProposal, filled_quantity: float,
                        average_price: float | None) -> ExecutionRecord:
        qty = abs(order.quantity)
        ratio = min(abs(filled_quantity) / qty, 1.0) if qty else 0.0
        shortfall = predicted = None
        if order.expected_fill is not None:
            self._fill_num = self._ewma(self._fill_num, ratio)
            self._fill_den = self._ewma(self._fill_den, order.expected_fill)
            if self._fill_den > 1e-6:
                self.fill_model.calibration = max(0.2, min(2.0, self._fill_num / self._fill_den))
            err = abs(ratio - order.expected_fill)
            self._error = err if self._error is None else self._ewma(self._error, err)
        if filled_quantity and average_price and order.arrival_mid:
            sign = 1.0 if order.quantity > 0 else -1.0
            shortfall = sign * (average_price - order.arrival_mid) / order.arrival_mid
            if order.expected_cost is not None and order.notional > 0:
                predicted = order.expected_cost / order.notional
                if self.learn_impact and predicted > 0:
                    self._cost_num = self._ewma(self._cost_num, max(shortfall, 0.0))
                    self._cost_den = self._ewma(self._cost_den, predicted)
                    if self._cost_den > 0:
                        base = self.cost_model.config.impact_eta
                        self.cost_model.eta = max(0.05, min(5.0, base * self._cost_num / self._cost_den))
        record = ExecutionRecord(order, filled_quantity, average_price, ratio, shortfall, predicted)
        self.records.append(record)
        return record

    @property
    def execution_confidence(self) -> float:
        n = len(self.records)
        if n == 0 or self._error is None:
            return 0.0
        return n / (n + self.prior_orders) * max(0.0, 1.0 - self._error)
