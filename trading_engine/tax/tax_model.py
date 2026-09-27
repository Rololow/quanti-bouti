"""Modèle fiscal : applique un `TaxProfile` aux opérations du moteur.

Sert à deux choses :

1. comptabiliser les taxes réellement dues (fills, dividendes) ;
2. estimer **avant** de décider le coût fiscal d'un rebalancement
   (taxe sur transactions + impôt marginal sur les plus-values), qui entre
   dans l'alpha net : alpha_net = alpha - coûts - C_tax (README §49.9).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Mapping

from trading_engine.tax.capital_gains import CapitalGainsTracker, RealizedGain
from trading_engine.tax.profile import Instrument, TaxProfile

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaxCharge:
    amount: float
    rule_id: str | None
    rate: float
    capped: bool = False


@dataclass(frozen=True)
class DividendTax:
    gross: float
    foreign_withholding: float
    local_tax: float

    @property
    def net(self) -> float:
        return self.gross - self.foreign_withholding - self.local_tax


@dataclass(frozen=True)
class RebalanceTaxCost:
    transaction_tax: float
    capital_gains_tax: float
    per_symbol: dict[str, float] = field(default_factory=dict)

    @property
    def total(self) -> float:
        return self.transaction_tax + self.capital_gains_tax


class TaxModel:
    def __init__(
        self,
        profile: TaxProfile,
        instruments: Mapping[str, Instrument] | None = None,
        step_up_prices: Mapping[str, float] | None = None,
    ) -> None:
        self.profile = profile
        self.instruments = dict(instruments or {})
        self.gains = (
            CapitalGainsTracker(profile.capital_gains, step_up_prices)
            if profile.capital_gains is not None else None
        )
        self.transaction_taxes_paid = 0.0
        self._unknown: set[str] = set()

    def instrument(self, symbol: str) -> Instrument:
        inst = self.instruments.get(symbol)
        if inst is None:
            if symbol not in self._unknown:
                logger.warning("no tax classification for %s: using the default rule", symbol)
                self._unknown.add(symbol)
            inst = Instrument(symbol, asset_class="unknown")
        return inst

    # ------------------------------------------------------------- transactions

    def transaction_tax(self, symbol: str, notional: float, side: str) -> TaxCharge:
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
        if side not in self.profile.transaction_tax_sides or notional <= 0:
            return TaxCharge(0.0, None, 0.0)
        rule = self.profile.transaction_rule(self.instrument(symbol))
        if rule is None:
            return TaxCharge(0.0, None, 0.0)
        amount = abs(notional) * rule.rate
        capped = rule.cap is not None and amount > rule.cap
        return TaxCharge(min(amount, rule.cap) if capped else amount, rule.id, rule.rate, capped)

    def record_fill(
        self, symbol: str, quantity: float, price: float, when: date | datetime
    ) -> tuple[TaxCharge, RealizedGain | None]:
        """Comptabilise un fill : taxe sur transaction et lots de plus-values."""
        side = "buy" if quantity > 0 else "sell"
        charge = self.transaction_tax(symbol, abs(quantity) * price, side)
        self.transaction_taxes_paid += charge.amount
        realized = None
        if self.gains is not None:
            if quantity > 0:
                self.gains.buy(symbol, quantity, price, when)
            else:
                realized = self.gains.sell(symbol, -quantity, price, when)
        return charge, realized

    # ------------------------------------------------------------- revenus

    def dividend_tax(self, symbol: str, gross: float) -> DividendTax:
        inc = self.profile.income
        domicile = self.instrument(symbol).domicile
        withholding_rate = 0.0
        if domicile and domicile != self.profile.country:
            withholding_rate = inc.foreign_withholding.get(domicile, inc.foreign_withholding.get("default", 0.0))
        foreign = gross * withholding_rate
        local = (gross - foreign) * inc.dividend_rate
        return DividendTax(gross, foreign, local)

    # ------------------------------------------------------------- estimation

    def rebalance_cost(
        self,
        trades: Mapping[str, float],
        prices: Mapping[str, float],
        when: date | datetime,
    ) -> RebalanceTaxCost:
        """Coût fiscal estimé de trades (notionnels signés : > 0 achat)."""
        tx_total, cg_total, per_symbol = 0.0, 0.0, {}
        extra_gain = 0.0
        year = (when.date() if isinstance(when, datetime) else when).year
        for sym, notional in sorted(trades.items()):
            if abs(notional) < 1e-9:
                continue
            side = "buy" if notional > 0 else "sell"
            tx = self.transaction_tax(sym, abs(notional), side).amount
            tx_total += tx
            per_symbol[sym] = tx
            if side == "sell" and self.gains is not None and prices.get(sym):
                qty = abs(notional) / prices[sym]
                covered, basis = self.gains.preview_sale(sym, qty)
                extra_gain += covered * prices[sym] - basis
        if self.gains is not None and extra_gain:
            eff = self.profile.capital_gains.effective_from
            if eff is None or (when.date() if isinstance(when, datetime) else when) >= eff:
                cg_total = self.gains.tax_due(year, extra_gain) - self.gains.tax_due(year)
        return RebalanceTaxCost(tx_total, cg_total, per_symbol)

    def account_tax(self, average_value: float) -> float:
        acc = self.profile.account_tax
        if acc is None or average_value <= acc.threshold:
            return 0.0
        return average_value * acc.rate

    def warnings(self) -> list[str]:
        out = []
        if not self.profile.verified:
            out.append(f"profil fiscal {self.profile.country} non vérifié (meta.verified_on vide)")
        cg = self.profile.capital_gains
        if cg is not None and cg.speculative_warning:
            out.append(cg.speculative_warning)
        if self._unknown:
            out.append(f"instruments sans classification fiscale : {', '.join(sorted(self._unknown))}")
        return out
