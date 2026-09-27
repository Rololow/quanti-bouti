"""Modèle fiscal : applique un `TaxProfile` aux opérations du moteur.

Sert à deux choses :

1. comptabiliser les taxes réellement dues (fills, dividendes) ;
2. estimer **avant** de décider le coût fiscal d'un rebalancement
   (taxe sur transactions + impôt marginal sur les plus-values), qui entre
   dans l'alpha net : alpha_net = alpha - coûts - C_tax (README §49.9).

Devises : les montants reçus sont dans la devise du portefeuille (ex. USD) ;
lots, plus-values, TOB et plafonds sont tenus dans la devise du profil
(ex. EUR), convertis au cours du jour de chaque opération (`FxRates`).
Les estimations pour la décision sont rendues dans la devise du portefeuille.

Registre : `snapshot()` / `restore()` (lots datés, plus-values réalisées,
taxes sur transactions) pour survivre aux redémarrages.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Mapping

from trading_engine.tax.capital_gains import CapitalGainsTracker, Lot, RealizedGain, _as_date
from trading_engine.tax.fx import FxRates
from trading_engine.tax.profile import Instrument, TaxProfile

LEDGER_VERSION = 1

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaxCharge:
    amount: float                          # devise du profil (ex. EUR)
    rule_id: str | None
    rate: float
    capped: bool = False
    portfolio_amount: float | None = None  # même montant dans la devise du portefeuille


@dataclass(frozen=True)
class TransactionTaxRecord:
    """Ligne de déclaration (ex. TOB) : montants dans la devise du profil."""

    day: date
    symbol: str
    side: str
    notional: float
    amount: float
    rule_id: str | None


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
        portfolio_currency: str | None = None,
    ) -> None:
        self.profile = profile
        self.instruments = dict(instruments or {})
        self.portfolio_currency = portfolio_currency or profile.currency
        self.fx: FxRates | None = None
        self.fx_missing = 0                  # conversions faites sans taux (1:1)
        self._step_up_raw = dict(step_up_prices or {})   # devise du portefeuille
        self.gains = (
            CapitalGainsTracker(profile.capital_gains, step_up_prices)
            if profile.capital_gains is not None else None
        )
        self.transaction_taxes_paid = 0.0
        self.transactions: list[TransactionTaxRecord] = []
        self._unknown: set[str] = set()

    # ------------------------------------------------------------- devises

    @property
    def needs_fx(self) -> bool:
        return self.portfolio_currency != self.profile.currency

    def set_fx(self, fx: FxRates) -> None:
        if (fx.base, fx.quote) != (self.profile.currency, self.portfolio_currency):
            raise ValueError(f"FX {fx.base}/{fx.quote} does not match "
                             f"{self.profile.currency}/{self.portfolio_currency}")
        self.fx = fx
        # Prix de step-up (devise du portefeuille) -> devise du profil, au cours
        # de la date de référence.
        cg = self.profile.capital_gains
        if self.gains is not None and self._step_up_raw:
            day = cg.step_up_date or (cg.effective_from - timedelta(days=1) if cg.effective_from else None)
            if day is not None:
                self.gains.step_up_prices = {s: fx.to_base(p, day) for s, p in self._step_up_raw.items()}

    def to_tax(self, amount: float, when: date | datetime) -> float:
        """Devise du portefeuille -> devise du profil."""
        if not self.needs_fx:
            return amount
        if self.fx is None or not len(self.fx):
            self.fx_missing += 1
            return amount
        return self.fx.to_base(amount, when)

    def to_portfolio(self, amount: float, when: date | datetime) -> float:
        if not self.needs_fx:
            return amount
        if self.fx is None or not len(self.fx):
            self.fx_missing += 1
            return amount
        return self.fx.to_quote(amount, when)

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
        """Comptabilise un fill (prix dans la devise du portefeuille) : taxe
        sur transaction et lots de plus-values (devise du profil)."""
        side = "buy" if quantity > 0 else "sell"
        price_tax = self.to_tax(price, when)
        notional = abs(quantity) * price_tax
        charge = self.transaction_tax(symbol, notional, side)
        charge = TaxCharge(charge.amount, charge.rule_id, charge.rate, charge.capped,
                           self.to_portfolio(charge.amount, when))
        self.transaction_taxes_paid += charge.amount
        if charge.amount:
            self.transactions.append(TransactionTaxRecord(_as_date(when), symbol, side, notional,
                                                          charge.amount, charge.rule_id))
        realized = None
        if self.gains is not None:
            if quantity > 0:
                self.gains.buy(symbol, quantity, price_tax, when)
            else:
                realized = self.gains.sell(symbol, -quantity, price_tax, when)
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
        """Coût fiscal estimé de trades (notionnels signés : > 0 achat).
        Entrées et résultat dans la devise du portefeuille."""
        tx_total, cg_total, per_symbol = 0.0, 0.0, {}
        extra_gain = 0.0
        year = (when.date() if isinstance(when, datetime) else when).year
        trades = {s: self.to_tax(v, when) for s, v in trades.items()}
        prices = {s: self.to_tax(v, when) for s, v in prices.items() if v}
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
        back = lambda v: self.to_portfolio(v, when)   # noqa: E731
        return RebalanceTaxCost(back(tx_total), back(cg_total), {s: back(v) for s, v in per_symbol.items()})

    # ------------------------------------------------------------- registre

    def lot_quantity(self, symbol: str) -> float:
        if self.gains is None:
            return 0.0
        return sum(lot.quantity for lot in self.gains.lots.get(symbol, ()))

    def align_lots(self, symbol: str, quantity: float, price: float | None, when: datetime,
                   *, realize: bool) -> float:
        """Aligne les lots sur une quantité détenue connue (broker).

        Excédent : nouveau lot au `price` (devise du portefeuille) daté de `when`.
        Manque : lots consommés en FIFO ; avec `realize`, comme une vente au
        `price` (fill manqué), sinon retirés sans plus-value (origine inconnue).
        Retourne l'écart appliqué (> 0 ajouté).
        """
        if self.gains is None:
            return 0.0
        diff = max(quantity, 0.0) - self.lot_quantity(symbol)
        if abs(diff) <= 1e-9:
            return 0.0
        if diff > 0:
            if not price:
                return 0.0
            self.gains.buy(symbol, diff, self.to_tax(price, when), when)
        elif realize and price:
            self.gains.sell(symbol, -diff, self.to_tax(price, when), when)
        else:
            self.gains.remove(symbol, -diff)
        return diff

    def snapshot(self) -> dict:
        gains = self.gains
        return {
            "version": LEDGER_VERSION, "country": self.profile.country, "currency": self.profile.currency,
            "lots": {} if gains is None else {
                sym: [[lot.quantity, lot.unit_cost, lot.acquired.isoformat()] for lot in lots]
                for sym, lots in sorted(gains.lots.items()) if lots
            },
            "realized": [] if gains is None else [
                [g.symbol, g.quantity, g.proceeds, g.tax_basis, g.sold.isoformat()] for g in gains.realized
            ],
            "transactions": [
                [t.day.isoformat(), t.symbol, t.side, t.notional, t.amount, t.rule_id] for t in self.transactions
            ],
        }

    def restore(self, ledger: Mapping) -> None:
        if ledger.get("version") != LEDGER_VERSION:
            raise ValueError(f"unsupported tax ledger version {ledger.get('version')!r}")
        if ledger.get("currency") != self.profile.currency or ledger.get("country") != self.profile.country:
            raise ValueError(f"tax ledger is for {ledger.get('country')}/{ledger.get('currency')}, "
                             f"profile is {self.profile.country}/{self.profile.currency}")
        if self.gains is not None:
            self.gains.lots = {
                sym: deque(Lot(float(q), float(c), date.fromisoformat(d)) for q, c, d in lots)
                for sym, lots in ledger.get("lots", {}).items()
            }
            self.gains.realized = [
                RealizedGain(sym, float(q), float(p), float(b), date.fromisoformat(d))
                for sym, q, p, b, d in ledger.get("realized", [])
            ]
        self.transactions = [
            TransactionTaxRecord(date.fromisoformat(d), sym, side, float(n), float(a), rule)
            for d, sym, side, n, a, rule in ledger.get("transactions", [])
        ]
        self.transaction_taxes_paid = sum(t.amount for t in self.transactions)

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
        if self.needs_fx and self.fx is None:
            out.append(f"pas de taux {self.profile.currency}/{self.portfolio_currency} : montants non convertis")
        if self.fx is not None and self.fx.before_first:
            out.append(f"{self.fx.before_first} conversion(s) avant le premier taux connu ({self.fx.first})")
        if self._unknown:
            out.append(f"instruments sans classification fiscale : {', '.join(sorted(self._unknown))}")
        return out
