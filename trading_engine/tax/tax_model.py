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

import bisect
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
    # Taxe sur la composante intérêts due à la vente (Reynders), devise du
    # portefeuille ; à part de la taxe sur transaction.
    interest_tax: float = 0.0


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
    interest_tax: float = 0.0              # Reynders déclenchée par les ventes

    @property
    def total(self) -> float:
        return self.transaction_tax + self.capital_gains_tax + self.interest_tax


def as_if_current_rules(profile: TaxProfile) -> TaxProfile:
    """Profil dont la taxe sur les plus-values s'applique depuis toujours (sans
    step-up) : un backtest historique mesure alors les règles d'aujourd'hui.
    Prudent : sans date d'entrée en vigueur, le report de l'exonération non
    utilisée n'est pas compté (exonération annuelle seule)."""
    import dataclasses

    cg = profile.capital_gains
    if cg is None:
        return profile
    return dataclasses.replace(profile, capital_gains=dataclasses.replace(cg, effective_from=None, step_up_date=None))


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
        self.interest_taxes_paid = 0.0       # Reynders (devise du profil)
        self.gains_taxes_paid = 0.0          # taxe annuelle sur les plus-values (devise du profil)
        self.transactions: list[TransactionTaxRecord] = []
        self._unknown: set[str] = set()
        # Composante intérêts cumulée par part (TIS), par symbole : (jours, cumul).
        self._interest_index: dict[str, tuple[list[date], list[float]]] = {}

    # ------------------------------------------------------------- Reynders

    def set_interest_index(self, payload: Mapping[str, list]) -> None:
        """{symbole: [[jour ISO, intérêts cumulés par part], ...]} (devise du
        prix de la part). Sans index, repli prudent : plus-value × part en créances."""
        self._interest_index = {}
        for sym, rows in payload.items():
            rows = sorted((date.fromisoformat(d), float(v)) for d, v in rows)
            self._interest_index[sym] = ([d for d, _ in rows], [v for _, v in rows])

    def interest_index_payload(self) -> dict:
        return {s: [[d.isoformat(), v] for d, v in zip(days, vals)]
                for s, (days, vals) in self._interest_index.items()}

    def accrued_interest(self, symbol: str, day: date) -> float | None:
        idx = self._interest_index.get(symbol)
        if idx is None:
            return None
        i = bisect.bisect_right(idx[0], day) - 1
        return idx[1][i] if i >= 0 else 0.0

    def interest_tax_applies(self, symbol: str) -> bool:
        inc = self.profile.income
        inst = self.instrument(symbol)
        return (inc.interest_component_at_sale and inc.interest_component_rate > 0
                and inst.bond_share > inc.interest_component_threshold)

    def interest_tax_on_sale(self, symbol: str, quantity: float, price: float, when: date | datetime) -> float:
        """Reynders d'une vente (prix et résultat dans la devise du portefeuille) :
        taux × part en créances × intérêts accumulés pendant la détention des
        lots vendus (FIFO). Sans index d'intérêts : plus-value × part en créances."""
        if quantity <= 0 or self.gains is None or not self.interest_tax_applies(symbol):
            return 0.0
        day = _as_date(when)
        inst = self.instrument(symbol)
        share = min(inst.bond_share, 1.0)
        now = self.accrued_interest(symbol, day)
        taxable = 0.0
        for take, lot in self.gains.preview_lots(symbol, quantity):
            if now is not None:
                taxable += take * max(0.0, now - (self.accrued_interest(symbol, lot.acquired) or 0.0))
            else:
                # lots en devise du profil : plus-value convertie au cours du jour
                gain = take * (self.to_tax(price, day) - lot.unit_cost)
                taxable += self.to_portfolio(gain, day) if gain > 0 else 0.0
        return self.profile.income.interest_component_rate * share * taxable

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
                interest = self.interest_tax_on_sale(symbol, -quantity, price, when)
                if interest:
                    self.interest_taxes_paid += self.to_tax(interest, when)
                    charge = TaxCharge(charge.amount, charge.rule_id, charge.rate, charge.capped,
                                       charge.portfolio_amount, interest)
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

    def distribution_keep(self, symbol: str) -> float:
        """Part d'une distribution (dividende, coupon) que l'investisseur garde
        et peut réinvestir, après les impôts sur les revenus.

        - distribuant : retenue du fonds, retenue du pays de domicile, puis
          précompte local ;
        - capitalisant : pas de précompte ; seule la retenue du fonds reste,
          plus la taxe sur la composante intérêts (ex. Reynders) si le fonds
          dépasse le seuil de créances — sauf si le profil la prélève à la
          revente (`at_sale`) : elle est alors comptée à chaque vente.

        L'exonération annuelle de dividendes (petits montants) est ignorée.
        """
        inst = self.instrument(symbol)
        inc = self.profile.income
        keep = 1.0 - inst.fund_withholding
        if inst.distribution == "accumulating":
            if inst.bond_share > inc.interest_component_threshold and not inc.interest_component_at_sale:
                keep *= 1.0 - inc.interest_component_rate * min(inst.bond_share, 1.0)
            return keep
        if inst.domicile and inst.domicile != self.profile.country:
            keep *= 1.0 - inc.foreign_withholding.get(inst.domicile, inc.foreign_withholding.get("default", 0.0))
        return keep * (1.0 - inc.dividend_rate)

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
        extra_gain = interest_total = 0.0
        raw_prices = dict(prices)
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
                if raw_prices.get(sym):
                    interest_total += self.interest_tax_on_sale(sym, qty, raw_prices[sym], when)
                covered, basis = self.gains.preview_sale(sym, qty)
                extra_gain += covered * prices[sym] - basis
        if self.gains is not None and extra_gain:
            eff = self.profile.capital_gains.effective_from
            if eff is None or (when.date() if isinstance(when, datetime) else when) >= eff:
                cg_total = self.gains.tax_due(year, extra_gain) - self.gains.tax_due(year)
        back = lambda v: self.to_portfolio(v, when)   # noqa: E731
        return RebalanceTaxCost(back(tx_total), back(cg_total), {s: back(v) for s, v in per_symbol.items()},
                                interest_total)

    # ------------------------------------------------------------- impôt annuel, liquidation, récolte

    def gains_tax_due(self, year: int) -> float:
        """Taxe sur les plus-values de l'année (devise du profil), 0 hors régime."""
        cg = self.profile.capital_gains
        if self.gains is None or (cg.effective_from is not None and year < cg.effective_from.year):
            return 0.0
        return self.gains.tax_due(year)

    def liquidation_cost(self, positions: Mapping[str, tuple[float, float]], when: date | datetime) -> float:
        """Impôts si tout était vendu maintenant (devise du portefeuille) : taxe
        sur transaction, Reynders et supplément de taxe sur les plus-values de
        l'année. Les positions sont {symbole: (quantité, prix)}."""
        longs = {s: (q, p) for s, (q, p) in positions.items() if q > 1e-12 and p > 0}
        cost = self.rebalance_cost({s: -q * p for s, (q, p) in longs.items()},
                                   {s: p for s, (q, p) in longs.items()}, when)
        return cost.total

    def harvest_plan(self, positions: Mapping[str, tuple[float, float]], when: date | datetime, *,
                     extra_cost_rate: float = 0.0, min_benefit: float = 2.0) -> list[tuple[str, float, float]]:
        """Récolte de l'exonération annuelle : quantités à vendre puis racheter
        aussitôt pour remonter la base fiscale sans payer d'impôt.

        Gains réalisés jusqu'à l'exonération restante de l'année, sur les lignes
        sans Reynders, les plus gros gains par euro vendu d'abord ; une ligne
        n'est récoltée que si l'impôt futur évité (taux × gain) vaut au moins
        `min_benefit` fois son coût (taxe sur transaction aller-retour +
        `extra_cost_rate` × 2 pour spread et slippage). Retourne
        [(symbole, quantité, gain récolté en devise du profil)]."""
        cg = self.profile.capital_gains
        if self.gains is None or cg is None:
            return []
        day = _as_date(when)
        if cg.effective_from is not None and day < cg.effective_from:
            return []
        remaining = self.gains.exemption(day.year) - self.gains.net_gain(day.year)
        if remaining <= 1.0:
            return []
        candidates = []
        for sym, (qty, price) in positions.items():
            if qty <= 1e-12 or price <= 0 or self.interest_tax_applies(sym):
                continue
            price_tax = self.to_tax(price, day)
            covered, basis = self.gains.preview_sale(sym, qty)
            gain = covered * price_tax - basis
            if gain > 0 and covered > 0:
                candidates.append((gain / (covered * price_tax), sym, price, price_tax))
        plan = []
        for _, sym, price, price_tax in sorted(candidates, reverse=True):
            if remaining <= 1.0:
                break
            take_qty = gain_taken = 0.0
            for take, lot in self.gains.preview_lots(sym, positions[sym][0]):
                lot_gain = take * (price_tax - self.gains.tax_basis(sym, lot))
                if lot_gain <= 0:
                    continue                       # perte : pas de récolte (vendre ne remonte rien)
                if gain_taken + lot_gain > remaining:
                    frac = (remaining - gain_taken) / lot_gain
                    take_qty += take * frac
                    gain_taken = remaining
                    break
                take_qty += take
                gain_taken += lot_gain
            if take_qty <= 1e-9 or gain_taken <= 0:
                continue
            notional = take_qty * price_tax
            tob = self.transaction_tax(sym, notional, "sell").amount + self.transaction_tax(sym, notional, "buy").amount
            cost = tob + 2.0 * extra_cost_rate * notional
            if cg.rate * gain_taken < min_benefit * cost:
                continue
            plan.append((sym, take_qty, gain_taken))
            remaining -= gain_taken
        return plan

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
            self.gains.reindex()
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
        if self.needs_fx and self.fx is None and self.fx_missing:
            out.append(f"pas de taux {self.profile.currency}/{self.portfolio_currency} : montants non convertis")
        if self.fx is not None and self.fx.before_first:
            out.append(f"{self.fx.before_first} conversion(s) avant le premier taux connu ({self.fx.first})")
        if self._unknown:
            out.append(f"instruments sans classification fiscale : {', '.join(sorted(self._unknown))}")
        return out
