"""Suivi des plus-values réalisées par lots (FIFO) et impôt annuel estimé.

- chaque achat crée un lot (quantité, coût unitaire, date) ;
- une vente consomme les lots les plus anciens (FIFO) ;
- lots acquis avant l'entrée en vigueur de la taxe : valeur d'acquisition
  fiscale = prix de référence à la date de step-up (ou coût réel s'il est
  plus élevé, si le profil le prévoit) ;
- gains et pertes se compensent sur l'année civile ;
- exonération annuelle, avec report de l'exonération non utilisée.

Ne gère que les positions longues (les ventes à découvert ne créent pas de lot).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import date, datetime
from typing import Mapping

from trading_engine.tax.profile import CapitalGainsRules


@dataclass
class Lot:
    quantity: float
    unit_cost: float
    acquired: date


@dataclass(frozen=True)
class RealizedGain:
    symbol: str
    quantity: float
    proceeds: float
    tax_basis: float
    sold: date

    @property
    def gain(self) -> float:
        return self.proceeds - self.tax_basis


def _as_date(d: date | datetime) -> date:
    return d.date() if isinstance(d, datetime) else d


class CapitalGainsTracker:
    def __init__(self, rules: CapitalGainsRules, step_up_prices: Mapping[str, float] | None = None) -> None:
        self.rules = rules
        self.step_up_prices = dict(step_up_prices or {})
        self.lots: dict[str, deque[Lot]] = {}
        self.realized: list[RealizedGain] = []
        self._carry: dict[int, float] = {}     # exonération reportée disponible au début de l'année
        self._net_by_year: dict[int, float] = {}

    def reindex(self) -> None:
        """Recalcule le cumul annuel après un remplacement de `realized`."""
        self._net_by_year = {}
        for g in self.realized:
            self._net_by_year[g.sold.year] = self._net_by_year.get(g.sold.year, 0.0) + g.gain

    # ----------------------------------------------------------------- lots

    def buy(self, symbol: str, quantity: float, price: float, when: date | datetime) -> None:
        if quantity <= 0:
            raise ValueError("buy quantity must be positive")
        self.lots.setdefault(symbol, deque()).append(Lot(quantity, price, _as_date(when)))

    def _tax_basis(self, symbol: str, lot: Lot) -> float:
        r = self.rules
        if r.effective_from is None or lot.acquired >= r.effective_from:
            return lot.unit_cost
        reference = self.step_up_prices.get(symbol)
        if reference is None:
            return lot.unit_cost
        return max(reference, lot.unit_cost) if r.step_up_keeps_higher_cost else reference

    def _consume(self, symbol: str, quantity: float, commit: bool) -> tuple[float, float]:
        """Retourne (quantité couverte par des lots, base fiscale correspondante)."""
        lots = self.lots.get(symbol, deque())
        remaining, basis, covered = quantity, 0.0, 0.0
        for lot in list(lots):
            if remaining <= 0:
                break
            take = min(lot.quantity, remaining)
            basis += take * self._tax_basis(symbol, lot)
            covered += take
            remaining -= take
            if commit:
                lot.quantity -= take
        if commit:
            while lots and lots[0].quantity <= 1e-12:
                lots.popleft()
        return covered, basis

    def preview_lots(self, symbol: str, quantity: float) -> list[tuple[float, Lot]]:
        """Lots (quantité prise, lot) qu'une vente FIFO consommerait."""
        out, remaining = [], quantity
        for lot in self.lots.get(symbol, ()):
            if remaining <= 1e-12:
                break
            take = min(lot.quantity, remaining)
            out.append((take, lot))
            remaining -= take
        return out

    def tax_basis(self, symbol: str, lot: Lot) -> float:
        return self._tax_basis(symbol, lot)

    def remove(self, symbol: str, quantity: float) -> float:
        """Retire des lots (FIFO) sans plus-value : titres sortis hors du
        suivi (transfert, écart de synchronisation). Retourne la quantité retirée."""
        return self._consume(symbol, quantity, commit=True)[0]

    def preview_sale(self, symbol: str, quantity: float) -> tuple[float, float]:
        """(quantité couverte, base fiscale) d'une vente, sans modifier les lots."""
        return self._consume(symbol, quantity, commit=False)

    def sell(self, symbol: str, quantity: float, price: float, when: date | datetime) -> RealizedGain | None:
        if quantity <= 0:
            raise ValueError("sell quantity must be positive")
        when = _as_date(when)
        covered, basis = self._consume(symbol, quantity, commit=True)
        if covered <= 0:
            return None
        gain = RealizedGain(symbol, covered, covered * price, basis, when)
        if self.rules.effective_from is None or when >= self.rules.effective_from:
            self.realized.append(gain)
            self._net_by_year[when.year] = self._net_by_year.get(when.year, 0.0) + gain.gain
        return gain

    # ----------------------------------------------------------------- impôt

    def net_gain(self, year: int) -> float:
        return self._net_by_year.get(year, 0.0)

    def exemption(self, year: int) -> float:
        """Exonération disponible : annuelle + réserve reportée.

        Chaque année, l'exonération annuelle est utilisée en premier, puis la
        réserve. La part non utilisée de l'exonération annuelle alimente la
        réserve (au plus `per_year`), plafonnée à `per_year * max_years`.
        """
        r = self.rules
        if r.carry_forward_per_year <= 0 or r.effective_from is None:
            return r.annual_exemption
        pool = 0.0
        cap = r.carry_forward_per_year * r.carry_forward_max_years
        for y in range(r.effective_from.year, year):
            used = min(max(self.net_gain(y), 0.0), r.annual_exemption + pool)
            pool -= max(0.0, used - r.annual_exemption)
            pool = min(cap, pool + min(max(0.0, r.annual_exemption - used), r.carry_forward_per_year))
        return r.annual_exemption + pool

    def tax_due(self, year: int, extra_gain: float = 0.0) -> float:
        taxable = max(0.0, self.net_gain(year) + extra_gain - self.exemption(year))
        return taxable * self.rules.rate

    def marginal_tax_of_sale(self, symbol: str, quantity: float, price: float, when: date | datetime) -> float:
        """Impôt supplémentaire de l'année si l'on vendait maintenant (sans rien modifier)."""
        when = _as_date(when)
        if quantity <= 0 or (self.rules.effective_from and when < self.rules.effective_from):
            return 0.0
        covered, basis = self.preview_sale(symbol, quantity)
        gain = covered * price - basis
        return self.tax_due(when.year, gain) - self.tax_due(when.year)
