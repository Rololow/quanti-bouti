"""Stratégies de référence, simulées sur les mêmes prix que le moteur.

- buy & hold équipondéré : achat à la première clôture, plus rien ensuite ;
- risk parity mensuelle : poids ∝ 1/volatilité (60 séances), rebalancement
  à la première séance de chaque mois.

Simplifications (en faveur des références, donc prudentes pour le moteur) :
exécution à la clôture du jour, fractions d'actions autorisées. Coûts : demi-
spread + slippage du modèle de coûts, et la même taxe sur transaction que le
moteur (TOB en EUR au cours du jour, plafonds compris).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING, Callable, Mapping, Sequence

from trading_engine.data.calendar import NY

if TYPE_CHECKING:                           # pragma: no cover
    from trading_engine.tax.tax_model import TaxModel

TaxFn = Callable[[str, float, str, date], tuple[float, float]]   # -> (devise portefeuille, devise fiscale)


@dataclass
class StrategyResult:
    name: str
    equity: list[tuple[date, float]]
    trades: list[tuple[date, float]] = field(default_factory=list)       # notionnels échangés
    taxes: list[tuple[date, float]] = field(default_factory=list)        # taxe sur transaction (devise fiscale)
    params: dict = field(default_factory=dict)
    kind: str = "benchmark"
    stats: dict = field(default_factory=dict)                            # infos propres au moteur


def daily_closes(closes: Mapping[str, Sequence[tuple[datetime, float]]]) -> tuple[list[date], dict[str, dict[date, float]]]:
    """Dernière clôture de chaque séance (date New York) par symbole."""
    by_symbol: dict[str, dict[date, float]] = {}
    for sym, rows in closes.items():
        out = by_symbol.setdefault(sym, {})
        for end, close in sorted(rows):
            out[end.astimezone(NY).date()] = close
    days = sorted({d for rows in by_symbol.values() for d in rows})
    return days, by_symbol


def simulate(
    name: str,
    days: Sequence[date],
    prices: Mapping[str, Mapping[date, float]],
    weights_on: Callable[[int], Mapping[str, float] | None],
    *,
    cash: float,
    cost_bps: float,
    tax: TaxFn | None = None,
    params: dict | None = None,
    cash_rate: Callable[[date], float] | None = None,
    tax_model: "TaxModel | None" = None,
    harvest: Mapping | None = None,
) -> StrategyResult:
    """`weights_on(i)` : poids cibles à la clôture du jour i (None = pas de rebalancement).
    `cash_rate(jour)` : taux court annuel rémunérant le cash (comme le moteur).

    `tax_model` (neuf, propre à cette stratégie) : mêmes impôts que le moteur —
    taxe sur transaction, Reynders à la vente, taxe annuelle sur les
    plus-values au changement d'année — et coût de liquidation final dans
    `stats`. `harvest` ({"from": (mois, jour), "min_benefit": x}) : même
    récolte de l'exonération que le moteur."""
    qty: dict[str, float] = {s: 0.0 for s in prices}
    last: dict[str, float] = {}
    result = StrategyResult(name, [], params=dict(params or {}))
    tax_year = None
    stats = {"gains_tax_paid": 0.0, "interest_tax_paid": 0.0, "harvested_gains": 0.0}
    harvested_year = None
    for i, day in enumerate(days):
        if cash_rate is not None and i > 0 and cash > 0:
            cash += cash * cash_rate(days[i - 1]) * (day - days[i - 1]).days / 365.0
        if tax_model is not None and tax_model.gains is not None:
            if tax_year is None:
                tax_year = day.year
            while tax_year < day.year:
                due = tax_model.to_portfolio(tax_model.gains_tax_due(tax_year), day)
                cash -= due
                stats["gains_tax_paid"] += due
                tax_year += 1
        for s in prices:
            if day in prices[s]:
                last[s] = prices[s][day]
        value = cash + sum(qty[s] * last[s] for s in qty if s in last)
        target = weights_on(i)
        if target is not None and value > 0:
            for s in sorted(target):              # exécution simultanée à la clôture
                if s not in last:
                    continue
                delta = target[s] * value - qty[s] * last[s]
                if abs(delta) < 1e-6:
                    continue
                cost = abs(delta) * cost_bps * 1e-4
                if tax_model is not None:
                    charge, _ = tax_model.record_fill(s, delta / last[s], last[s], day)
                    cost += (charge.portfolio_amount or 0.0) + charge.interest_tax
                    stats["interest_tax_paid"] += charge.interest_tax
                    result.taxes.append((day, charge.amount))
                elif tax is not None:
                    tax_p, tax_t = tax(s, abs(delta), "buy" if delta > 0 else "sell", day)
                    cost += tax_p
                    result.taxes.append((day, tax_t))
                qty[s] += delta / last[s]
                cash -= delta + cost
                result.trades.append((day, abs(delta)))
            value = cash + sum(qty[s] * last[s] for s in qty if s in last)
        if (harvest and tax_model is not None and harvested_year != day.year
                and (day.month, day.day) >= tuple(harvest["from"])):
            harvested_year = day.year
            positions = {s: (qty[s], last[s]) for s in qty if s in last and qty[s] > 0}
            rate = cost_bps * 1e-4
            for s, q, gain in tax_model.harvest_plan(positions, day, extra_cost_rate=rate,
                                                     min_benefit=harvest.get("min_benefit", 2.0)):
                for signed in (-q, q):
                    charge, _ = tax_model.record_fill(s, signed, last[s], day)
                    cash -= (charge.portfolio_amount or 0.0) + charge.interest_tax + q * last[s] * rate
                    result.taxes.append((day, charge.amount))
                stats["harvested_gains"] += gain
            value = cash + sum(qty[s] * last[s] for s in qty if s in last)
        result.equity.append((day, value))
    if tax_model is not None and days:
        positions = {s: (qty[s], last[s]) for s in qty if s in last and qty[s] > 0}
        stats["liquidation_cost"] = (tax_model.liquidation_cost(positions, days[-1])
                                     + sum(q * p for q, p in positions.values()) * cost_bps * 1e-4)
        result.stats = stats
    return result


def buy_and_hold(days, prices, *, cash: float, cost_bps: float, tax: TaxFn | None = None,
                 min_cash: float = 0.02, cash_rate=None, **tax_kw) -> StrategyResult:
    symbols = sorted(prices)
    w = (1.0 - min_cash) / len(symbols)
    return simulate("Buy & hold équipondéré", days, prices,
                    lambda i: {s: w for s in symbols} if i == 0 else None,
                    cash=cash, cost_bps=cost_bps, tax=tax, cash_rate=cash_rate, **tax_kw)


def monthly_inverse_vol(days, prices, *, cash: float, cost_bps: float, tax: TaxFn | None = None,
                        lookback: int = 60, min_obs: int = 20, min_cash: float = 0.02,
                        cash_rate=None, **tax_kw) -> StrategyResult:
    symbols = sorted(prices)

    def vol(sym: str, i: int) -> float | None:
        series = [prices[sym][d] for d in days[max(0, i - lookback):i + 1] if d in prices[sym]]
        r = [math.log(b / a) for a, b in zip(series, series[1:]) if a > 0 and b > 0]
        if len(r) < min_obs:
            return None
        m = sum(r) / len(r)
        sd = math.sqrt(sum((x - m) ** 2 for x in r) / (len(r) - 1))
        return sd if sd > 0 else None

    def weights_on(i: int):
        if i > 0 and days[i].month == days[i - 1].month:
            return None
        vols = {s: vol(s, i) for s in symbols}
        if any(v is None for v in vols.values()):
            inv = {s: 1.0 for s in symbols}           # pas assez d'historique : équipondéré
        else:
            inv = {s: 1.0 / v for s, v in vols.items()}
        total = sum(inv.values())
        return {s: (1.0 - min_cash) * x / total for s, x in inv.items()}

    return simulate("Risk parity mensuelle (1/vol)", days, prices, weights_on,
                    cash=cash, cost_bps=cost_bps, tax=tax, params={"lookback": lookback}, cash_rate=cash_rate,
                    **tax_kw)


def monthly_fixed(name: str, days, prices, weights: Mapping[str, float], *, cash: float, cost_bps: float,
                  tax: TaxFn | None = None, min_cash: float = 0.02, cash_rate=None,
                  **tax_kw) -> StrategyResult | None:
    """Poids fixes rebalancés chaque mois (ex. 60/40 actions/obligations)."""
    if not all(s in prices for s in weights):
        return None
    target = {s: (1.0 - min_cash) * w for s, w in weights.items()}
    return simulate(name, days, prices,
                    lambda i: target if i == 0 or days[i].month != days[i - 1].month else None,
                    cash=cash, cost_bps=cost_bps, tax=tax, params=dict(weights), cash_rate=cash_rate, **tax_kw)
