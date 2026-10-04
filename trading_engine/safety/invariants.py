"""Invariants vérifiés en continu (README §50 : défense en profondeur).

Les contrôles amont (Constraint Engine, hard controls, dimensionnement des
ordres) doivent garantir ces propriétés. Ce moniteur ne fait pas confiance à
ces garanties : il vérifie l'état réel du moteur après chaque intervalle et
chaque événement du compte broker. Une violation signifie un bug ou un état
inattendu du broker : le moteur passe en HALTED (mode `halt`), annule ses
ordres et lève une alerte critique. Il ne liquide rien.

Invariants :
- FINITE      : cash, quantités et prix finis, valeur du portefeuille > 0 ;
- CASH        : cash >= -tolérance × valeur (pas de levier involontaire) ;
- GROSS       : exposition brute <= limite des hard controls + tolérance ;
- SHORT       : aucune position courte si le portefeuille est long-only ;
- TARGET      : la cible courante passe les hard controls ;
- TAX_LOTS    : lots fiscaux = quantités détenues (positions longues), dès
                que le compte broker a été lu ;
- HANDLER     : aucun gestionnaire d'événement n'a levé d'exception (un
                état partiellement mis à jour n'est plus fiable).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:                       # pragma: no cover
    from trading_engine.execution.hard_controls import HardControls

QTY_EPS = 1e-6


@dataclass(frozen=True)
class Violation:
    name: str
    detail: str

    def __str__(self) -> str:
        return f"{self.name}: {self.detail}"


def check_portfolio(
    cash: float,
    positions: Mapping[str, tuple[float, float]],      # symbole -> (quantité, prix)
    *,
    max_gross: float,
    long_only: bool,
    cash_tolerance: float = 0.01,
    gross_tolerance: float = 0.02,
) -> list[Violation]:
    out: list[Violation] = []
    bad = [s for s, (q, p) in positions.items() if not (math.isfinite(q) and math.isfinite(p) and p > 0)]
    if not math.isfinite(cash) or bad:
        out.append(Violation("FINITE", f"cash={cash} positions invalides={sorted(bad)}"))
        return out
    value = cash + sum(q * p for q, p in positions.values())
    if not value > 0:
        return [Violation("FINITE", f"valeur du portefeuille {value:,.2f} <= 0")]
    if cash < -cash_tolerance * value:
        out.append(Violation("CASH", f"cash {cash:,.2f} < -{cash_tolerance:.0%} de {value:,.2f}"))
    gross = sum(abs(q * p) for q, p in positions.values()) / value
    if gross > max_gross + gross_tolerance:
        out.append(Violation("GROSS", f"exposition brute {gross:.3f} > {max_gross:.3f}"))
    shorts = sorted(s for s, (q, _) in positions.items() if q < -QTY_EPS)
    if long_only and shorts:
        out.append(Violation("SHORT", f"positions courtes {shorts}"))
    return out


def check_tax_lots(positions: Mapping[str, float], lots: Mapping[str, float]) -> list[Violation]:
    diffs = {}
    for sym in sorted(set(positions) | set(lots)):
        held = max(positions.get(sym, 0.0), 0.0)
        if abs(held - lots.get(sym, 0.0)) > QTY_EPS * max(1.0, held):
            diffs[sym] = (round(held, 6), round(lots.get(sym, 0.0), 6))
    return [Violation("TAX_LOTS", f"détenu vs lots {diffs}")] if diffs else []


def check_targets(weights: Mapping[str, float], hard_controls: "HardControls") -> list[Violation]:
    if not weights:
        return []
    verdict = hard_controls.validate_targets(weights)
    return [] if verdict.approved else [Violation("TARGET", "; ".join(verdict.violations))]


class InvariantMonitor:
    """Rassemble les vérifications sur l'état d'un `Engine`."""

    MODES = ("halt", "alert", "off")

    def __init__(self, mode: str = "halt", *, cash_tolerance: float = 0.01, gross_tolerance: float = 0.02) -> None:
        if mode not in self.MODES:
            raise ValueError(f"invariants mode must be one of {self.MODES}, got {mode!r}")
        self.mode = mode
        self.cash_tolerance = cash_tolerance
        self.gross_tolerance = gross_tolerance
        self.handler_errors_seen = 0
        self.checks = 0
        self.violations: list[Violation] = []

    def check(self, engine) -> list[Violation]:
        if self.mode == "off":
            return []
        self.checks += 1
        portfolio = engine.portfolio
        positions = {s: (p.quantity, portfolio.prices.get(s, p.avg_price))
                     for s, p in portfolio.positions.items() if abs(p.quantity) > QTY_EPS}
        out = check_portfolio(
            portfolio.cash, positions,
            max_gross=engine.hard_controls.limits.max_target_gross,
            long_only=engine.config.allocation.constraints.min_weight >= 0,
            cash_tolerance=self.cash_tolerance, gross_tolerance=self.gross_tolerance,
        )
        out += check_targets(portfolio.target_weights, engine.hard_controls)
        tax = engine.tax
        # Avant la première lecture du compte broker, les positions sont celles de
        # la config et le registre attend la synchronisation : rien à comparer
        # (et aucun ordre n'est envoyé dans cet état).
        if tax is not None and tax.gains is not None and engine.ledger_source is not None \
                and engine.account_synced:
            out += check_tax_lots({s: q for s, (q, _) in positions.items()},
                                  {s: tax.lot_quantity(s) for s in tax.gains.lots})
        errors = engine.bus.error_count
        if errors > self.handler_errors_seen:
            out.append(Violation("HANDLER", f"{errors - self.handler_errors_seen} exception(s) dans les gestionnaires"))
            self.handler_errors_seen = errors
        self.violations.extend(out)
        return out
