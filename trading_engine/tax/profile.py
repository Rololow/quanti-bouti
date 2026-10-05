"""Profils fiscaux par pays, décrits en TOML (config/taxes/<PAYS>.toml).

Le code est générique : toutes les règles (taux, plafonds, exonérations,
dates) viennent du fichier. Ajouter un pays = écrire un nouveau TOML.

Sections reconnues (toutes optionnelles sauf [meta]) :

    [meta]              pays, devise, sources, date de vérification
    [regions]           groupes de pays (ex. EEA) utilisables dans les règles
    [transaction_tax]   taxe sur les transactions ; [[transaction_tax.rules]]
                        évaluées dans l'ordre, première correspondance
    [income_tax]        dividendes / intérêts, retenues étrangères
    [capital_gains]     plus-values réalisées
    [account_tax]       taxe annuelle sur la valeur d'un compte
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class Instrument:
    """Caractéristiques fiscales d'un instrument."""

    symbol: str
    asset_class: str = "stock"          # stock | etf | fund | bond | ...
    domicile: str | None = None         # code pays ISO-2
    distribution: str | None = None     # distributing | accumulating
    registered_locally: bool = False    # offert publiquement dans le pays du profil
    bond_share: float = 0.0             # part investie en créances (fonds)
    # Retenue subie par le fonds lui-même sur ses revenus (ex. ETF irlandais
    # détenant des actions US : 15 %), avant toute fiscalité de l'investisseur.
    fund_withholding: float = 0.0


@dataclass(frozen=True)
class TransactionTaxRule:
    id: str
    rate: float
    cap: float | None
    match: Mapping[str, Any]
    description: str = ""

    def matches(self, instrument: Instrument, regions: Mapping[str, frozenset[str]]) -> bool:
        for key, expected in self.match.items():
            allowed = expected if isinstance(expected, list) else [expected]
            if key == "domicile_region":
                if not any(instrument.domicile in regions.get(r, ()) for r in allowed):
                    return False
                continue
            if not hasattr(instrument, key):
                raise ValueError(f"rule {self.id!r}: unknown match key {key!r}")
            if getattr(instrument, key) not in allowed:
                return False
        return True


@dataclass(frozen=True)
class CapitalGainsRules:
    rate: float
    effective_from: date | None = None
    annual_exemption: float = 0.0
    carry_forward_per_year: float = 0.0
    carry_forward_max_years: int = 0
    loss_offset: str = "same_year"
    cost_basis_method: str = "fifo"
    step_up_date: date | None = None
    step_up_keeps_higher_cost: bool = True
    speculative_rate: float | None = None
    speculative_warning: str = ""


@dataclass(frozen=True)
class IncomeTaxRules:
    dividend_rate: float = 0.0
    interest_rate: float = 0.0
    dividend_exemption: float = 0.0
    foreign_withholding: Mapping[str, float] = field(default_factory=dict)
    interest_component_rate: float = 0.0
    interest_component_threshold: float = 1.0
    # Taxe due à la revente (Reynders) plutôt que chaque année : l'impôt
    # sur les intérêts capitalisés est reporté jusqu'à la vente.
    interest_component_at_sale: bool = False


@dataclass(frozen=True)
class AccountTaxRules:
    rate: float
    threshold: float


@dataclass(frozen=True)
class TaxProfile:
    country: str
    name: str
    currency: str
    verified_on: str
    sources: tuple[str, ...]
    regions: Mapping[str, frozenset[str]]
    transaction_tax_name: str
    transaction_tax_sides: frozenset[str]
    transaction_rules: tuple[TransactionTaxRule, ...]
    income: IncomeTaxRules
    capital_gains: CapitalGainsRules | None
    account_tax: AccountTaxRules | None

    @property
    def verified(self) -> bool:
        return bool(self.verified_on)

    def transaction_rule(self, instrument: Instrument) -> TransactionTaxRule | None:
        for rule in self.transaction_rules:
            if rule.matches(instrument, self.regions):
                return rule
        return None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TaxProfile":
        meta = raw.get("meta")
        if not meta or "country" not in meta:
            raise ValueError("tax profile requires [meta] with a country")

        tx = raw.get("transaction_tax") or {}
        rules = []
        for r in tx.get("rules", []):
            rate = float(r["rate"])
            if not 0 <= rate < 1:
                raise ValueError(f"rule {r.get('id')!r}: rate must be a fraction in [0, 1)")
            rules.append(TransactionTaxRule(
                id=r["id"], rate=rate, cap=r.get("cap"), match=r.get("match", {}),
                description=r.get("description", ""),
            ))

        inc = raw.get("income_tax") or {}
        income = IncomeTaxRules(
            dividend_rate=float(inc.get("dividend_rate", 0.0)),
            interest_rate=float(inc.get("interest_rate", 0.0)),
            dividend_exemption=float((inc.get("dividend_exemption") or {}).get("amount", 0.0)),
            foreign_withholding=dict(inc.get("foreign_withholding") or {}),
            interest_component_rate=float((inc.get("interest_component") or {}).get("rate", 0.0)),
            interest_component_threshold=float(
                (inc.get("interest_component") or {}).get("bond_share_threshold", 1.0)
            ),
            interest_component_at_sale=bool((inc.get("interest_component") or {}).get("at_sale", False)),
        )

        cg = raw.get("capital_gains")
        capital_gains = None
        if cg:
            carry = cg.get("exemption_carry_forward") or {}
            spec = cg.get("speculative") or {}
            capital_gains = CapitalGainsRules(
                rate=float(cg["rate"]),
                effective_from=cg.get("effective_from"),
                annual_exemption=float(cg.get("annual_exemption", 0.0)),
                carry_forward_per_year=float(carry.get("per_year", 0.0)),
                carry_forward_max_years=int(carry.get("max_years", 0)),
                loss_offset=cg.get("loss_offset", "same_year"),
                cost_basis_method=cg.get("cost_basis_method", "fifo"),
                step_up_date=cg.get("step_up_date"),
                step_up_keeps_higher_cost=bool(cg.get("step_up_keeps_higher_cost", True)),
                speculative_rate=spec.get("rate"),
                speculative_warning=spec.get("warning", ""),
            )
            if capital_gains.cost_basis_method != "fifo":
                raise ValueError("only the 'fifo' cost basis method is implemented")

        acc = raw.get("account_tax")
        account_tax = AccountTaxRules(float(acc["rate"]), float(acc["threshold"])) if acc else None

        return cls(
            country=meta["country"],
            name=meta.get("name", meta["country"]),
            currency=meta.get("currency", "EUR"),
            verified_on=str(meta.get("verified_on", "")),
            sources=tuple(meta.get("sources", ())),
            regions={k: frozenset(v) for k, v in (raw.get("regions") or {}).items()},
            transaction_tax_name=tx.get("name", "transaction tax"),
            transaction_tax_sides=frozenset(tx.get("sides", ("buy", "sell"))),
            transaction_rules=tuple(rules),
            income=income,
            capital_gains=capital_gains,
            account_tax=account_tax,
        )


def load_tax_profile(path: str | Path) -> TaxProfile:
    with open(path, "rb") as fh:
        return TaxProfile.from_dict(tomllib.load(fh))
