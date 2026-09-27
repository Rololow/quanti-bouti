"""Fondamentaux point-in-time et score qualitatif (README §19-21, §27, §44).

Anti-look-ahead : chaque fait est stocké avec `available_at` ; une lecture
« as of t » ne voit que les faits publiés à t ou avant, même si la base
contient déjà la valeur (reçue plus tôt, ou révisée plus tard).

Features dérivées (préfixe fund_) :

    fund_eps_surprise      (EPS réalisé - consensus) / |consensus|
    fund_eps_growth        EPS vs même trimestre de l'année précédente (sinon période précédente)
    fund_revenue_growth    idem pour le chiffre d'affaires
    fund_guidance_change   variation relative de la dernière guidance EPS
    fund_margin_change     variation de la marge opérationnelle (points)
    fund_days_since_earnings

Score qualitatif (README §27) : Q = Σ w_k · tanh(f_k / échelle_k), atténué
avec l'ancienneté de l'information (demi-vie). Les poids sont fixes ici ;
l'apprentissage de leur pouvoir prédictif passe par les prédicteurs online,
qui peuvent utiliser les fund_* comme features.
"""

from __future__ import annotations

import bisect
import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping

from trading_engine.data.events import FundamentalEvent

_PERIOD = re.compile(r"^(\d{4})(Q[1-4]|FY)$")


@dataclass(frozen=True)
class Fact:
    available_at: datetime
    period: str
    value: float
    estimate: float | None = None


def previous_year_period(period: str) -> str | None:
    m = _PERIOD.match(period)
    return None if m is None else f"{int(m.group(1)) - 1}{m.group(2)}"


class FundamentalStore:
    def __init__(self) -> None:
        self._facts: dict[tuple[str, str], list[Fact]] = {}
        self.count = 0

    def add(self, ev: FundamentalEvent) -> None:
        facts = self._facts.setdefault((ev.symbol, ev.name), [])
        fact = Fact(ev.available_at, ev.period, ev.value, ev.estimate)
        bisect.insort(facts, fact, key=lambda f: (f.available_at, f.period))
        self.count += 1

    def history(self, symbol: str, name: str, as_of: datetime) -> list[Fact]:
        """Faits publiés au plus tard à `as_of`, dans l'ordre de publication."""
        facts = self._facts.get((symbol, name), [])
        i = bisect.bisect_right([f.available_at for f in facts], as_of)
        return facts[:i]

    def latest(self, symbol: str, name: str, as_of: datetime) -> Fact | None:
        hist = self.history(symbol, name, as_of)
        return hist[-1] if hist else None

    def value_for_period(self, symbol: str, name: str, period: str, as_of: datetime) -> float | None:
        """Dernière valeur connue à `as_of` pour une période (les révisions postérieures sont ignorées)."""
        for fact in reversed(self.history(symbol, name, as_of)):
            if fact.period == period:
                return fact.value
        return None

    def symbols(self) -> list[str]:
        return sorted({sym for sym, _ in self._facts})


DEFAULT_WEIGHTS = {
    "fund_eps_surprise": 0.4,
    "fund_eps_growth": 0.2,
    "fund_revenue_growth": 0.2,
    "fund_guidance_change": 0.2,
}
DEFAULT_SCALES = {
    "fund_eps_surprise": 0.10,
    "fund_eps_growth": 0.20,
    "fund_revenue_growth": 0.10,
    "fund_guidance_change": 0.05,
    "fund_margin_change": 0.02,
}


class FundamentalFeatures:
    def __init__(
        self,
        store: FundamentalStore | None = None,
        *,
        weights: Mapping[str, float] | None = None,
        scales: Mapping[str, float] | None = None,
        half_life_days: float = 30.0,
    ) -> None:
        self.store = store or FundamentalStore()
        self.weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
        self.scales = {**DEFAULT_SCALES, **(scales or {})}
        self.half_life_days = half_life_days

    def _growth(self, symbol: str, name: str, as_of: datetime) -> float | None:
        latest = self.store.latest(symbol, name, as_of)
        if latest is None:
            return None
        base = None
        prior = previous_year_period(latest.period)
        if prior is not None:
            base = self.store.value_for_period(symbol, name, prior, as_of)
        if base is None:
            earlier = [f for f in self.store.history(symbol, name, as_of) if f.period != latest.period]
            base = earlier[-1].value if earlier else None
        if base is None or base == 0:
            return None
        return (latest.value - base) / abs(base)

    def snapshot(self, symbol: str, as_of: datetime | None) -> dict[str, float]:
        if as_of is None:
            return {}
        out: dict[str, float] = {}
        eps = self.store.latest(symbol, "eps", as_of)
        if eps is not None:
            if eps.estimate:
                out["fund_eps_surprise"] = (eps.value - eps.estimate) / abs(eps.estimate)
            out["fund_days_since_earnings"] = (as_of - eps.available_at).total_seconds() / 86400
        for name in ("eps", "revenue"):
            g = self._growth(symbol, name, as_of)
            if g is not None:
                out[f"fund_{name}_growth"] = g
        guidance = self.store.history(symbol, "guidance_eps", as_of)
        if len(guidance) >= 2 and guidance[-2].value:
            out["fund_guidance_change"] = (guidance[-1].value - guidance[-2].value) / abs(guidance[-2].value)
        rev = self.store.history(symbol, "revenue", as_of)
        op = self.store.history(symbol, "operating_income", as_of)
        if len(rev) >= 2 and len(op) >= 2 and rev[-1].value and rev[-2].value:
            out["fund_margin_change"] = op[-1].value / rev[-1].value - op[-2].value / rev[-2].value

        score_parts = [
            w * math.tanh(out[k] / self.scales.get(k, 1.0)) for k, w in self.weights.items() if k in out
        ]
        if score_parts:
            age = out.get("fund_days_since_earnings", 0.0)
            decay = 0.5 ** (age / self.half_life_days) if self.half_life_days > 0 else 1.0
            out["qual_score"] = sum(score_parts) * decay
        return out

    def latest_fact(self, symbol: str, name: str, as_of: datetime) -> Fact | None:
        return self.store.latest(symbol, name, as_of)
