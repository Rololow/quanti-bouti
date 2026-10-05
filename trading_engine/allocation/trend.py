"""Filtre de tendance (time-series momentum) appliqué à l'allocation par le risque.

Pour chaque actif, la part des horizons (1, 3, 6, 12 mois par défaut) sur
lesquels son rendement dépasse celui du cash donne un score entre 0 et 1 :
le poids issu de la parité de risque est multiplié par
`floor + (1 - floor) × score`. Un actif en baisse sur tous les horizons sort
(ou garde `floor`), ce qui retire du portefeuille, pendant les marchés
baissiers prolongés (2008, 2022 pour les obligations), le risque que la
parité de risque y laisse — et qui rend le levier dangereux.

Références : Moskowitz, Ooi & Pedersen (2012) « Time series momentum » ;
Hurst, Ooi & Pedersen (2017) « A century of evidence on trend-following
investing » ; Faber (2007) « A quantitative approach to tactical asset
allocation ». Le moteur l'utilise comme filtre de risque, pas comme alpha :
aucun actif n'est vendu à découvert.

Les prix sont des clôtures quotidiennes (barres `timeframe`, dividendes inclus
dans le dataset de backtest) ; il faut `max(horizons) + 1` barres avant le
premier score, sinon l'actif garde son poids (score 1).
"""

from __future__ import annotations

import math
from collections import deque
from typing import Mapping, Sequence


class TrendFilter:
    """`update_every` : le score d'un actif n'est recalculé que toutes les N
    clôtures (5 = chaque semaine) ; `min_change` : un nouveau score ne remplace
    l'ancien que s'il s'en écarte d'au moins cette valeur, ou s'il est à 0 ou 1
    (tous les horizons d'accord). Les deux limitent les allers-retours, donc la
    TOB, quand un horizon court change de signe d'un jour à l'autre."""

    def __init__(self, horizons: Sequence[int] = (21, 63, 126, 252), floor: float = 0.0, *,
                 update_every: int = 1, min_change: float = 0.0) -> None:
        if not horizons or any(h < 1 for h in horizons):
            raise ValueError("trend horizons must be >= 1 bar")
        if not 0.0 <= floor <= 1.0:
            raise ValueError("trend floor must be in [0, 1]")
        if update_every < 1 or not 0.0 <= min_change <= 1.0:
            raise ValueError("trend update_every must be >= 1 and min_change in [0, 1]")
        self.horizons = tuple(sorted(set(int(h) for h in horizons)))
        self.floor = floor
        self.update_every = update_every
        self.min_change = min_change
        self._closes: dict[str, deque[float]] = {}
        self._held: dict[str, float] = {}          # score retenu par actif
        self._since: dict[str, int] = {}           # clôtures depuis la dernière mise à jour

    @property
    def ready_after(self) -> int:
        return self.horizons[-1] + 1

    def on_close(self, symbol: str, price: float) -> None:
        if price > 0 and math.isfinite(price):
            self._closes.setdefault(symbol, deque(maxlen=self.ready_after)).append(price)
            self._since[symbol] = self._since.get(symbol, 0) + 1

    def score(self, symbol: str, cash_rate: float = 0.0) -> float | None:
        """Part des horizons en tendance haussière (rendement > cash), ou None."""
        closes = self._closes.get(symbol)
        if closes is None or len(closes) < self.ready_after:
            return None
        last = closes[-1]
        up = 0
        for h in self.horizons:
            hurdle = cash_rate * h / 252.0
            if math.log(last / closes[-1 - h]) > hurdle:
                up += 1
        return up / len(self.horizons)

    def held_score(self, symbol: str, cash_rate: float = 0.0) -> float | None:
        """Score retenu, mis à jour selon `update_every` et `min_change`."""
        held = self._held.get(symbol)
        if held is not None and self._since.get(symbol, 0) < self.update_every:
            return held
        new = self.score(symbol, cash_rate)
        if new is None:
            return held
        self._since[symbol] = 0
        if held is None or new in (0.0, 1.0) or abs(new - held) >= self.min_change - 1e-12:
            self._held[symbol] = new
        return self._held[symbol]

    def multipliers(self, symbols: Sequence[str], cash_rate: float = 0.0) -> dict[str, float]:
        out = {}
        for sym in symbols:
            s = self.held_score(sym, cash_rate)
            out[sym] = 1.0 if s is None else self.floor + (1.0 - self.floor) * s
        return out


def apply_trend(weights: Mapping[str, float], multipliers: Mapping[str, float]) -> dict[str, float]:
    return {s: w * multipliers.get(s, 1.0) for s, w in weights.items()}
