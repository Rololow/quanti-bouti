"""Drift monitoring (README §17, §49.6).

- drift       : poids courant - poids cible, par position ;
- turnover    : somme |drift| = ce qu'il faudrait échanger pour rejoindre la cible ;
- target_change : combien la cible a bougé depuis la précédente — une cible
                  qui oscille (15 % -> 14 % -> 16 %) signale un signal instable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class DriftReport:
    drifts: dict[str, float]
    max_abs_drift: float
    turnover_to_target: float
    target_change: float | None
    over_threshold: tuple[str, ...]


class DriftMonitor:
    def __init__(self, threshold: float = 0.05) -> None:
        self.threshold = threshold
        self._previous_target: dict[str, float] | None = None
        self.last_target_change: float | None = None

    def on_new_target(self, target: Mapping[str, float]) -> float | None:
        if self._previous_target is not None:
            keys = set(target) | set(self._previous_target)
            self.last_target_change = sum(
                abs(target.get(k, 0.0) - self._previous_target.get(k, 0.0)) for k in keys
            )
        self._previous_target = dict(target)
        return self.last_target_change

    def evaluate(self, current: Mapping[str, float], target: Mapping[str, float]) -> DriftReport:
        keys = sorted(set(current) | set(target))
        drifts = {k: current.get(k, 0.0) - target.get(k, 0.0) for k in keys}
        return DriftReport(
            drifts=drifts,
            max_abs_drift=max((abs(v) for v in drifts.values()), default=0.0),
            turnover_to_target=sum(abs(v) for v in drifts.values()),
            target_change=self.last_target_change,
            over_threshold=tuple(k for k, v in drifts.items() if abs(v) > self.threshold),
        )
