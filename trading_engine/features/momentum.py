"""Momentum multi-horizon normalisé par la volatilité (README §11).

    M_{i,h} = R_{i,h} / sigma_{i,h}

avec R_{i,h} le log-rendement sur l'horizon h et sigma_{i,h} la volatilité
EWMA par barre remise à l'échelle de l'horizon (sigma_bar * sqrt(n)).
"""

from __future__ import annotations

import math
from typing import Iterable

from trading_engine.features.returns import PriceHistory
from trading_engine.timeutils import parse_timeframe


def resolve_horizon(horizon: str, timeframes: Iterable[str]) -> tuple[str, int]:
    """Choisit la barre la plus longue qui divise l'horizon.

    '30m' avec (5m, 1h, 1d) -> ('5m', 6) ; '20d' -> ('1d', 20) ; '1h' -> ('1h', 1).
    """
    target = parse_timeframe(horizon)
    candidates = []
    for tf in timeframes:
        step = parse_timeframe(tf)
        if step <= target and not target % step:
            candidates.append((step, tf))
    if not candidates:
        raise ValueError(f"no bar timeframe divides horizon {horizon!r}")
    step, tf = max(candidates)
    return tf, target // step


def momentum(history: PriceHistory, vol_per_bar: float | None, n: int) -> float | None:
    ret = history.log_return(n)
    if ret is None or not vol_per_bar or vol_per_bar <= 0:
        return None
    return ret / (vol_per_bar * math.sqrt(n))
