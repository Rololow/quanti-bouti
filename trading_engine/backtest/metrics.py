"""Métriques de performance sur une série de valeurs de fin de journée.

Rendements quotidiens simples ; taux sans risque nul (Sharpe et Sortino sont
donc des ratios rendement / risque bruts) ; 252 séances par an. Les variantes
EUR convertissent chaque valeur au cours BCE du jour (point de vue d'un
investisseur belge : l'effet de change fait partie du résultat).
"""

from __future__ import annotations

import math
from datetime import date
from typing import Iterable, Sequence

from trading_engine.tax.fx import FxRates

DAYS_PER_YEAR = 252


def window(series: Sequence[tuple[date, float]], start: date | None, end: date | None) -> list[tuple[date, float]]:
    return [(d, v) for d, v in series if (start is None or d >= start) and (end is None or d <= end)]


def daily_returns(series: Sequence[tuple[date, float]]) -> list[float]:
    return [v / p - 1.0 for (_, p), (_, v) in zip(series, series[1:]) if p > 0]


def max_drawdown(values: Iterable[float]) -> float:
    peak, worst = -math.inf, 0.0
    for v in values:
        peak = max(peak, v)
        if peak > 0:
            worst = min(worst, v / peak - 1.0)
    return worst


def _stats(series: Sequence[tuple[date, float]]) -> dict:
    if len(series) < 2 or series[0][1] <= 0:
        return {"total_return": None, "cagr": None, "vol": None, "sharpe": None, "sortino": None,
                "max_drawdown": None, "calmar": None}
    r = daily_returns(series)
    n = len(r)
    mean = sum(r) / n
    var = sum((x - mean) ** 2 for x in r) / (n - 1) if n > 1 else 0.0
    std = math.sqrt(var)
    downside = math.sqrt(sum(min(x, 0.0) ** 2 for x in r) / n)
    total = series[-1][1] / series[0][1] - 1.0
    years = max((series[-1][0] - series[0][0]).days / 365.25, 1e-9)
    cagr = (1.0 + total) ** (1.0 / years) - 1.0 if total > -1 else -1.0
    mdd = max_drawdown(v for _, v in series)
    return {
        "total_return": total,
        "cagr": cagr,
        "vol": std * math.sqrt(DAYS_PER_YEAR),
        "sharpe": mean / std * math.sqrt(DAYS_PER_YEAR) if std > 0 else None,
        "sortino": mean / downside * math.sqrt(DAYS_PER_YEAR) if downside > 0 else None,
        "max_drawdown": mdd,
        "calmar": cagr / -mdd if mdd < 0 else None,
    }


def compute(
    series: Sequence[tuple[date, float]],
    *,
    start: date | None = None,
    end: date | None = None,
    fx: FxRates | None = None,
) -> dict:
    """Métriques sur [start, end] (devise du portefeuille, et EUR si `fx`)."""
    s = window(series, start, end)
    out = {"start": s[0][0] if s else None, "end": s[-1][0] if s else None, "days": len(s)}
    out.update(_stats(s))
    if fx is not None and len(fx):
        eur = _stats([(d, fx.to_base(v, d)) for d, v in s])
        out.update({f"{k}_{fx.base.lower()}": v for k, v in eur.items()
                    if k in ("total_return", "cagr", "vol", "sharpe", "max_drawdown")})
    return out


def turnover(trades: Iterable[tuple[date, float]], series: Sequence[tuple[date, float]],
             start: date | None, end: date | None) -> float | None:
    """Volume échangé annualisé / valeur moyenne (1.0 = le portefeuille entier par an)."""
    s = window(series, start, end)
    if len(s) < 2:
        return None
    traded = sum(abs(n) for d, n in trades if s[0][0] <= d <= s[-1][0])
    avg = sum(v for _, v in s) / len(s)
    years = max((s[-1][0] - s[0][0]).days / 365.25, 1e-9)
    return traded / avg / years if avg > 0 else None
