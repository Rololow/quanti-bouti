"""Statistiques de robustesse des résultats de backtest.

Un backtest unique peut sembler bon par chance. Ces outils disent avec quelle
confiance :

- **bootstrap par blocs** (blocs de 20 séances, autocorrélation et clusters
  de volatilité conservés) : intervalle de confiance du Sharpe, et probabilité
  que le moteur ait un meilleur Sharpe qu'une référence (rééchantillonnage
  apparié des mêmes jours) ;
- **Sharpe dégonflé** (Bailey & López de Prado, 2014) : probabilité que le
  vrai Sharpe soit positif, en tenant compte du nombre de configurations
  essayées (le meilleur de N essais est gonflé), de l'asymétrie et des queues
  épaisses des rendements ;
- découpage par **année** et par **crise** (un résultat porté par un seul
  épisode n'est pas robuste).
"""

from __future__ import annotations

import math
from datetime import date
from statistics import NormalDist
from typing import Mapping, Sequence

import numpy as np

from trading_engine.backtest import metrics

DAYS = metrics.DAYS_PER_YEAR
EULER_GAMMA = 0.5772156649015329
_N = NormalDist()

# Épisodes de stress (pic -> creux du S&P 500, sauf 2022 : année des taux).
CRISES: tuple[tuple[str, date, date], ...] = (
    ("Crise financière 2008", date(2007, 10, 9), date(2009, 3, 9)),
    ("Dette US / euro 2011", date(2011, 4, 29), date(2011, 10, 3)),
    ("Chine / pétrole 2015-16", date(2015, 5, 21), date(2016, 2, 11)),
    ("Volmageddon 2018", date(2018, 1, 26), date(2018, 2, 8)),
    ("Fin 2018", date(2018, 9, 20), date(2018, 12, 24)),
    ("Covid 2020", date(2020, 2, 19), date(2020, 3, 23)),
    ("Taux 2022", date(2022, 1, 3), date(2022, 10, 12)),
    ("Banques régionales 2023", date(2023, 3, 8), date(2023, 3, 13)),
    ("Carry trade 2024", date(2024, 7, 16), date(2024, 8, 5)),
    ("Droits de douane 2025", date(2025, 2, 19), date(2025, 4, 8)),
)


def returns_by_day(series: Sequence[tuple[date, float]]) -> dict[date, float]:
    return {d: v / p - 1.0 for (_, p), (d, v) in zip(series, series[1:]) if p > 0}


def sharpe(r: np.ndarray) -> float | None:
    if len(r) < 2:
        return None
    sd = r.std(ddof=1)
    return float(r.mean() / sd * math.sqrt(DAYS)) if sd > 0 else None


def _block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Moving block bootstrap : blocs contigus tirés au hasard, recollés."""
    block = max(1, min(block, n))
    starts = rng.integers(0, n - block + 1, size=math.ceil(n / block))
    return (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]


def bootstrap_sharpe(r: Sequence[float], *, block: int = 20, samples: int = 2000, seed: int = 0,
                     level: float = 0.90) -> dict:
    """Sharpe annualisé et intervalle de confiance (bootstrap par blocs)."""
    r = np.asarray(r, dtype=float)
    point = sharpe(r)
    if point is None or len(r) < 2 * block:
        return {"sharpe": point, "low": None, "high": None}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(samples):
        s = sharpe(r[_block_indices(len(r), block, rng)])
        if s is not None:
            draws.append(s)
    lo, hi = np.quantile(draws, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"sharpe": point, "low": float(lo), "high": float(hi)}


def bootstrap_difference(a: Mapping[date, float], b: Mapping[date, float], *, block: int = 20,
                         samples: int = 2000, seed: int = 0, level: float = 0.90) -> dict:
    """Écart de Sharpe a - b sur les jours communs, rééchantillonnés ensemble."""
    days = sorted(set(a) & set(b))
    ra = np.array([a[d] for d in days])
    rb = np.array([b[d] for d in days])
    sa, sb = sharpe(ra), sharpe(rb)
    if sa is None or sb is None or len(days) < 2 * block:
        return {"difference": None if sa is None or sb is None else sa - sb, "low": None, "high": None,
                "p_better": None, "days": len(days)}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(samples):
        idx = _block_indices(len(days), block, rng)
        x, y = sharpe(ra[idx]), sharpe(rb[idx])
        if x is not None and y is not None:
            diffs.append(x - y)
    diffs = np.array(diffs)
    lo, hi = np.quantile(diffs, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"difference": sa - sb, "low": float(lo), "high": float(hi),
            "p_better": float((diffs > 0).mean()), "days": len(days)}


def deflated_sharpe(r: Sequence[float], trial_sharpes: Sequence[float]) -> dict:
    """Probabilité que le vrai Sharpe soit > 0 après N essais (PSR si N = 1).

    `trial_sharpes` : Sharpe annualisés de toutes les configurations testées.
    """
    r = np.asarray(r, dtype=float)
    n = len(r)
    if n < 10 or r.std(ddof=1) == 0:
        return {"dsr": None, "trials": len(trial_sharpes), "sr0": None}
    sr = r.mean() / r.std(ddof=1)                      # par séance
    centered = r - r.mean()
    m2 = (centered ** 2).mean()
    skew = (centered ** 3).mean() / m2 ** 1.5
    kurt = (centered ** 4).mean() / m2 ** 2
    trials = [s / math.sqrt(DAYS) for s in trial_sharpes if s is not None and math.isfinite(s)]
    k = len(trials)
    if k > 1 and np.var(trials, ddof=1) > 0:
        sd = math.sqrt(np.var(trials, ddof=1))
        sr0 = sd * ((1 - EULER_GAMMA) * _N.inv_cdf(1 - 1 / k) + EULER_GAMMA * _N.inv_cdf(1 - 1 / (k * math.e)))
    else:
        sr0 = 0.0
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr ** 2
    if denom <= 0:
        return {"dsr": None, "trials": k, "sr0": sr0 * math.sqrt(DAYS)}
    z = (sr - sr0) * math.sqrt(n - 1) / math.sqrt(denom)
    return {"dsr": _N.cdf(z), "trials": k, "sr0": sr0 * math.sqrt(DAYS)}


def yearly_returns(series: Sequence[tuple[date, float]], start: date, end: date) -> dict[int, float]:
    """Rendement de chaque année civile sur [start, end] (base : dernière valeur
    de l'année précédente, ou première valeur de la fenêtre)."""
    s = metrics.window(series, start, end)
    out: dict[int, float] = {}
    for year in sorted({d.year for d, _ in s}):
        inside = [v for d, v in s if d.year == year]
        before = [v for d, v in s if d.year < year]
        base = before[-1] if before else inside[0]
        if base > 0:
            out[year] = inside[-1] / base - 1.0
    return out


def period_stats(series: Sequence[tuple[date, float]], start: date, end: date) -> dict | None:
    """Rendement et drawdown sur une période (la veille du début sert de base)."""
    before = [x for x in series if x[0] < start]
    inside = [x for x in series if start <= x[0] <= end]
    if not inside:
        return None
    s = ([before[-1]] if before else []) + inside
    if len(s) < 2 or s[0][1] <= 0:
        return None
    return {"return": s[-1][1] / s[0][1] - 1.0, "max_drawdown": metrics.max_drawdown(v for _, v in s)}
