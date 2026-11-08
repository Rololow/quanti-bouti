"""Audit de robustesse du profil belge : le profil est-il un pic de surapprentissage ?

Trois mesures, toutes sur le même dataset EUR que le notebook :

- sensibilité : chaque paramètre réglé est déplacé seul autour de la valeur
  retenue (les autres restent au profil). Un plateau (résultat voisin pour des
  valeurs voisines) plaide pour la robustesse ; un pic isolé est suspect ;
- deux périodes : 2007-2016 et 2017-2026, mêmes paramètres, pour voir si le
  profil garde son rang ;
- Sharpe dégonflé : probabilité que le vrai Sharpe soit positif après N essais,
  pour N = nombre de configurations de cet audit, et pour un total estimé des
  essais faits sur ces données (hypothèse, pas une mesure).

Les périodes utilisent la courbe brute en EUR (sans liquidation) ; le rendement
« liquidé » n'existe que sur la période complète.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import date
from pathlib import Path
from statistics import NormalDist

import numpy as np

from trading_engine.backtest import metrics
from trading_engine.backtest.stats import DAYS, EULER_GAMMA, deflated_sharpe
from trading_engine.config import DEFAULT_CONFIG_PATH, resolve_path

OVERLAYS = ("config/backtest_daily.yaml", "config/profil_belge.yaml")
EVAL_START = date(2007, 3, 1)
EVAL_END = date(2026, 11, 6)
FIRST_HALF = (EVAL_START, date(2016, 12, 31))
SECOND_HALF = (date(2017, 1, 1), EVAL_END)
# Essais faits sur ces données hors de cet audit (hypothèse : ~30 configurations
# depuis le début du projet). Estimation, à remplacer par un décompte réel.
TOTAL_TRIALS_ESTIMATE = 30

BASE = ("retenu", {})

# Un paramètre à la fois, autour de la valeur retenue dans config/profil_belge.yaml.
ONE_AT_A_TIME: list[tuple[str, dict]] = [
    ("allocation.drawdown_peak_decay = 0.05", {"allocation.drawdown_peak_decay": 0.05}),
    ("allocation.drawdown_peak_decay = 0.20", {"allocation.drawdown_peak_decay": 0.20}),
    ("decision.force_risk_reduction = 0.05", {"decision.force_risk_reduction": 0.05}),
    ("decision.force_risk_reduction = 0.20", {"decision.force_risk_reduction": 0.20}),
    ("allocation.drawdown_gap_sigmas = 1.0", {"allocation.drawdown_gap_sigmas": 1.0}),
    ("allocation.drawdown_gap_sigmas = 3.5", {"allocation.drawdown_gap_sigmas": 3.5}),
    ("allocation.drawdown_control = 0.12", {"allocation.drawdown_control": 0.12}),
    ("allocation.drawdown_control = 0.18", {"allocation.drawdown_control": 0.18}),
    ("allocation.drawdown_multiplier = 1.5", {"allocation.drawdown_multiplier": 1.5}),
    ("allocation.drawdown_multiplier = 2.5", {"allocation.drawdown_multiplier": 2.5}),
    ("allocation.target_vol = 0.10", {"allocation.target_vol": 0.10}),
    ("allocation.target_vol = 0.14", {"allocation.target_vol": 0.14}),
    ("allocation.trend.min_change = 0.25", {"allocation.trend.min_change": 0.25}),
    ("allocation.trend.min_change = 0.75", {"allocation.trend.min_change": 0.75}),
    ("allocation.trend.horizons = 63,126,252", {"allocation.trend.horizons": (63, 126, 252)}),
    ("allocation.trend.horizons = 21,63,126,252,504", {"allocation.trend.horizons": (21, 63, 126, 252, 504)}),
]


def deflated_for_k(returns: list[float], trial_sd_annual: float, k: int) -> float:
    """Même formule que stats.deflated_sharpe, pour K essais de dispersion donnée.

    Permet de compter des essais qui ne figurent pas dans le rapport.
    """
    r = np.asarray(returns, dtype=float)
    n = len(r)
    if n < 10 or r.std(ddof=1) == 0 or k < 1:
        return float("nan")
    sr = r.mean() / r.std(ddof=1)
    c = r - r.mean()
    m2 = (c ** 2).mean()
    skew = (c ** 3).mean() / m2 ** 1.5
    kurt = (c ** 4).mean() / m2 ** 2
    sd_day = trial_sd_annual / math.sqrt(DAYS)
    nd = NormalDist()
    if k == 1:
        sr0 = 0.0
    else:
        sr0 = sd_day * ((1 - EULER_GAMMA) * nd.inv_cdf(1 - 1 / k)
                        + EULER_GAMMA * nd.inv_cdf(1 - 1 / (k * math.e)))
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr ** 2
    if denom <= 0:
        return float("nan")
    z = (sr - sr0) * math.sqrt(n - 1) / math.sqrt(denom)
    return nd.cdf(z)


def window_metrics(points: list[tuple[date, float]], start: date, end: date) -> dict:
    m = metrics.compute(points, start=start, end=end)
    return {"cagr": m["cagr"], "vol": m["vol"], "sharpe": m["sharpe"], "max_drawdown": m["max_drawdown"]}


def profile_row(report: dict) -> dict:
    rows = [r for r in report["rows"] if r["kind"] == "engine:profil"]
    if len(rows) != 1:
        raise ValueError(f"attendu une ligne moteur profil, trouvé {len(rows)}")
    return rows[0]


def run_config(dataset: str | Path, params: dict, *, bootstrap_samples: int = 100) -> dict:
    from trading_engine.backtest.runner import run_backtest

    report = run_backtest(
        dataset, config=DEFAULT_CONFIG_PATH, overlays=list(OVERLAYS),
        grid=[{"allocation.gross_band": 0.10, **params}], variants=["profil"],
        cost_stress=0, bootstrap_samples=bootstrap_samples, include_series=True,
    )
    row = profile_row(report)
    points = [(date.fromisoformat(d), float(v)) for d, v in report["series"]["equity_eur"][row["name"]]]
    m = row["metrics"]
    return {
        "name": row["name"],
        "full": {
            "cagr_liquide": m.get("cagr_eur_net"), "cagr": m.get("cagr_eur"),
            "max_drawdown": m.get("max_drawdown_eur"), "sharpe": m.get("sharpe_eur"),
            "vol": m.get("vol_eur"), "tob": m.get("transaction_tax"), "rotation": m.get("turnover"),
        },
        "first_half": window_metrics(points, *FIRST_HALF),
        "second_half": window_metrics(points, *SECOND_HALF),
        "returns": metrics.daily_returns(points),
        "points": points,
    }


def audit(dataset: str | Path) -> dict:
    results = []
    for label, params in [BASE, *ONE_AT_A_TIME]:
        res = run_config(dataset, params)
        res.update({"label": label, "params": {k: list(v) if isinstance(v, tuple) else v
                                              for k, v in params.items()}})
        results.append(res)
        print(f"{label}: liquidé {res['full']['cagr_liquide']:.2%}, perte max {res['full']['max_drawdown']:.1%}",
              flush=True)

    base = results[0]
    sharpes = [r["full"]["sharpe"] for r in results if r["full"]["sharpe"] is not None]
    sd_measured = float(np.std(sharpes, ddof=1)) if len(sharpes) > 1 else 0.0
    # Les réglages de l'audit sont voisins : leur dispersion mesurée sous-estime la
    # correction de multiplicité. On montre donc aussi des dispersions plus larges,
    # qui couvrent des choix structurels (classes, levier, limite) hors audit.
    sds = {"mesurée": sd_measured, "0,15": 0.15, "0,25": 0.25}
    dsr = {
        "audit_trials": len(results),
        "total_trials_estimate": TOTAL_TRIALS_ESTIMATE,
        "p_single": deflated_sharpe(base["returns"], [base["full"]["sharpe"]])["dsr"],
        "sd_sharpe_annual": sd_measured,
        "grid": {name: {"audit": deflated_for_k(base["returns"], sd, len(results)),
                        "total": deflated_for_k(base["returns"], sd, TOTAL_TRIALS_ESTIMATE)}
                 for name, sd in sds.items()},
    }
    for r in results:                       # les courbes ne servent qu'au calcul
        del r["returns"], r["points"]
    return {"results": results, "dsr": dsr, "periods": {"full": [EVAL_START, EVAL_END],
                                                       "first_half": FIRST_HALF, "second_half": SECOND_HALF}}


def _pct(x, digits=1):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.{digits}f} %".replace(".", ",")


def _num(x, digits=2):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}f}".replace(".", ",")


def to_markdown(out: dict) -> str:
    res = out["results"]
    lines = ["## Sensibilité : un paramètre à la fois", "",
             "| Réglage | Rendement liquidé | Perte max | Sharpe | Vol | TOB € | Rotation |",
             "|---|---|---|---|---|---|---|"]
    for r in res:
        f = r["full"]
        tob = f"{round(f['tob'] or 0):,}".replace(",", " ")
        lines.append(f"| {r['label']} | {_pct(f['cagr_liquide'])} | {_pct(f['max_drawdown'])} | "
                     f"{_num(f['sharpe'])} | {_pct(f['vol'])} | {tob} | {_num(f['rotation'])} |")
    lines += ["", "## Deux périodes (courbe brute en EUR, sans liquidation)", "",
              "| Réglage | 2007-2016 : rendement | Perte max | Sharpe | 2017-2026 : rendement | Perte max | Sharpe |",
              "|---|---|---|---|---|---|---|"]
    for r in res:
        a, b = r["first_half"], r["second_half"]
        lines.append(f"| {r['label']} | {_pct(a['cagr'])} | {_pct(a['max_drawdown'])} | {_num(a['sharpe'])} | "
                     f"{_pct(b['cagr'])} | {_pct(b['max_drawdown'])} | {_num(b['sharpe'])} |")
    d = out["dsr"]
    lines += ["", "## Sharpe dégonflé (probabilité que le vrai Sharpe soit > 0)", "",
              f"Sans correction : {_pct(d['p_single'], 0)}. Après correction pour N essais, "
              f"selon la dispersion des Sharpe entre essais (σ) :", "",
              f"| σ | N = {d['audit_trials']} (cet audit) | N = {d['total_trials_estimate']} (estimation) |",
              "|---|---|---|"]
    for name, v in d["grid"].items():
        lines.append(f"| {name} | {_pct(v['audit'], 0)} | {_pct(v['total'], 0)} |")
    lines += ["", f"σ mesurée sur les réglages de cet audit : {_num(d['sd_sharpe_annual'])} "
                  "(trop faible pour corriger seule : les réglages sont voisins)."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Audit de robustesse du profil belge")
    p.add_argument("--data", default="data/backtest/daily_eur.jsonl")
    p.add_argument("--out", default="data/backtest/audit.json")
    args = p.parse_args(argv)
    out = audit(args.data)
    path = resolve_path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, default=str, ensure_ascii=False, indent=1), encoding="utf-8")
    md = to_markdown(out)
    path.with_suffix(".md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
