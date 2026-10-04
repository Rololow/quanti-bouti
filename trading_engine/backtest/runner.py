"""Backtest : moteur sur données historiques, références, walk-forward, rapport.

Le moteur tourne **tel quel** sur le dataset (flux `dataset`) ; seules les
variantes de config changent. Toutes les stratégies sont mesurées sur la même
fenêtre d'évaluation, après une période de démarrage (le moteur y apprend :
covariance, régimes, prédicteurs).

Walk-forward : chaque année d'évaluation utilise le paramètre qui avait le
meilleur Sharpe sur les années d'évaluation précédentes (fenêtre croissante ;
première année : paramètre par défaut). La série hors échantillon est la
concaténation de ces choix. Approximation : chaque paramètre a son propre run
continu (on ne redémarre pas le moteur à chaque année).

Le HALTED du Safety Engine exige une reprise manuelle : le runner joue
l'opérateur (reprise après `halt_reset_after`) et compte les arrêts.
"""

from __future__ import annotations

import asyncio
import dataclasses
import itertools
import json
import logging
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

from trading_engine.backtest import metrics, stats
from trading_engine.backtest.benchmarks import (
    StrategyResult,
    buy_and_hold,
    daily_closes,
    monthly_fixed,
    monthly_inverse_vol,
)
from trading_engine.backtest.dataset import fx_path, read_closes
from trading_engine.config import Config, load_config, resolve_path
from trading_engine.data.calendar import NY
from trading_engine.data.events import EventType
from trading_engine.safety.safety_engine import SafetyState
from trading_engine.tax.fx import FxRates
from trading_engine.tax.profile import load_tax_profile
from trading_engine.tax.tax_model import TaxModel

logger = logging.getLogger(__name__)

# Variantes du moteur comparées (paramètres communs à toutes leurs exécutions).
VARIANTS: dict[str, dict[str, Any]] = {
    "complet": {"allocation.method": "signal", "decision.include_alpha": True},
    "sans_modeles": {"allocation.method": "risk_parity", "decision.include_alpha": False},
    # Une décision par séance (au lieu de chaque heure) : horizon où le momentum
    # documenté existe, et beaucoup moins de bruit de rebalancement.
    "seance": {"allocation.method": "risk_parity", "decision.include_alpha": False,
               "allocation.rebalance_timeframe": "session"},
}
VARIANT_LABELS = {
    "complet": "Moteur complet (signaux + décision)",
    "sans_modeles": "Moteur sans modèles (risk parity + décision)",
    "seance": "Moteur sans modèles, 1 décision/séance",
}
COST_KEYS = ("execution.cost.default_spread_bps", "execution.cost.slippage_bps",
             "decision.default_spread_bps", "decision.slippage_bps")


# ------------------------------------------------------------------ config

def with_params(cfg: Config, params: Mapping[str, Any]) -> Config:
    """Applique des paramètres pointés (`decision.holding_period`) à la config."""
    for key, value in params.items():
        parts = key.split(".")
        cfg = _replace_path(cfg, parts, value)
    return cfg


def _replace_path(obj: Any, parts: Sequence[str], value: Any) -> Any:
    head = parts[0]
    if not dataclasses.is_dataclass(obj) or not any(f.name == head for f in dataclasses.fields(obj)):
        raise KeyError(f"unknown config key {head!r}")
    current = getattr(obj, head)
    if len(parts) > 1:
        return dataclasses.replace(obj, **{head: _replace_path(current, parts[1:], value)})
    if isinstance(current, bool):
        value = value if isinstance(value, bool) else str(value).lower() in ("1", "true", "yes", "oui")
    elif isinstance(current, (int, float)) and not isinstance(value, (int, float)):
        value = type(current)(value)
    return dataclasses.replace(obj, **{head: value})


def parse_grid(items: Sequence[str]) -> list[dict[str, Any]]:
    """`["decision.holding_period=20d,60d"]` -> produit cartésien de dicts."""
    keys, values = [], []
    for item in items:
        key, _, raw = item.partition("=")
        if not key or not raw:
            raise ValueError(f"grid item must look like key=v1,v2: {item!r}")
        keys.append(key.strip())
        values.append([v.strip() for v in raw.split(",") if v.strip()])
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)] if keys else [{}]


def label_of(params: Mapping[str, Any]) -> str:
    return ", ".join(f"{k.split('.')[-1]}={v}" for k, v in params.items()) or "défaut"


# ------------------------------------------------------------------ exécution du moteur

def run_engine(cfg: Config, name: str, params: Mapping[str, Any], *,
               halt_reset_after: timedelta = timedelta(days=1), variant: str | None = None) -> StrategyResult:
    from trading_engine.engine import Engine        # import local : processus enfants

    engine = Engine(cfg)
    base_tf = cfg.bar_timeframes[0]
    equity: dict[date, float] = {}
    counters = {"bars": 0, "halted_bars": 0, "halts": 0, "resets": 0, "decisions": 0, "rebalances": 0}
    halt_reasons: dict[str, int] = {}
    halted_since: list[datetime | None] = [None]

    def on_bar(bar) -> None:
        if bar.timeframe != base_tf:
            return
        equity[bar.end.astimezone(NY).date()] = engine.portfolio.total_value()
        counters["bars"] += 1
        if engine.safety.state is SafetyState.HALTED:
            counters["halted_bars"] += 1
            if halted_since[0] is None:
                halted_since[0] = bar.end
                counters["halts"] += 1
                for reason in engine.safety.status.reasons:
                    key = reason.split(" ")[0]
                    halt_reasons[key] = halt_reasons.get(key, 0) + 1
            elif bar.end - halted_since[0] >= halt_reset_after:
                engine.safety.reset("backtest", bar.end)     # l'opérateur relance
                counters["resets"] += 1
                halted_since[0] = None
        else:
            halted_since[0] = None

    def on_decision(ev) -> None:
        counters["decisions"] += 1
        counters["rebalances"] += ev.payload.get("action") == "UREBALANCE"

    engine.bus.subscribe(EventType.BAR, on_bar)
    engine.bus.subscribe(EventType.DECISION, on_decision)
    t0 = time.perf_counter()
    asyncio.run(engine.run())
    tax = engine.tax
    return StrategyResult(
        name=name,
        equity=sorted(equity.items()),
        trades=[(f.timestamp.astimezone(NY).date(), abs(f.quantity * f.price)) for f in engine.fills],
        taxes=[] if tax is None else [(t.day, t.amount) for t in tax.transactions],
        params=dict(params),
        kind=f"engine:{variant}" if variant else "engine",
        stats={
            **counters,
            "halted_share": counters["halted_bars"] / counters["bars"] if counters["bars"] else 0.0,
            "halt_reasons": halt_reasons,
            "fills": len(engine.fills),
            "handler_errors": engine.bus.error_count,
            "data_rejected": sum(v for k, v in engine.integrity.counts.items()
                                 if k in ("INCONSISTENT_BAR", "GLITCH", "INVALID_SIZE")),
            "elapsed_s": round(time.perf_counter() - t0, 1),
            "final_safety": engine.safety.state.value,
        },
    )


def _run_job(job: dict) -> StrategyResult:
    logging.basicConfig(level=logging.ERROR)
    cfg = load_config(job["config"], tuple(job["overlays"]))
    cfg = with_params(cfg, {"feed.dataset_path": job["dataset"], **job["params"], **job["fixed"]})
    return run_engine(cfg, job["name"], job["params"], variant=job["variant"],
                      halt_reset_after=timedelta(days=job["halt_reset_days"]))


# ------------------------------------------------------------------ walk-forward

def walk_forward(runs: Sequence[StrategyResult], eval_start: date, eval_end: date,
                 default: StrategyResult, *, min_days: int = 60) -> tuple[StrategyResult, list[dict]]:
    """Choix annuel du meilleur Sharpe passé ; série hors échantillon recollée."""
    series = {r.name: dict(r.equity) for r in runs}
    folds, stitched, value = [], [], 1.0
    year_starts = [eval_start] + [date(y, 1, 1) for y in range(eval_start.year + 1, eval_end.year + 1)]
    days_all = sorted({d for r in runs for d, _ in r.equity if eval_start <= d <= eval_end})
    for k, fold_start in enumerate(year_starts):
        fold_end = year_starts[k + 1] - timedelta(days=1) if k + 1 < len(year_starts) else eval_end
        best, best_sharpe, reason = default, None, "défaut (pas assez d'historique)"
        for r in runs:
            past = metrics.window(r.equity, eval_start, fold_start - timedelta(days=1))
            if len(past) < min_days:
                continue
            sharpe = metrics.compute(past)["sharpe"]
            if sharpe is not None and (best_sharpe is None or sharpe > best_sharpe):
                best, best_sharpe, reason = r, sharpe, f"meilleur Sharpe passé ({sharpe:.2f})"
        chosen = series[best.name]
        fold_idx = [i for i, d in enumerate(days_all) if fold_start <= d <= fold_end]
        start_value = value
        for i in fold_idx:
            d, prev = days_all[i], days_all[i - 1] if i > 0 else None
            if prev is not None and chosen.get(prev, 0) > 0 and d in chosen:
                value *= chosen[d] / chosen[prev]
            stitched.append((d, value))
        folds.append({"start": fold_start, "end": fold_end, "chosen": best.name, "params": best.params,
                      "reason": reason, "return": value / start_value - 1.0 if fold_idx else None})
    # Coûts et taxes sont déjà dans chaque série recollée.
    return StrategyResult("", stitched, kind="walk_forward"), folds


# ------------------------------------------------------------------ rapport

def _tax_fn(cfg: Config, fx: FxRates | None):
    if not cfg.tax.profile:
        return None
    model = TaxModel(load_tax_profile(resolve_path(cfg.tax.profile)), cfg.instruments,
                     portfolio_currency=cfg.fx.portfolio_currency if fx is not None else None)
    if fx is not None:
        model.set_fx(fx)

    def tax(symbol: str, notional: float, side: str, day: date) -> tuple[float, float]:
        amount = model.transaction_tax(symbol, model.to_tax(notional, day), side).amount
        return model.to_portfolio(amount, day), amount
    return tax


def load_fx(cfg: Config, dataset: Path) -> FxRates | None:
    path = fx_path(dataset)
    if cfg.fx.provider == "off":
        return None
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            return FxRates.from_payload(json.load(fh))
    return FxRates.fixed("EUR", cfg.fx.portfolio_currency, cfg.fx.fixed_rate, date(1970, 1, 1))


def row(result: StrategyResult, eval_start: date, eval_end: date, fx: FxRates | None) -> dict:
    m = metrics.compute(result.equity, start=eval_start, end=eval_end, fx=fx)
    m["turnover"] = metrics.turnover(result.trades, result.equity, eval_start, eval_end)
    m["transaction_tax"] = sum(a for d, a in result.taxes if eval_start <= d <= eval_end)
    return {"name": result.name, "kind": result.kind, "params": result.params, "metrics": m,
            "stats": result.stats}


def run_backtest(
    dataset: str | Path,
    *,
    config: str | Path,
    overlays: Sequence[str | Path],
    eval_start: date | None = None,
    eval_end: date | None = None,
    grid: Sequence[Mapping[str, Any]] = ({},),
    variants: Sequence[str] = tuple(VARIANTS),
    workers: int | None = None,
    halt_reset_days: float = 1.0,
    cost_stress: float = 2.0,
    bootstrap_samples: int = 2000,
) -> dict:
    dataset = resolve_path(dataset)
    cfg = with_params(load_config(config, tuple(overlays)), {"feed.dataset_path": str(dataset)})
    closes = read_closes(dataset)
    if not closes:
        raise ValueError(f"no bars in {dataset}")
    days, prices = daily_closes(closes)
    eval_start = eval_start or days[0] + timedelta(days=365)
    eval_end = eval_end or days[-1]
    if eval_start >= eval_end:
        raise ValueError(f"evaluation window is empty ({eval_start} >= {eval_end})")
    fx = load_fx(cfg, dataset)

    def job(v, params, fixed, name, variant_kind):
        return {"config": str(config), "overlays": [str(o) for o in overlays], "dataset": str(dataset),
                "params": dict(params), "fixed": {**VARIANTS[v], **fixed}, "variant": variant_kind,
                "halt_reset_days": halt_reset_days, "name": name}

    default_index = len(grid) // 2                     # valeur centrale de la grille
    jobs = [job(v, p, {}, f"{VARIANT_LABELS[v]} [{label_of(p)}]", v) for v in variants for p in grid]
    stressed_costs = {}
    if cost_stress and cost_stress != 1.0:
        # Coûts multipliés (exécution ET estimation de la décision) : le moteur
        # doit survivre à des coûts plus élevés que prévu.
        stressed_costs = {k: _get_path(cfg, k) * cost_stress for k in COST_KEYS}
        jobs += [job(v, grid[default_index], stressed_costs,
                     f"{VARIANT_LABELS[v]} [{label_of(grid[default_index])}, coûts ×{cost_stress:g}]", f"stress:{v}")
                 for v in variants]
    workers = max(1, min(workers or os.cpu_count() or 1, len(jobs)))
    logger.info("backtest: %d engine runs on %d worker(s)", len(jobs), workers)
    if workers == 1:
        engine_runs = [_run_job(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            engine_runs = list(pool.map(_run_job, jobs))

    cost_bps = cfg.execution.cost.default_spread_bps / 2 + cfg.execution.cost.slippage_bps
    tax = _tax_fn(cfg, fx)
    cash = cfg.portfolio.cash
    min_cash = cfg.allocation.constraints.min_cash
    benchmarks = [
        buy_and_hold(days, prices, cash=cash, cost_bps=cost_bps, tax=tax, min_cash=min_cash),
        monthly_inverse_vol(days, prices, cash=cash, cost_bps=cost_bps, tax=tax, min_cash=min_cash),
    ]

    sixty_forty = monthly_fixed("60/40 SPY/TLT (mensuel)", days, prices, {"SPY": 0.6, "TLT": 0.4},
                                cash=cash, cost_bps=cost_bps, tax=tax, min_cash=min_cash)
    if sixty_forty is not None:
        benchmarks.append(sixty_forty)

    rows = [row(b, eval_start, eval_end, fx) for b in benchmarks]
    walk = []
    wf_series: dict[str, list[tuple[date, float]]] = {}
    for v in variants:
        runs = [r for r in engine_runs if r.kind == f"engine:{v}"]
        rows.extend(row(r, eval_start, eval_end, fx) for r in runs)
        if len(runs) > 1:
            wf, folds = walk_forward(runs, eval_start, eval_end, runs[default_index])
            wf.name = f"{VARIANT_LABELS[v]} — walk-forward (hors échantillon)"
            r = row(wf, eval_start, eval_end, fx)
            r["metrics"]["turnover"] = None             # dépend des runs choisis, cf. folds
            r["metrics"]["transaction_tax"] = None      # déjà dans chaque série recollée
            rows.append(r)
            walk.append({"variant": v, "folds": folds})
            wf_series[wf.name] = wf.equity

    # Stress des coûts : moteur (runs dédiés) et références (recalculées).
    stress_rows = []
    if stressed_costs:
        bps = cost_bps * cost_stress
        stressed = [buy_and_hold(days, prices, cash=cash, cost_bps=bps, tax=tax, min_cash=min_cash),
                    monthly_inverse_vol(days, prices, cash=cash, cost_bps=bps, tax=tax, min_cash=min_cash)]
        if sixty_forty is not None:
            stressed.append(monthly_fixed("60/40 SPY/TLT (mensuel)", days, prices, {"SPY": 0.6, "TLT": 0.4},
                                          cash=cash, cost_bps=bps, tax=tax, min_cash=min_cash))
        for b in stressed:
            b.name += f" [coûts ×{cost_stress:g}]"
            stress_rows.append(row(b, eval_start, eval_end, fx))
        stress_rows += [row(r, eval_start, eval_end, fx) for r in engine_runs if r.kind.startswith("engine:stress:")]

    series = {r.name: r.equity for r in benchmarks}
    series.update({r.name: r.equity for r in engine_runs})
    series.update(wf_series)
    trial_sharpes = [r["metrics"]["sharpe"] for r in rows if r["kind"].startswith("engine:")]
    for r in rows:
        r.update(robustness(series[r["name"]], eval_start, eval_end,
                            benchmarks=[] if r["kind"] == "benchmark" else benchmarks,
                            trial_sharpes=trial_sharpes if r["kind"] != "benchmark" else None,
                            samples=bootstrap_samples))

    return {
        "dataset": str(dataset), "bars_start": days[0], "bars_end": days[-1],
        "eval_start": eval_start, "eval_end": eval_end, "symbols": sorted(prices),
        "fx": None if fx is None else {"source": fx.source, "rates": len(fx)},
        "tax_currency": None if tax is None else load_tax_profile(resolve_path(cfg.tax.profile)).currency,
        "currency": cfg.fx.portfolio_currency, "initial_cash": cash, "rows": rows, "walk_forward": walk,
        "cost_stress": {"multiplier": cost_stress, "rows": stress_rows} if stress_rows else None,
        "crises": [{"name": n, "start": a, "end": b} for n, a, b in stats.CRISES if b >= eval_start and a <= eval_end],
        "provenance": provenance(dataset, config, overlays, grid, variants, cost_stress),
        "caveats": CAVEATS,
    }


def _get_path(cfg: Any, key: str) -> Any:
    for part in key.split("."):
        cfg = getattr(cfg, part)
    return cfg


def robustness(series: Sequence[tuple[date, float]], start: date, end: date, *,
               benchmarks: Sequence[StrategyResult], trial_sharpes: Sequence[float] | None,
               samples: int = 2000) -> dict:
    """Intervalle de confiance du Sharpe, comparaison appariée aux références,
    Sharpe dégonflé, rendements annuels et par crise."""
    window = metrics.window(series, start, end)
    daily = stats.returns_by_day(window)
    returns = [daily[d] for d in sorted(daily)]
    out: dict[str, Any] = {
        "sharpe_ci": stats.bootstrap_sharpe(returns, samples=samples),
        "years": stats.yearly_returns(series, start, end),
        "crises": {name: stats.period_stats(series, max(a, start), min(b, end))
                   for name, a, b in stats.CRISES if b >= start and a <= end},
    }
    if benchmarks:
        out["vs"] = {b.name: stats.bootstrap_difference(daily, stats.returns_by_day(metrics.window(b.equity, start, end)),
                                                        samples=samples)
                     for b in benchmarks}
    if trial_sharpes is not None:
        out["deflated_sharpe"] = stats.deflated_sharpe(returns, trial_sharpes)
    return out


def provenance(dataset: Path, config, overlays, grid, variants, cost_stress) -> dict:
    """De quoi refaire exactement ce backtest : données, code, config, versions."""
    import hashlib
    import platform
    import subprocess

    import numpy

    h = hashlib.sha256()
    with open(dataset, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
                                timeout=10).stdout.strip() or None
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                                    capture_output=True, text=True, timeout=10).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    cfg_hash = hashlib.sha256()
    for path in [config, *overlays]:
        cfg_hash.update(Path(path).read_bytes())
    return {
        "dataset_sha256": h.hexdigest(), "dataset_bytes": dataset.stat().st_size,
        "git_commit": commit, "git_dirty": dirty,
        "config": str(config), "overlays": [str(o) for o in overlays], "config_sha256": cfg_hash.hexdigest(),
        "grid": [dict(g) for g in grid], "variants": list(variants), "cost_stress": cost_stress,
        "python": platform.python_version(), "numpy": numpy.__version__, "platform": platform.platform(),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


CAVEATS = [
    "Barres 30 min rejouées en trades synthétiques (O, plus bas/haut, C) : le chemin intra-barre est approximé.",
    "Pas de cotations historiques : spread = spread par défaut du modèle de coûts.",
    "Prix ajustés (adjustment=all) : dividendes réinvestis implicitement, précompte et retenue US non déduits.",
    "Impôt sur les plus-values non déduit des séries (estimé séparément par le moteur) ; TOB déduite.",
    "Références exécutées à la clôture avec fractions d'actions : hypothèse favorable aux références.",
    "Sharpe/Sortino avec taux sans risque nul.",
    "Walk-forward : un run continu par paramètre (le moteur n'est pas redémarré à chaque année).",
]


# ------------------------------------------------------------------ présentation

def _pct(v, digits=1):
    return "–" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v * 100:.{digits}f} %"


def _num(v, digits=2):
    return "–" if v is None else f"{v:.{digits}f}"


def _row_line(r: dict) -> str:
    m, s = r["metrics"], r["stats"]
    halts = "–" if not s else f"{s.get('halts', 0)} ({_pct(s.get('halted_share'), 0)})"
    if s and s.get("halt_reasons"):
        halts += " " + ", ".join(f"{k}×{n}" for k, n in sorted(s["halt_reasons"].items()))
    ci = (r.get("sharpe_ci") or {})
    sharpe = _num(m.get("sharpe"))
    if ci.get("low") is not None:
        sharpe += f" [{_num(ci['low'])} ; {_num(ci['high'])}]"
    return (f"| {r['name']} | {_pct(m.get('cagr'))} | {_pct(m.get('vol'))} | {sharpe} | "
            f"{_pct(m.get('max_drawdown'))} | {_pct(m.get('cagr_eur'))} | {_pct(m.get('max_drawdown_eur'))} | "
            f"{_num(m.get('turnover'))} | {_num(m.get('transaction_tax'), 0)} | {halts} |")


def format_report(report: dict) -> str:
    ccy, tccy = report["currency"], report.get("tax_currency") or ""
    header = (f"| Stratégie | CAGR {ccy} | Vol | Sharpe [IC 90 %] | Max DD | CAGR EUR | Max DD EUR | "
              f"Turnover/an | TOB {tccy} | Arrêts |\n|---|---|---|---|---|---|---|---|---|---|")
    prov = report.get("provenance") or {}
    lines = [
        f"Backtest {report['symbols']} — données {report['bars_start']} → {report['bars_end']}, "
        f"évaluation {report['eval_start']} → {report['eval_end']}",
        f"Capital initial {report['initial_cash']:,.0f} {ccy} ; change : "
        f"{(report['fx'] or {}).get('source', 'aucun')}",
        "", "## Résultats", "", header,
    ]
    lines += [_row_line(r) for r in report["rows"]]

    engine_rows = [r for r in report["rows"] if r["kind"] != "benchmark" and r.get("vs")]
    if engine_rows:
        bench = list(engine_rows[0]["vs"])
        lines += ["", "## Le moteur bat-il les références ?", "",
                  "Écart de Sharpe (moteur − référence), IC 90 % par bootstrap par blocs de 20 séances, "
                  "et probabilité que l'écart soit positif. Sharpe dégonflé : probabilité que le vrai "
                  "Sharpe soit > 0 compte tenu du nombre de configurations essayées.", "",
                  "| Stratégie | " + " | ".join(bench) + " | Sharpe dégonflé |",
                  "|---|" + "---|" * (len(bench) + 1)]
        for r in engine_rows:
            cells = []
            for b in bench:
                d = r["vs"][b]
                cells.append("–" if d["difference"] is None else
                             f"{d['difference']:+.2f} [{_num(d['low'])} ; {_num(d['high'])}], "
                             f"P>0 = {_pct(d['p_better'], 0)}" if d["low"] is not None else f"{d['difference']:+.2f}")
            dsr = (r.get("deflated_sharpe") or {})
            cells.append("–" if dsr.get("dsr") is None else f"{_pct(dsr['dsr'], 0)} (N={dsr['trials']})")
            lines.append(f"| {r['name']} | " + " | ".join(cells) + " |")

    years = sorted({y for r in report["rows"] for y in (r.get("years") or {})})
    if years:
        lines += ["", "## Rendement par année", "", "| Stratégie | " + " | ".join(map(str, years)) + " |",
                  "|---|" + "---|" * len(years)]
        for r in report["rows"]:
            ys = r.get("years") or {}
            lines.append(f"| {r['name']} | " + " | ".join(_pct(ys.get(y)) for y in years) + " |")

    crises = report.get("crises") or []
    if crises:
        names = [c["name"] for c in crises]
        lines += ["", "## Épisodes de stress (rendement / drawdown max)", "",
                  "| Stratégie | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for r in report["rows"]:
            cs = r.get("crises") or {}
            lines.append(f"| {r['name']} | " + " | ".join(
                "–" if not cs.get(n) else f"{_pct(cs[n]['return'])} / {_pct(cs[n]['max_drawdown'])}"
                for n in names) + " |")

    stress = report.get("cost_stress")
    if stress:
        lines += ["", f"## Stress des coûts (spread et slippage ×{stress['multiplier']:g})", "", header]
        lines += [_row_line(r) for r in stress["rows"]]

    for wf in report["walk_forward"]:
        lines += ["", f"## Walk-forward « {VARIANT_LABELS[wf['variant']]} »", ""]
        for f in wf["folds"]:
            lines.append(f"- {f['start']} → {f['end']} : {label_of(f['params'])} — {f['reason']} — "
                         f"rendement {_pct(f['return'])}")
    lines += ["", "## Limites", ""] + [f"- {c}" for c in report["caveats"]]
    if prov:
        lines += ["", "## Traçabilité", "",
                  f"- données : `{report['dataset']}` sha256 {prov.get('dataset_sha256', '')[:16]}…",
                  f"- code : commit {prov.get('git_commit') or '?'}" + (" (modifications non commitées)"
                                                                       if prov.get("git_dirty") else ""),
                  f"- config : {prov.get('config')} + {prov.get('overlays')} sha256 {prov.get('config_sha256', '')[:16]}…",
                  f"- Python {prov.get('python')}, numpy {prov.get('numpy')}, {prov.get('platform')}",
                  f"- généré le {prov.get('created_at')}"]
    return "\n".join(lines)


def to_json(report: dict) -> str:
    def default(o):
        if isinstance(o, (date, datetime)):
            return o.isoformat()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(type(o).__name__)
    return json.dumps(report, default=default, indent=1, ensure_ascii=False)
