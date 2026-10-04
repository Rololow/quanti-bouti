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

from trading_engine.backtest import metrics
from trading_engine.backtest.benchmarks import (
    StrategyResult,
    buy_and_hold,
    daily_closes,
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
}
VARIANT_LABELS = {
    "complet": "Moteur complet (signaux + décision)",
    "sans_modeles": "Moteur sans modèles (risk parity + décision)",
}


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

    jobs = [
        {"config": str(config), "overlays": [str(o) for o in overlays], "dataset": str(dataset),
         "params": dict(p), "fixed": VARIANTS[v], "variant": v, "halt_reset_days": halt_reset_days,
         "name": f"{VARIANT_LABELS[v]} [{label_of(p)}]"}
        for v in variants for p in grid
    ]
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

    rows = [row(b, eval_start, eval_end, fx) for b in benchmarks]
    walk = []
    default_index = len(grid) // 2                     # valeur centrale de la grille
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

    return {
        "dataset": str(dataset), "bars_start": days[0], "bars_end": days[-1],
        "eval_start": eval_start, "eval_end": eval_end, "symbols": sorted(prices),
        "fx": None if fx is None else {"source": fx.source, "rates": len(fx)},
        "tax_currency": None if tax is None else load_tax_profile(resolve_path(cfg.tax.profile)).currency,
        "currency": cfg.fx.portfolio_currency, "initial_cash": cash, "rows": rows, "walk_forward": walk,
        "caveats": CAVEATS,
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


def format_report(report: dict) -> str:
    ccy, tccy = report["currency"], report.get("tax_currency") or ""
    lines = [
        f"Backtest {report['symbols']} — données {report['bars_start']} → {report['bars_end']}, "
        f"évaluation {report['eval_start']} → {report['eval_end']}",
        f"Capital initial {report['initial_cash']:,.0f} {ccy} ; change : "
        f"{(report['fx'] or {}).get('source', 'aucun')}",
        "",
        f"| Stratégie | CAGR {ccy} | Vol | Sharpe | Max DD | CAGR EUR | Max DD EUR | Turnover/an | TOB {tccy} | Arrêts |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in report["rows"]:
        m, s = r["metrics"], r["stats"]
        halts = "–" if not s else f"{s.get('halts', 0)} ({_pct(s.get('halted_share'), 0)})"
        if s and s.get("halt_reasons"):
            halts += " " + ", ".join(f"{k}×{n}" for k, n in sorted(s["halt_reasons"].items()))
        lines.append(
            f"| {r['name']} | {_pct(m.get('cagr'))} | {_pct(m.get('vol'))} | {_num(m.get('sharpe'))} | "
            f"{_pct(m.get('max_drawdown'))} | {_pct(m.get('cagr_eur'))} | {_pct(m.get('max_drawdown_eur'))} | "
            f"{_num(m.get('turnover'))} | {_num(m.get('transaction_tax'), 0)} | {halts} |")
    for wf in report["walk_forward"]:
        lines += ["", f"Walk-forward « {VARIANT_LABELS[wf['variant']]} » :"]
        for f in wf["folds"]:
            lines.append(f"- {f['start']} → {f['end']} : {label_of(f['params'])} — {f['reason']} — "
                         f"rendement {_pct(f['return'])}")
    lines += ["", "Limites :"] + [f"- {c}" for c in report["caveats"]]
    return "\n".join(lines)


def to_json(report: dict) -> str:
    def default(o):
        if isinstance(o, (date, datetime)):
            return o.isoformat()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(type(o).__name__)
    return json.dumps(report, default=default, indent=1, ensure_ascii=False)
