"""État complet du moteur sous forme JSON (README §36) : ce que montre le
dashboard, et ce que n'importe quel client peut lire via l'API.

Lecture seule : rien ici ne modifie le moteur.
"""

from __future__ import annotations

import dataclasses
import json
import math
from datetime import datetime, timedelta
from typing import Any

import numpy as np


def to_jsonable(value: Any) -> Any:
    """Convertit récursivement en types JSON (NaN / inf -> null)."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, timedelta):
        return value.total_seconds()
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: to_jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if hasattr(value, "value") and hasattr(value, "name") and type(value).__module__ != "builtins":
        return to_jsonable(value.value)          # Enum
    if isinstance(value, dict) or hasattr(value, "items"):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value) if isinstance(value, (set, frozenset)) else value
        return [to_jsonable(v) for v in items]
    return str(value)


def _signal(sig) -> dict | None:
    if sig is None:
        return None
    return {
        "horizon": sig.horizon, "mean": sig.mean, "std": sig.std, "t_stat": sig.t_stat,
        "prob_positive": sig.prob_positive, "reliability": sig.reliability,
        "disagreement": sig.disagreement, "n_obs": sig.n_obs, "source": sig.source,
    }


def build_state(engine, *, history: int = 500, decisions: int = 50, alerts: int = 50) -> dict:
    state = engine.snapshot()
    symbols = engine.universe()
    report = engine.risk.evaluate(engine.portfolio.snapshot())
    now = engine.features.now

    positions = []
    for p in state.positions:
        risk = report.positions.get(p.symbol)
        positions.append({
            "symbol": p.symbol, "quantity": p.quantity, "price": p.price, "avg_price": p.avg_price,
            "market_value": p.market_value, "pnl": p.pnl, "realized_pnl": p.realized_pnl,
            "weight": p.weight, "target_weight": p.target_weight, "drift": p.drift,
            "volatility": None if risk is None else risk.volatility,
            "risk_share": None if risk is None else risk.risk_share,
            "drawdown": None if risk is None else risk.drawdown,
            "data_score": engine.integrity.score(p.symbol),
            "frozen": p.symbol in engine.safety.status.frozen_symbols,
        })

    signals = {}
    regimes = {}
    qualitative = {}
    for sym in symbols:
        combined = next(iter(engine.models.signals(sym)), None)
        signals[sym] = {
            "combined": _signal(combined),
            "predictors": {n: _signal(s) for n, s in engine.models.predictor_signals(sym).items()},
        }
        regimes[sym] = {
            h: {"probabilities": st.probabilities, "most_likely": st.most_likely,
                "confidence": st.confidence, "degraded": st.degraded, "fit_zscore": st.fit_zscore}
            for h, st in sorted(engine.models.regimes(sym).items())
        }
        snap = engine.features.snapshot(sym)
        qualitative[sym] = {k: v for k, v in snap.items() if k.startswith(("fund_", "news_", "qual_"))}
        qualitative[sym]["momentum"] = {k: v for k, v in snap.items() if k.startswith("mom_")}

    predictors = {
        name: {"skill": m.monitor.skill, "coverage": m.monitor.coverage, "samples": m.regression.n_updates,
               "drifts": m.monitor.drift_count,
               "contexts": {c: {"n": n, "skill": sk} for c, (n, sk) in m.reliability.contexts().items()}}
        for name, m in engine.models.predictors.items()
    }
    ens = engine.models.ensemble
    ensemble = None if ens is None else {
        "models": ens.names, "weights": ens.weights(), "effective_models": ens.effective_models(),
        "error_correlation": ens.correlation(), "samples": ens.n,
    }

    corr = None
    tf = engine.config.risk.timeframe
    if tf in engine.features.covariance_timeframes and engine.features.correlation_updates(tf) >= 2:
        corr_symbols, matrix = engine.features.correlation(tf)
        corr = {"timeframe": tf, "symbols": corr_symbols, "matrix": matrix}

    news = []
    for sym in symbols:
        for c in engine.news.clusters(sym)[-5:]:
            news.append({"symbol": sym, "cluster_id": c.cluster_id, "first_seen": c.first_seen,
                         "headline": c.headlines[0], "articles": c.articles,
                         "providers": sorted(p for p in c.providers if p)})
    news.sort(key=lambda n: n["first_seen"], reverse=True)

    ai_events = []
    for sym in symbols:
        for e in engine.ai.events(sym)[-5:]:
            ai_events.append({
                "symbol": sym, "event_id": e.event_id, "first_seen": e.first_seen,
                "event_type": e.event_type, "description": e.description, "impact": e.impact,
                "confidence": e.confidence, "importance": e.importance,
                "confirmation": e.confirmation, "articles": len(e.analyses),
                "sources": sorted(e.qualities), "score": e.score(),
            })
    ai_events.sort(key=lambda e: e["first_seen"], reverse=True)

    fb = engine.execution_feedback
    plan = engine.last_plan
    tax_cost = engine.estimate_rebalance_tax()
    status = engine.safety.status

    out = {
        "meta": {
            "now": now, "portfolio_time": state.timestamp,
            "feed": engine.config.feed.provider, "allocation_method": engine.config.allocation.method,
            "execution_mode": engine.config.execution.mode,
            "events": engine.bus.published_count, "handler_errors": engine.bus.error_count,
            "feed_status": getattr(engine.feed, "status", None),
        },
        "safety": {"state": status.state, "reasons": status.reasons, "frozen": status.frozen_symbols,
                   "daily_return": engine.safety.daily_return},
        "portfolio": {
            "total_value": state.total_value, "cash": state.cash, "market_value": state.market_value,
            "unrealized_pnl": state.unrealized_pnl, "realized_pnl": state.realized_pnl,
            "leverage": state.leverage, "portfolio_vol": report.portfolio_vol,
            "drawdown": report.drawdown, "max_drawdown": report.max_drawdown,
            "effective_positions": report.effective_positions, "effective_bets": report.effective_bets,
            "diversification_ratio": report.diversification_ratio,
        },
        "positions": positions,
        "risk": {"breaches": [str(b) for b in report.breaches], "correlation": corr},
        "signals": signals,
        "predictors": predictors,
        "ensemble": ensemble,
        "regimes": regimes,
        "qualitative": qualitative,
        "news": news[:20],
        "ai": {
            "analyzer": None if engine.ai.analyzer is None else engine.ai.analyzer.name,
            "model": engine.config.ai.model if engine.ai.analyzer is not None
                     and engine.ai.analyzer.name == "claude" else None,
            "stats": engine.ai.stats,
            "events": ai_events[:15],
        },
        "allocation": engine.last_allocation,
        "stress": engine.last_stress,
        "drift": engine.drift_report(),
        "decision": engine.last_decision,
        "decisions": list(engine.decisions)[-decisions:],
        "execution": {
            "confidence": fb.execution_confidence, "fill_calibration": engine.fill_model.calibration,
            "impact_eta": engine.cost_model.eta, "orders_closed": len(fb.records),
            "fills": engine.fills[-30:], "plan": plan,
            "working": [wo.order for wo in (engine.broker.working.values() if engine.broker else [])],
        },
        "tax": None if engine.tax is None else {
            "country": engine.tax.profile.country, "verified": engine.tax.profile.verified,
            "transaction_taxes_paid": engine.tax.transaction_taxes_paid,
            "rebalance_cost": None if tax_cost is None else {
                "transaction_tax": tax_cost.transaction_tax, "capital_gains_tax": tax_cost.capital_gains_tax,
                "total": tax_cost.total},
            "warnings": engine.tax.warnings(),
        },
        "alerts": list(engine.alerts)[-alerts:],
        "history": list(engine.history)[-history:],
    }
    return to_jsonable(out)


def dumps_state(engine, **kwargs) -> str:
    return json.dumps(build_state(engine, **kwargs), ensure_ascii=False, separators=(",", ":"))
