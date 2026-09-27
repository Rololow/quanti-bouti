"""Point d'entrée : `python -m trading_engine.main [--config path]`."""

from __future__ import annotations

import argparse
import asyncio
import logging

from trading_engine.config import DEFAULT_CONFIG_PATH, load_config
from trading_engine.api.server import DashboardServer, export_static
from trading_engine.engine import Engine
from trading_engine.portfolio.portfolio import PortfolioState


def _fmt(value: float | None, spec: str) -> str:
    return "-" if value is None else format(value, spec)


def print_state(state: PortfolioState) -> None:
    ts = "-" if state.timestamp is None else f"{state.timestamp:%Y-%m-%d %H:%M:%S}"
    print(
        f"\n[{ts}] "
        f"value={state.total_value:,.2f} cash={state.cash:,.2f} "
        f"uPnL={state.unrealized_pnl:+,.2f} leverage={state.leverage:.2f}"
    )
    print(f"  {'symbol':<6} {'qty':>8} {'price':>10} {'weight':>7} {'target':>7} "
          f"{'drift':>7} {'pnl':>10} {'vol':>7} {'RC':>6}")
    for p in state.positions:
        print(
            f"  {p.symbol:<6} {p.quantity:>8.2f} {p.price:>10.2f} {p.weight:>7.1%} "
            f"{p.target_weight:>7.1%} {p.drift:>+7.1%} {p.pnl:>+10.2f} "
            f"{_fmt(p.volatility, '.1%'):>7} {_fmt(p.risk_contribution, '.0%'):>6}"
        )


def print_risk(engine: Engine) -> None:
    report = engine.risk.evaluate(engine.portfolio.snapshot())
    print(
        f"  risk: vol={_fmt(report.portfolio_vol, '.1%')} drawdown={report.drawdown:.2%} "
        f"maxDD={report.max_drawdown:.2%} eff.positions={report.effective_positions:.1f} "
        f"eff.bets={_fmt(report.effective_bets, '.1f')} "
        f"div.ratio={_fmt(report.diversification_ratio, '.2f')}"
    )
    for breach in report.breaches:
        print(f"    ! {breach}")
    drift = engine.drift_report()
    print(f"  drift: max={drift.max_abs_drift:.1%} turnover_to_target={drift.turnover_to_target:.1%} "
          f"target_change={_fmt(drift.target_change, '.1%')}")


def print_safety(engine: Engine) -> None:
    status = engine.safety.status
    scores = engine.integrity.scores()
    worst = min(scores.items(), key=lambda kv: kv[1]) if scores else None
    print(f"  safety: {status.state.value} {'; '.join(status.reasons) or ''}".rstrip())
    print(f"  data: accepted={engine.integrity.accepted} withheld={engine.integrity.withheld} "
          f"worst_score={'-' if worst is None else f'{worst[0]} {worst[1]:.2f}'} "
          f"issues={dict(sorted(engine.integrity.counts.items())) or '-'} "
          f"rejected_targets={engine.rejected_targets}")


def print_tax(engine: Engine) -> None:
    cost = engine.estimate_rebalance_tax()
    if cost is None:
        return
    value = engine.portfolio.total_value()
    print(f"  tax ({engine.tax.profile.country}): rejoindre la cible coûterait "
          f"{engine.tax.profile.transaction_tax_name} {cost.transaction_tax:,.2f} + "
          f"plus-values {cost.capital_gains_tax:,.2f} = {cost.total:,.2f} "
          f"({cost.total / value:.3%} du portefeuille)")


def print_decision(engine: Engine) -> None:
    d = engine.last_decision
    if d is None:
        return
    c = d.costs
    print(f"  DECISION {d.decision_id}: {d.action}"
          + (f" à {d.fraction:.0%} du chemin" if d.action == "UREBALANCE" else "")
          + f" | net {d.net_benefit:+.2f} = risque {d.risk_benefit:+.2f} + alpha {d.alpha_benefit:+.2f}"
          f" - coûts {c.total:.2f} - k·σ ({d.uncertainty:.2f}) | urgence {d.urgency:.2f}")
    print(f"    diagnostics: TE={_fmt(d.tracking_error, '.2%')} modèles eff.={_fmt(d.model_agreement, '.2f')} "
          f"fiabilité={_fmt(d.model_reliability, '.2f')} données={_fmt(d.data_quality, '.2f')} "
          f"robustesse={_fmt(d.robustness, '.2f')} safety={d.safety_state}")
    for reason in d.reasons:
        print(f"    - {reason}")
    if d.action == "UREBALANCE":
        for sym, sd in sorted(d.symbols.items()):
            if abs(sd.notional) > 0:
                print(f"    {sym:<6} {sd.current_weight:6.1%} → {sd.execution_weight:6.1%} "
                      f"(cible {sd.target_weight:.1%}) {sd.notional:+,.0f}")


def print_execution(engine: Engine) -> None:
    plan = engine.last_plan
    fb = engine.execution_feedback
    mode = engine.config.execution.mode
    print(f"  execution ({mode}): fills={len(engine.fills)} ordres clos={len(fb.records)} "
          f"confiance={fb.execution_confidence:.2f} calibration fill={engine.fill_model.calibration:.2f} "
          f"η impact={engine.cost_model.eta:.2f} TOB payée={engine.tax.transaction_taxes_paid if engine.tax else 0:.2f}")
    if plan is None:
        return
    print(f"    dernier plan {plan.decision_id}: {len(plan.orders)} ordre(s), "
          f"{len(plan.rejected)} rejeté(s), coût attendu {plan.expected_cost:.2f}")
    for o in plan.orders:
        print(f"    {o.side.upper():<4} {o.symbol:<5} {abs(o.quantity):>6.0f} @ {o.limit_price:.2f} "
              f"agressivité {o.aggressiveness:.1f} durée {int(o.duration.total_seconds() // 60)}min "
              f"fill attendu {_fmt(o.expected_fill, '.0%')} participation {_fmt(o.participation, '.1%')}")
    for r in plan.rejected:
        print(f"    REJETÉ {r.order.symbol}: {'; '.join(r.violations)}")


def print_alerts(engine: Engine, last: int = 5) -> None:
    recent = list(engine.alerts)[-last:]
    if recent:
        print(f"  alertes récentes ({len(engine.alerts)} au total) :")
        for a in recent:
            ts = "-" if a.timestamp is None else f"{a.timestamp:%m-%d %H:%M}"
            print(f"    [{ts}] {a.severity.upper():<8} {a.kind:<20} {a.symbol or '':<5} {a.message}")


def print_allocation(engine: Engine) -> None:
    alloc = engine.last_allocation
    if alloc is None:
        return
    stress = engine.last_stress
    worst = "" if stress is None or stress.worst is None else (
        f" (pire scénario {stress.worst}: {stress.instability[stress.worst]:.1%})")
    print(f"  allocation ({engine.config.allocation.method}): "
          f"ex-ante vol={_fmt(alloc.portfolio_vol, '.1%')} "
          f"robustness={_fmt(alloc.robustness, '.2f')}{worst} "
          f"binding={', '.join(alloc.binding) or '-'}")
    for sym, steps in sorted(alloc.attribution.items()):
        parts = "  ".join(f"{name} {delta:+.1%}" for name, delta in steps.items())
        print(f"    {sym:<6} {alloc.weights.get(sym, 0.0):>6.1%} = {parts}")


def print_features(engine: Engine) -> None:
    shown = ("mom_5m", "mom_30m", "mom_1h", "mom_1d", "z_5m", "vwap_dist")
    print(f"  {'features':<8} " + " ".join(f"{name:>9}" for name in shown))
    for sym in engine.features.symbols():
        feats = engine.features.snapshot(sym)
        print(f"  {sym:<8} " + " ".join(f"{_fmt(feats.get(name), '+.3f'):>9}" for name in shown))
    for tf in engine.config.features.correlation_timeframes or engine.config.bar_timeframes:
        if engine.features.correlation_updates(tf) < 2:
            continue
        symbols, corr = engine.features.correlation(tf)
        pairs = [
            f"{symbols[i]}/{symbols[j]} {corr[i, j]:+.2f}"
            for i in range(len(symbols)) for j in range(i + 1, len(symbols))
        ]
        print(f"  corr {tf}: " + "  ".join(pairs))


def print_qualitative(engine: Engine) -> None:
    now = engine.features.now
    shown = ("fund_eps_surprise", "fund_eps_growth", "fund_guidance_change", "qual_score",
             "news_activity", "news_echo")
    if engine.fundamentals.store.count == 0 and engine.news.articles == 0:
        return
    print(f"  qualitatif: {engine.fundamentals.store.count} faits fondamentaux, "
          f"{engine.news.articles} articles -> "
          f"{sum(len(engine.news.clusters(s)) for s in engine.features.symbols())} événements news")
    print(f"  {'':<8} " + " ".join(f"{name.replace('fund_', '').replace('news_', 'n_'):>14}" for name in shown))
    for sym in engine.features.symbols():
        feats = engine.features.snapshot(sym)
        print(f"  {sym:<8} " + " ".join(f"{_fmt(feats.get(name), '+.3f'):>14}" for name in shown))
    for sym in engine.features.symbols():
        clusters = engine.news.clusters(sym)
        if clusters and now is not None:
            c = clusters[-1]
            print(f"    {sym} dernière news: \"{c.headlines[0]}\" ({c.articles} article(s), "
                  f"{len(c.providers)} source(s))")


def print_models(engine: Engine) -> None:
    for sym in engine.features.symbols():
        regimes = engine.models.regimes(sym)
        parts = [
            f"{h} {'DEGRADED' if st.degraded else st.most_likely} {st.confidence:.0%}"
            for h, st in sorted(regimes.items())
        ]
        sigs = [
            f"{sg.horizon} {sg.mean * 1e4:+.1f}bp ±{sg.std * 1e4:.1f}bp "
            f"(P>0 {sg.prob_positive:.0%}, fiab. {_fmt(sg.reliability, '.2f')})"
            for sg in engine.models.signals(sym)
        ]
        print(f"  {sym:<6} regime: {' | '.join(parts) or '-':<48} signal: {', '.join(sigs) or '-'}")
    for name, model in engine.models.predictors.items():
        mon = model.monitor
        contexts = ", ".join(
            f"{ctx} {_fmt(skill, '+.2f')} (n={n})"
            for ctx, (n, skill) in model.reliability.contexts().items()
        )
        print(f"  predictor {name}: samples={model.regression.n_updates} "
              f"skill={_fmt(mon.skill, '+.3f')} coverage(±1σ)={_fmt(mon.coverage, '.0%')} "
              f"drifts={mon.drift_count} | skill par contexte: {contexts}")
    ens = engine.models.ensemble
    if ens is not None and len(ens.names) > 1:
        corr = ens.correlation()
        pairs = ", ".join(
            f"{ens.names[i]}/{ens.names[j]} {corr[i, j]:+.2f}"
            for i in range(len(ens.names)) for j in range(i + 1, len(ens.names))
        )
        weights = ", ".join(f"{n} {w:.2f}" for n, w in zip(ens.names, ens.weights()))
        print(f"  ensemble: modèles effectifs={ens.effective_models():.2f} "
              f"corr. erreurs: {pairs} | poids: {weights} (n={ens.n})")


def print_feed_status(engine: Engine) -> None:
    status = getattr(engine.feed, "status", None)
    if status is None:
        return
    latency = "-" if status.last_latency is None else f"{status.last_latency.total_seconds() * 1000:.0f}ms"
    print(f"  feed: connected={status.connected} reconnects={status.reconnect_count} "
          f"messages={status.messages_received} latency={latency} last_error={status.last_error}")


async def _run(args: argparse.Namespace) -> None:
    engine: Engine

    def report(state: PortfolioState) -> None:
        if args.quiet:
            return
        print_state(state)
        print_features(engine)
        print_qualitative(engine)
        print_models(engine)
        print_risk(engine)
        print_safety(engine)
        print_tax(engine)
        print_decision(engine)
        print_execution(engine)
        print_alerts(engine)
        print_allocation(engine)
        print_feed_status(engine)

    engine = Engine(load_config(args.config), reporter=report)
    if engine.tax is not None:
        for warning in engine.tax.warnings():
            print(f"[tax] ⚠ {warning}")

    server = None
    if args.dashboard is not None:
        server = DashboardServer(engine, host=args.dashboard_host, port=args.dashboard)
        await server.start()
        print(f"[dashboard] http://{server.host}:{server.port}  (lecture seule)")
    try:
        final = await engine.run()
    finally:
        print(f"\nevents={engine.bus.published_count} bars={len(engine.bars)} "
              f"handler_errors={engine.bus.error_count}")
    print("\n=== final ===")
    args.quiet = False
    report(final)
    if args.export_dashboard:
        export_static(engine, args.export_dashboard)
        print(f"[dashboard] export statique : {args.export_dashboard}")
    if server is not None:
        if args.keep_open:
            print("[dashboard] flux terminé ; le dashboard reste ouvert (Ctrl+C pour quitter)")
            try:
                await asyncio.Event().wait()
            finally:
                await server.stop()
        else:
            await server.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--dashboard", type=int, nargs="?", const=8050, default=None, metavar="PORT",
                        help="sert le dashboard (lecture seule) sur ce port (défaut 8050)")
    parser.add_argument("--dashboard-host", default="127.0.0.1",
                        help="adresse d'écoute (défaut 127.0.0.1 ; pas d'authentification)")
    parser.add_argument("--keep-open", action="store_true",
                        help="garde le dashboard ouvert après la fin du flux")
    parser.add_argument("--export-dashboard", metavar="FICHIER.html",
                        help="écrit un dashboard autonome (état final embarqué)")
    parser.add_argument("--quiet", action="store_true", help="pas de rapport console intermédiaire")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
