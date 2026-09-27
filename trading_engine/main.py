"""Point d'entrée : `python -m trading_engine.main [--config path]`."""

from __future__ import annotations

import argparse
import asyncio
import logging

from trading_engine.config import DEFAULT_CONFIG_PATH, load_config
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


def print_allocation(engine: Engine) -> None:
    alloc = engine.last_allocation
    if alloc is None:
        return
    print(f"  allocation ({engine.config.allocation.method}): "
          f"ex-ante vol={_fmt(alloc.portfolio_vol, '.1%')} "
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


def print_models(engine: Engine) -> None:
    for sym in engine.features.symbols():
        regimes = engine.models.regimes(sym)
        parts = [f"{h} {st.most_likely} {st.confidence:.0%}" for h, st in sorted(regimes.items())]
        sigs = [
            f"{sg.horizon} {sg.mean * 1e4:+.1f}bp ±{sg.std * 1e4:.1f}bp (P>0 {sg.prob_positive:.0%})"
            for sg in engine.models.signals(sym)
        ]
        print(f"  {sym:<6} regime: {' | '.join(parts) or '-':<48} signal: {', '.join(sigs) or '-'}")
    fm = engine.models.factor_model
    if fm is not None:
        mon = fm.monitor
        skill = "-" if mon.skill is None else f"{mon.skill:+.3f}"
        coverage = "-" if mon.coverage is None else f"{mon.coverage:.0%}"
        print(f"  factor model: samples={fm.regression.n_updates} skill={skill} "
              f"coverage(±1σ)={coverage} drifts={mon.drift_count}")


def print_feed_status(engine: Engine) -> None:
    status = getattr(engine.feed, "status", None)
    if status is None:
        return
    latency = "-" if status.last_latency is None else f"{status.last_latency.total_seconds() * 1000:.0f}ms"
    print(f"  feed: connected={status.connected} reconnects={status.reconnect_count} "
          f"messages={status.messages_received} latency={latency} last_error={status.last_error}")


async def _run(config_path: str) -> None:
    engine: Engine

    def report(state: PortfolioState) -> None:
        print_state(state)
        print_features(engine)
        print_models(engine)
        print_risk(engine)
        print_allocation(engine)
        print_feed_status(engine)

    engine = Engine(load_config(config_path), reporter=report)
    try:
        final = await engine.run()
    finally:
        print(f"\nevents={engine.bus.published_count} bars={len(engine.bars)} "
              f"handler_errors={engine.bus.error_count}")
    print("\n=== final ===")
    report(final)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        asyncio.run(_run(args.config))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
