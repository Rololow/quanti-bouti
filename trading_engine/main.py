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
          f"{'drift':>7} {'pnl':>10} {'vol(tick)':>10}")
    for p in state.positions:
        print(
            f"  {p.symbol:<6} {p.quantity:>8.2f} {p.price:>10.2f} {p.weight:>7.1%} "
            f"{p.target_weight:>7.1%} {p.drift:>+7.1%} {p.pnl:>+10.2f} "
            f"{_fmt(p.volatility, '.6f'):>10}"
        )


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
