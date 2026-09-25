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
    print(
        f"\n[{state.timestamp:%Y-%m-%d %H:%M:%S}] "
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


async def _run(config_path: str) -> None:
    engine = Engine(load_config(config_path), reporter=print_state)
    final = await engine.run()
    print("\n=== final ===")
    print_state(final)
    print(f"\nevents={engine.bus.published_count} bars={len(engine.bars)} "
          f"handler_errors={engine.bus.error_count}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    asyncio.run(_run(args.config))


if __name__ == "__main__":
    main()
