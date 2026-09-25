"""Assemblage de la boucle principale (README §37, milestone §49 — partie Core).

    Feed → EventBus → MarketState / BarBuilder / EWMA / Portfolio → console
"""

from __future__ import annotations

from typing import Callable

from trading_engine.config import Config
from trading_engine.data.alpaca_feed import AlpacaMarketFeed
from trading_engine.data.bar_builder import BarBuilder
from trading_engine.data.event_bus import EventBus
from trading_engine.data.events import BarEvent, EventType, QuoteEvent, TradeEvent
from trading_engine.data.market_feed import MarketFeed, SimulatedMarketFeed
from trading_engine.data.market_state import MarketStateStore
from trading_engine.features.volatility import VolatilityBook
from trading_engine.portfolio.portfolio import Portfolio, PortfolioState
from trading_engine.portfolio.positions import Position

TICK = "tick"


class Engine:
    def __init__(
        self,
        config: Config,
        feed: MarketFeed | None = None,
        reporter: Callable[[PortfolioState], None] | None = None,
    ) -> None:
        self.config = config
        self.bus = EventBus()
        self.market_state = MarketStateStore()
        self.bar_builder = BarBuilder(config.bar_timeframes)
        self.volatility = VolatilityBook(config.ewma_lambda)
        self.portfolio = Portfolio(
            cash=config.portfolio.cash,
            positions={
                sym: Position(sym, p.quantity, p.avg_price)
                for sym, p in config.portfolio.positions.items()
            },
            target_weights=config.portfolio.target_weights,
        )
        self.feed = feed or self._default_feed()
        self.reporter = reporter
        self.trade_count = 0
        self.bars: list[BarEvent] = []

        self.bus.subscribe(EventType.TRADE, self._on_trade)
        self.bus.subscribe(EventType.QUOTE, self._on_quote)
        self.bus.subscribe(EventType.BAR, self._on_bar)

    def _default_feed(self) -> MarketFeed:
        feed_cfg = self.config.feed
        if feed_cfg.provider == "alpaca":
            return AlpacaMarketFeed.from_env(
                feed_cfg.symbols,
                max_events=self.config.engine.max_events,
                **vars(feed_cfg.alpaca),
            )
        if feed_cfg.provider != "simulated":
            raise ValueError(f"unknown feed provider {feed_cfg.provider!r}")
        initial = {
            sym: (
                self.config.portfolio.positions[sym].avg_price
                if sym in self.config.portfolio.positions
                else 100.0
            )
            for sym in feed_cfg.symbols
        }
        return SimulatedMarketFeed(
            initial,
            seed=feed_cfg.seed,
            tick_seconds=feed_cfg.tick_seconds,
            max_events=self.config.engine.max_events,
        )

    async def _on_trade(self, event: TradeEvent) -> None:
        self.market_state.update(event)
        self.portfolio.update_price(event.symbol, event.price, event.timestamp)
        self.volatility.update(event.symbol, TICK, event.price)
        for bar in self.bar_builder.on_trade(event):
            await self.bus.publish(bar)

        self.trade_count += 1
        every = self.config.engine.report_every
        if self.reporter is not None and every > 0 and self.trade_count % every == 0:
            self.reporter(self.snapshot())

    def _on_quote(self, event: QuoteEvent) -> None:
        self.market_state.update(event)

    def _on_bar(self, event: BarEvent) -> None:
        self.bars.append(event)
        self.market_state.update(event)
        # Une barre corrigée (trades tardifs) remplace la précédente : ne pas
        # la compter une seconde fois dans la volatilité.
        if not event.payload.get("correction"):
            self.volatility.update(event.symbol, event.timeframe, event.close)

    def snapshot(self) -> PortfolioState:
        vols = {sym: self.volatility.get(sym, TICK) for sym in self.market_state.prices()}
        return self.portfolio.snapshot(volatilities=vols)

    async def run(self) -> PortfolioState:
        async for event in self.feed:
            await self.bus.publish(event)
        return self.snapshot()
