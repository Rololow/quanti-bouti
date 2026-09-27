"""Assemblage de la boucle principale (README §40, milestone §53 — partie Core).

    Feed → EventBus → MarketState / BarBuilder / FeatureEngine / Portfolio → console
"""

from __future__ import annotations

from typing import Callable

from trading_engine.allocation.baseline import BaselineAllocator, TargetAllocation
from trading_engine.config import Config
from trading_engine.data.alpaca_feed import AlpacaMarketFeed
from trading_engine.data.bar_builder import BarBuilder
from trading_engine.data.event_bus import EventBus
from trading_engine.data.events import BarEvent, EventType, QuoteEvent, TradeEvent
from trading_engine.data.market_feed import MarketFeed, SimulatedMarketFeed
from trading_engine.data.market_state import MarketStateStore
from trading_engine.data.replay_feed import ReplayFeed
from trading_engine.features.feature_engine import TICK, FeatureEngine
from trading_engine.portfolio.portfolio import Portfolio, PortfolioState
from trading_engine.portfolio.positions import Position
from trading_engine.storage.event_log import EventLogWriter


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
        self.features = FeatureEngine(
            config.bar_timeframes,
            momentum_horizons=config.features.momentum_horizons,
            mean_reversion_timeframes=config.features.mean_reversion_timeframes,
            zscore_window=config.features.zscore_window,
            correlation_timeframes=config.features.correlation_timeframes,
            lam=config.ewma_lambda,
        )
        self.portfolio = Portfolio(
            cash=config.portfolio.cash,
            positions={
                sym: Position(sym, p.quantity, p.avg_price)
                for sym, p in config.portfolio.positions.items()
            },
            target_weights=config.portfolio.target_weights,
        )
        alloc = config.allocation
        self.allocator = (
            BaselineAllocator(**vars(alloc.baseline)) if alloc.method == "baseline" else None
        )
        self.last_allocation: TargetAllocation | None = None
        self._allocation_due = False

        self.feed = feed or self._default_feed()
        self.recorder = (
            EventLogWriter(config.storage.event_log) if config.storage.event_log else None
        )
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
        if feed_cfg.provider == "replay":
            if not feed_cfg.replay_path:
                raise ValueError("feed.replay_path is required for the replay provider")
            return ReplayFeed(feed_cfg.replay_path, max_events=self.config.engine.max_events)
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
        self.features.on_trade(event)
        # L'horloge de marché (timestamp des trades) clôture les barres de
        # tous les symboles, même ceux qui ne tradent pas.
        closed = self.bar_builder.flush(event.timestamp) + self.bar_builder.on_trade(event)
        for bar in sorted(closed, key=lambda b: (b.end, b.timeframe, b.symbol)):
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
        self.features.on_bar(event)
        if self.allocator is not None and event.timeframe == self.config.allocation.rebalance_timeframe:
            self._allocation_due = True

    def _reallocate(self) -> None:
        """Recalcule la cible une fois toutes les barres de l'intervalle traitées."""
        self._allocation_due = False
        symbols = sorted(set(self.config.feed.symbols) | set(self.features.symbols()))
        self.last_allocation = self.allocator.allocate(self.features, symbols)
        self.portfolio.set_target_weights(self.last_allocation.weights)

    def snapshot(self) -> PortfolioState:
        vols = {sym: self.features.volatility.get(sym, TICK) for sym in self.market_state.prices()}
        return self.portfolio.snapshot(volatilities=vols)

    async def run(self) -> PortfolioState:
        try:
            async for event in self.feed:
                if self.recorder is not None:
                    self.recorder.write(event)
                await self.bus.publish(event)
                if self._allocation_due:
                    self._reallocate()
        finally:
            if self.recorder is not None:
                self.recorder.close()
        return self.snapshot()
