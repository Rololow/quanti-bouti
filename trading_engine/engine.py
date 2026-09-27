"""Assemblage de la boucle principale (README §40, milestone §53 — partie Core).

    Feed → EventBus → MarketState / BarBuilder / FeatureEngine / Portfolio → console
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from trading_engine.allocation.allocator import RiskAllocator
from trading_engine.allocation.baseline import BaselineAllocator
from trading_engine.allocation.constraints import ConstraintEngine
from trading_engine.allocation.drift import DriftMonitor, DriftReport
from trading_engine.allocation.types import TargetAllocation
from trading_engine.config import Config
from trading_engine.data.alpaca_feed import AlpacaMarketFeed
from trading_engine.data.bar_builder import BarBuilder
from trading_engine.data.event_bus import EventBus
from trading_engine.data.events import BarEvent, EventType, QuoteEvent, RiskEvent, TradeEvent
from trading_engine.data.market_feed import MarketFeed, SimulatedMarketFeed
from trading_engine.data.market_state import MarketStateStore
from trading_engine.data.replay_feed import ReplayFeed
from trading_engine.features.feature_engine import FeatureEngine
from trading_engine.models.model_engine import ModelEngine
from trading_engine.portfolio.portfolio import Portfolio, PortfolioState
from trading_engine.portfolio.positions import Position
from trading_engine.risk.portfolio_risk import portfolio_vol
from trading_engine.risk.risk_engine import RiskEngine, RiskReport
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
        self.models = ModelEngine(config.models, self.features)
        self.portfolio = Portfolio(
            cash=config.portfolio.cash,
            positions={
                sym: Position(sym, p.quantity, p.avg_price)
                for sym, p in config.portfolio.positions.items()
            },
            target_weights=config.portfolio.target_weights,
        )
        self.risk = RiskEngine(
            self.features, timeframe=config.risk.timeframe,
            shrinkage=config.risk.shrinkage, limits=config.risk.limits,
        )
        alloc = config.allocation
        self.baseline = BaselineAllocator(**vars(alloc.baseline)) if alloc.method == "baseline" else None
        self.allocator = (
            RiskAllocator(alloc.method, target_vol=alloc.target_vol,
                          max_gross=alloc.constraints.max_gross, min_skill=alloc.min_skill)
            if alloc.method not in ("static", "baseline") else None
        )
        self.constraints = ConstraintEngine(alloc.constraints)
        self.drift = DriftMonitor(config.risk.limits.max_drift)
        self.last_allocation: TargetAllocation | None = None
        self.risk_report: RiskReport | None = None
        self._active_breaches: frozenset[str] = frozenset()
        self._interval_due = False

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
        self.risk.on_price(event.symbol, event.price, event.timestamp)
        self.risk.on_value(self.portfolio.total_value(), event.timestamp)
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
        self.models.on_bar(event)
        if event.timeframe == self.config.allocation.rebalance_timeframe:
            self._interval_due = True

    def universe(self) -> list[str]:
        return sorted(set(self.config.feed.symbols) | set(self.features.symbols()))

    async def _on_interval(self) -> None:
        """Une fois toutes les barres de l'intervalle traitées : nouvelle cible
        (sauf allocation statique), puis rapport de risque."""
        self._interval_due = False
        if self.config.allocation.method != "static":
            self._reallocate()
        self.risk_report = self.risk.evaluate(self._raw_snapshot())
        await self._publish_breaches(self.risk_report)

    def _reallocate(self) -> None:
        symbols = self.universe()
        cov = self.risk.covariance(symbols)
        if self.baseline is not None:
            proposal = self.baseline.allocate(self.features, symbols)
            requested = proposal.weights
            attribution = {s: {"baseline": w} for s, w in requested.items()}
        elif cov is not None:
            signals = {s: next(iter(self.models.signals(s)), None) for s in symbols}
            fm = self.models.factor_model
            skill = None if fm is None else fm.monitor.skill
            requested, attribution, _ = self.allocator.allocate(symbols, cov, signals, skill)
        else:
            return  # pas encore de covariance : on garde la cible actuelle

        current = {p.symbol: p.weight for p in self._raw_snapshot().positions}
        constrained = self.constraints.apply(requested, current, cov, symbols)
        for sym, delta in constrained.adjustments.items():
            attribution.setdefault(sym, {})["constraints"] = delta
        weights = constrained.weights
        vol = None
        if cov is not None:
            vol = portfolio_vol(np.array([weights.get(s, 0.0) for s in symbols]), cov)
        self.last_allocation = TargetAllocation(
            weights=weights, attribution=attribution, portfolio_vol=vol,
            binding=constrained.binding,
        )
        self.drift.on_new_target(weights)
        self.portfolio.set_target_weights(weights)

    async def _publish_breaches(self, report: RiskReport) -> None:
        """Publie un RiskEvent quand de nouveaux dépassements apparaissent."""
        keys = frozenset(f"{b.kind}:{b.symbol or 'PORTFOLIO'}" for b in report.breaches)
        new = keys - self._active_breaches
        self._active_breaches = keys
        if new and report.timestamp is not None:
            await self.bus.publish(RiskEvent(
                timestamp=report.timestamp, received_at=report.timestamp, symbol=None,
                source="risk_engine",
                payload={"new": sorted(new), "breaches": [str(b) for b in report.breaches]},
            ))

    def _raw_snapshot(self) -> PortfolioState:
        return self.portfolio.snapshot()

    def drift_report(self) -> DriftReport:
        state = self._raw_snapshot()
        return self.drift.evaluate(
            {p.symbol: p.weight for p in state.positions}, self.portfolio.target_weights
        )

    def snapshot(self) -> PortfolioState:
        report = self.risk.evaluate(self._raw_snapshot())
        vols = {s: p.volatility for s, p in report.positions.items()}
        rcs = {s: p.risk_share for s, p in report.positions.items()}
        return self.portfolio.snapshot(volatilities=vols, risk_contributions=rcs)

    async def run(self) -> PortfolioState:
        try:
            async for event in self.feed:
                if self.recorder is not None:
                    self.recorder.write(event)
                await self.bus.publish(event)
                if self._interval_due:
                    await self._on_interval()
        finally:
            if self.recorder is not None:
                self.recorder.close()
        return self.snapshot()
