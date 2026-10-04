"""Assemblage de la boucle principale (README §40, milestone §55 — partie Core).

    Feed → EventBus → MarketState / BarBuilder / FeatureEngine / Portfolio → console
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import logging
import math
from collections import deque
from datetime import date, datetime, timedelta, timezone
from typing import Callable

import numpy as np

from trading_engine.alerts.alerts import Alert, AlertEngine
from trading_engine.allocation.allocator import RiskAllocator
from trading_engine.allocation.baseline import BaselineAllocator
from trading_engine.allocation.constraints import ConstraintEngine, fit_gross_after_freeze
from trading_engine.allocation.drift import DriftMonitor, DriftReport
from trading_engine.allocation.types import TargetAllocation
from trading_engine.config import Config, resolve_path
from trading_engine.data.alpaca_feed import AlpacaMarketFeed
from trading_engine.data.bar_builder import BarBuilder
from trading_engine.data.event_bus import EventBus
from trading_engine.data.integrity import REJECT, DataIntegrity, DataIssue
from trading_engine.data.events import (
    AlertEvent,
    BarEvent,
    DecisionEvent,
    EventType,
    QuoteEvent,
    RiskEvent,
    TradeEvent,
)
from trading_engine.decision.rebalance import Decision, DecisionContext, DecisionEngine
from trading_engine.data.market_feed import MarketFeed, SimulatedMarketFeed
from trading_engine.data.market_state import MarketStateStore
from trading_engine.data.replay_feed import ReplayFeed
from trading_engine.data.alpaca_feed import AlpacaNewsFeed
from trading_engine.data.edgar import EdgarFundamentalFeed
from trading_engine.data.event_file_feed import EventFileFeed
from trading_engine.data.events import FundamentalEvent, NewsEvent
from trading_engine.data.merge import ConcurrentFeed, TimeMergedFeed
from trading_engine.data.simulated_qualitative import SimulatedQualitativeFeed
from trading_engine.features.feature_engine import FeatureEngine
from trading_engine.features.fundamentals import FundamentalFeatures
from trading_engine.ai.news_analyzer import ClaudeNewsAnalyzer, SimulatedNewsAnalyzer
from trading_engine.ai.news_intelligence import AIConfig, NewsIntelligence
from trading_engine.data.events import NewsAnalysisEvent
from trading_engine.news.news_engine import NewsEngine
from trading_engine.models.model_engine import ModelEngine
from trading_engine.portfolio.portfolio import Portfolio, PortfolioState
from trading_engine.portfolio.positions import Position
from trading_engine.execution.cost_model import ExecutionCostModel
from trading_engine.execution.feedback import ExecutionFeedback
from trading_engine.execution.fill_model import FillModel
from trading_engine.execution.hard_controls import HardControls
from trading_engine.execution.optimizer import OrderOptimizer
from trading_engine.execution.orders import ExecutionPlan, Fill, RejectedOrder
from trading_engine.execution.alpaca_trading import (
    AlpacaPaperBroker,
    AlpacaTradeUpdatesFeed,
    AlpacaTradingClient,
    client_order_id,
)
from trading_engine.execution.paper_broker import PaperBroker
from trading_engine.data.alpaca_history import AlpacaHistoricalClient
from trading_engine.data.events import CalendarEvent, OrderUpdateEvent, PortfolioEvent
from trading_engine.data.events import FxEvent, TaxLedgerEvent
from trading_engine.tax.fx import FxRates, fetch_ecb_rates
from trading_engine.backtest.dataset import BarDatasetFeed, fx_path as dataset_fx_path
from trading_engine.data.calendar import MarketCalendar
from trading_engine.execution.volume import VolumeTracker
from trading_engine.risk.portfolio_risk import portfolio_vol
from trading_engine.risk.risk_engine import RiskEngine, RiskReport
from trading_engine.robustness.stress import StressReport, stress_test
from trading_engine.safety.invariants import InvariantMonitor
from trading_engine.safety.safety_engine import SafetyEngine, SafetyState, SafetyStatus
from trading_engine.storage.decision_log import DecisionLogWriter
from trading_engine.storage.event_log import EventLogWriter
from trading_engine.tax.profile import load_tax_profile
from trading_engine.timeutils import parse_timeframe, utcnow
from trading_engine.tax.tax_model import RebalanceTaxCost, TaxModel

logger = logging.getLogger(__name__)


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
        self.integrity = DataIntegrity(config.integrity)
        self.data_issues: deque[DataIssue] = deque(maxlen=200)
        self.safety = SafetyEngine(config.safety)
        self.hard_controls = HardControls(config.hard_controls)
        self.rejected_targets = 0
        self.bar_builder = BarBuilder(config.bar_timeframes)
        q = config.qualitative
        self.fundamentals = FundamentalFeatures(
            weights=q.fundamentals.weights, half_life_days=q.fundamentals.half_life_days
        )
        self.news = NewsEngine(
            similarity=q.news.similarity, window=parse_timeframe(q.news.window),
            activity_half_life=parse_timeframe(q.news.activity_half_life),
        )
        self.ai = NewsIntelligence(
            self._build_analyzer(),
            AIConfig(
                fast_effort=config.ai.fast_effort, escalated_effort=config.ai.escalated_effort,
                escalate_below_confidence=config.ai.escalate_below_confidence,
                min_confidence=config.ai.min_confidence, max_calls_per_hour=config.ai.max_calls_per_hour,
                important_threshold=config.ai.important_threshold,
                extract_fundamentals=config.ai.extract_fundamentals,
                simulated_latency=timedelta(seconds=config.ai.simulated_latency_seconds),
            ),
            last_eps=lambda sym, t: getattr(self.fundamentals.store.latest(sym, "eps", t), "value", None),
        )
        self.features = FeatureEngine(
            config.bar_timeframes,
            momentum_horizons=config.features.momentum_horizons,
            mean_reversion_timeframes=config.features.mean_reversion_timeframes,
            zscore_window=config.features.zscore_window,
            correlation_timeframes=config.features.correlation_timeframes,
            lam=config.ewma_lambda,
            fundamentals=self.fundamentals,
            news=self.news,
            ai=self.ai,
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
        self.tax = self._build_tax_model()
        self.risk = RiskEngine(
            self.features, timeframe=config.risk.timeframe,
            shrinkage=config.risk.shrinkage, limits=config.risk.limits,
            min_observations=config.risk.min_observations,
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
        self.last_stress: StressReport | None = None
        ex = config.execution
        self.volume = VolumeTracker(ex.volume_timeframe)
        self.cost_model = ExecutionCostModel(ex.cost)
        self.fill_model = FillModel(ex.max_participation)
        self.optimizer = OrderOptimizer(ex.optimizer, self.cost_model, self.fill_model, self.volume)
        self.execution_feedback = ExecutionFeedback(
            self.fill_model, self.cost_model, learn_impact=ex.learn_impact
        )
        self.broker = PaperBroker(ex.fill_share) if ex.mode == "paper" else None
        # Compte PAPER Alpaca : seulement avec le flux Alpaca live. En replay, les
        # fills et l'état du compte viennent du journal (aucun appel au broker).
        self.remote_broker: AlpacaPaperBroker | None = None
        if ex.mode == "alpaca_paper":
            if config.feed.provider == "alpaca":
                self.remote_broker = AlpacaPaperBroker(AlpacaTradingClient.from_env(),
                                                       time_in_force=ex.time_in_force)
            elif config.feed.provider != "replay":
                raise ValueError("execution.mode alpaca_paper requires feed.provider alpaca (or replay)")
        self._orders_by_cid: dict[str, "OrderProposal"] = {}
        self._injected: list = []            # événements produits par le moteur (sync, rapprochement)
        self.remote_errors = 0
        self.taxes_outside_broker = 0.0      # taxes dues mais non prélevées par le broker
        sc = config.safety
        self.invariants = InvariantMonitor(sc.invariants, cash_tolerance=sc.invariant_cash_tolerance,
                                           gross_tolerance=sc.invariant_gross_tolerance)
        self._active_invariants: set[str] = set()
        # Fills broker appliqués par ordre : (quantité cumulée, valeur cumulée).
        self._applied_fills: dict[str, tuple[float, float]] = {}
        self.ignored_order_updates = 0
        # Compte broker : aucun ordre tant que l'état réel n'a pas été lu (sinon le
        # moteur calculerait ses ordres sur les positions de la config).
        self.account_synced = not (self.remote_broker is not None and ex.sync_portfolio)
        # Séances de marché : reçues comme CalendarEvent (journalisé).
        self.calendar: MarketCalendar | None = None
        self.market_block: str | None = None  # raison de ne pas trader au dernier intervalle
        self._bootstrapped = False
        self.fx: FxRates | None = None
        self._fx_attempt: datetime | None = None
        # Registre fiscal persistant seulement avec un compte broker réel (paper
        # Alpaca) : c'est lui qui garde les positions d'un démarrage à l'autre.
        # Jamais en simulation, en replay ni avec le broker paper local.
        ledger = config.tax.ledger_path
        self.ledger_path = resolve_path(ledger) if ledger and self.remote_broker is not None \
            and self.tax is not None else None
        self.ledger_source: str | None = None
        self.last_plan: ExecutionPlan | None = None
        self.plans: deque[ExecutionPlan] = deque(maxlen=500)
        self.fills: list[Fill] = []
        self.decision_engine = DecisionEngine(config.decision, self.tax, self.cost_model)
        self.last_decision: Decision | None = None
        self.decisions: deque[Decision] = deque(maxlen=500)
        self.alert_engine = AlertEngine(config.alerts)
        self.alerts: deque[Alert] = deque(maxlen=500)
        # Série temporelle par intervalle (dashboard : courbes de valeur, drawdown…).
        self.history: deque[dict] = deque(maxlen=2000)
        self.decision_log = (
            DecisionLogWriter(config.storage.decision_log) if config.storage.decision_log else None
        )
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
        self.bus.subscribe(EventType.FUNDAMENTAL, self._on_fundamental)
        self.bus.subscribe(EventType.NEWS, self._on_news)
        self.bus.subscribe(EventType.NEWS_ANALYSIS, self._on_news_analysis)
        self.bus.subscribe(EventType.ORDER_UPDATE, self._on_order_update)
        self.bus.subscribe(EventType.PORTFOLIO, self._on_portfolio_event)
        self.bus.subscribe(EventType.CALENDAR, self._on_calendar)
        self.bus.subscribe(EventType.FX, self._on_fx)
        self.bus.subscribe(EventType.TAX_LEDGER, self._on_tax_ledger)

    def _build_tax_model(self) -> TaxModel | None:
        if not self.config.tax.profile:
            return None
        profile = load_tax_profile(resolve_path(self.config.tax.profile))
        fx = self.config.fx
        model = TaxModel(
            profile,
            self.config.instruments,
            self.config.tax.step_up_prices,
            portfolio_currency=profile.currency if fx.provider == "off" else fx.portfolio_currency,
        )
        # Les lots fiscaux (registre persistant ou positions initiales) arrivent
        # au démarrage par un TaxLedgerEvent journalisé : voir `_bootstrap`.
        for warning in model.warnings():
            logger.warning("tax: %s", warning)
        return model

    def record_fill(self, symbol: str, quantity: float, price: float, timestamp: datetime,
                    *, tax_in_cash: bool = True) -> None:
        """Applique un fill : position, cash, taxe sur transaction et lots fiscaux.

        `tax_in_cash=False` : fill d'un broker externe dont le cash fait foi. Le
        broker ne prélève pas la taxe (TOB via broker étranger : déclarée et payée
        par l'investisseur) ; elle est comptée à part dans `taxes_outside_broker`.
        """
        self.portfolio.apply_fill(symbol, quantity, price)
        if self.tax is not None:
            charge, _ = self.tax.record_fill(symbol, quantity, price, timestamp)
            if tax_in_cash:
                self.portfolio.cash -= charge.portfolio_amount or 0.0
            else:
                self.taxes_outside_broker += charge.amount      # devise du profil (à déclarer)
            self._save_ledger()

    def estimate_rebalance_tax(self) -> RebalanceTaxCost | None:
        """Coût fiscal estimé pour rejoindre la cible actuelle depuis les poids courants."""
        if self.tax is None:
            return None
        state = self._raw_snapshot()
        trades = {
            p.symbol: (p.target_weight - p.weight) * state.total_value for p in state.positions
        }
        prices = {p.symbol: p.price for p in state.positions}
        when = state.timestamp or datetime.now(timezone.utc)
        return self.tax.rebalance_cost(trades, prices, when)

    def _build_analyzer(self):
        provider = self.config.ai.provider
        if provider == "auto":
            provider = "simulated" if self.config.feed.provider == "simulated" else "none"
        if provider == "simulated":
            return SimulatedNewsAnalyzer()
        if provider == "claude":
            return ClaudeNewsAnalyzer(model=self.config.ai.model,
                                      server_fallbacks=self.config.ai.server_fallbacks)
        return None          # none, ou replay : les analyses viennent du journal

    def _qualitative_provider(self, provider: str) -> str:
        if provider == "auto":
            return "simulated" if self.config.feed.provider == "simulated" else "none"
        return provider

    def _qualitative_feeds(self, market_start: datetime | None, market_end: datetime | None) -> list[MarketFeed]:
        q = self.config.qualitative
        symbols = self.config.feed.symbols
        feeds: list[MarketFeed] = []
        f_provider = self._qualitative_provider(q.fundamentals.provider)
        n_provider = self._qualitative_provider(q.news.provider)
        if "simulated" in (f_provider, n_provider):
            sim = SimulatedQualitativeFeed(
                symbols, start=market_start, until=market_end, seed=self.config.feed.seed,
                earnings_every=parse_timeframe(q.sim_earnings_every) if f_provider == "simulated" else timedelta(days=36500),
                news_every=parse_timeframe(q.sim_news_every) if n_provider == "simulated" else timedelta(days=36500),
                duplicate_probability=q.sim_duplicate_probability,
            )
            feeds.append(sim)
        for provider, cfg in ((f_provider, q.fundamentals), (n_provider, q.news)):
            if provider == "file":
                if not cfg.path:
                    raise ValueError("qualitative source 'file' requires a path")
                feeds.append(EventFileFeed(resolve_path(cfg.path)))
        if f_provider == "edgar":
            feeds.append(EdgarFundamentalFeed(
                q.fundamentals.edgar_ciks, user_agent=q.fundamentals.edgar_user_agent,
                poll_every=q.fundamentals.edgar_poll_every,
            ))
        if n_provider == "alpaca":
            feeds.append(AlpacaNewsFeed.from_env(symbols or ["*"]))
        unknown = {f_provider, n_provider} - {"none", "simulated", "file", "edgar", "alpaca"}
        if unknown:
            raise ValueError(f"unknown qualitative provider(s): {sorted(unknown)}")
        return feeds

    def _default_feed(self) -> MarketFeed:
        feed_cfg = self.config.feed
        if feed_cfg.provider == "alpaca":
            market = AlpacaMarketFeed.from_env(
                feed_cfg.symbols,
                max_events=self.config.engine.max_events,
                **vars(feed_cfg.alpaca),
            )
            extra = self._qualitative_feeds(None, None)
            if self.config.execution.mode == "alpaca_paper":
                key, secret = self.remote_broker.client.credentials()
                extra.append(AlpacaTradeUpdatesFeed(key_id=key, secret_key=secret))
            return ConcurrentFeed([market, *extra]) if extra else market
        if feed_cfg.provider == "replay":
            # Le journal contient déjà tous les événements (marché, news, fondamentaux).
            if not feed_cfg.replay_path:
                raise ValueError("feed.replay_path is required for the replay provider")
            return ReplayFeed(feed_cfg.replay_path, max_events=self.config.engine.max_events)
        if feed_cfg.provider == "dataset":
            if not feed_cfg.dataset_path:
                raise ValueError("feed.dataset_path is required for the dataset provider")
            return BarDatasetFeed(resolve_path(feed_cfg.dataset_path), max_events=self.config.engine.max_events)
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
        market = SimulatedMarketFeed(
            initial,
            seed=feed_cfg.seed,
            tick_seconds=feed_cfg.tick_seconds,
            annual_vol=feed_cfg.annual_vol,
            speed=feed_cfg.sim_speed,
            max_events=self.config.engine.max_events,
        )
        end = None
        if self.config.engine.max_events is not None:
            end = market.clock + market.step * self.config.engine.max_events
        extra = self._qualitative_feeds(market.clock, end)
        return TimeMergedFeed([market, *extra]) if extra else market

    def _on_fundamental(self, event: FundamentalEvent) -> None:
        # Stocké tout de suite, visible seulement à partir de available_at.
        self.fundamentals.store.add(event)

    def _on_news(self, event: NewsEvent) -> None:
        self.news.on_news(event)
        self.ai.submit(event)

    def _on_news_analysis(self, event: NewsAnalysisEvent) -> None:
        _, facts = self.ai.on_analysis(event)
        for fact in facts:
            self.fundamentals.store.add(fact)

    async def _on_trade(self, event: TradeEvent) -> None:
        self.market_state.update(event)
        self.portfolio.update_price(event.symbol, event.price, event.timestamp)
        self.risk.on_price(event.symbol, event.price, event.timestamp)
        value = self.portfolio.total_value()
        self.risk.on_value(value, event.timestamp)
        self.safety.on_value(value, event.timestamp)
        self.features.on_trade(event)
        # L'horloge de marché (timestamp des trades) clôture les barres de
        # tous les symboles, même ceux qui ne tradent pas.
        closed = self.bar_builder.flush(event.timestamp) + self.bar_builder.on_trade(event)
        for bar in sorted(closed, key=lambda b: (b.end, b.timeframe, b.symbol)):
            await self.bus.publish(bar)

        if self.broker is not None:
            update = self.broker.on_trade(event)
            for fill in update.fills:
                self.fills.append(fill)
                self.record_fill(fill.symbol, fill.quantity, fill.price, fill.timestamp)
            for wo in update.closed:
                self.execution_feedback.on_order_closed(wo.order, wo.filled, wo.average_price)

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
        if event.timeframe == self.volume.timeframe and not event.payload.get("correction"):
            self.volume.update(event.symbol, event.volume)
        if event.timeframe == self.config.allocation.rebalance_timeframe:
            self._interval_due = True

    def universe(self) -> list[str]:
        return sorted(set(self.config.feed.symbols) | set(self.features.symbols()))

    async def _on_interval(self) -> None:
        """Une fois toutes les barres de l'intervalle traitées : nouvelle cible
        (sauf allocation statique), puis rapport de risque."""
        self._interval_due = False
        if self.calendar is not None and self.features.now is not None:
            cal = self.config.calendar
            self.market_block = self.calendar.trading_block(
                self.features.now, cal.avoid_open_minutes, cal.avoid_close_minutes)
            if self._calendar_source() is not None and \
                    self.calendar.end < self.features.now.date() + timedelta(days=5):
                await self._load_calendar(self.features.now)
        if (self._fx_source() == "ecb" and self.fx is not None and self.features.now is not None
                and self.fx.last < self.features.now.date() - timedelta(days=1)
                and (self._fx_attempt is None or utcnow() - self._fx_attempt > timedelta(hours=6))):
            await self._load_fx(self.features.now)
        self.risk_report = self.risk.evaluate(self._raw_snapshot())
        await self._publish_breaches(self.risk_report)
        status = self.evaluate_safety()
        if self.config.allocation.method != "static" and status.can_decide:
            self._reallocate(status)
        if not status.can_decide:                    # HALTED : plus aucun ordre en cours
            if self.broker is not None:
                for wo in self.broker.cancel_all():
                    self.execution_feedback.on_order_closed(wo.order, wo.filled, wo.average_price)
            if self.remote_broker is not None:
                await self.remote_broker.cancel_all()
        decision = None
        if self.config.decision.enabled:
            decision = self.decide(status)
            await self._publish_decision(decision)
            if decision.action == "UREBALANCE" and self.config.execution.mode != "off":
                await self.execute(self.plan_execution(decision, status))
        if self.remote_broker is not None and not self.account_synced:
            await self._fetch_account("sync")             # nouvelle tentative
        elif (self.remote_broker is not None and self.config.execution.reconcile
                and not self.remote_broker.working):
            await self._fetch_account("reconcile")
        await self._publish_alerts(status, decision)
        self._record_history(status, decision)
        await self._check_invariants(status.timestamp)

    async def _check_invariants(self, timestamp: datetime | None) -> None:
        """Défense en profondeur : état réel vérifié, HALTED si violation."""
        violations = self.invariants.check(self)
        names = {v.name for v in violations}
        for v in violations:
            if v.name not in self._active_invariants:      # une alerte par apparition
                self.alerts.append(Alert("INVARIANT", None, timestamp, str(v), "critical"))
                logger.error("invariant violated: %s", v)
        self._active_invariants = names
        if violations and self.invariants.mode == "halt" and self.safety.state is not SafetyState.HALTED:
            self.safety.halt("INVARIANT " + ",".join(sorted(names)), timestamp)
            if self.broker is not None:
                for wo in self.broker.cancel_all():
                    self.execution_feedback.on_order_closed(wo.order, wo.filled, wo.average_price)
            if self.remote_broker is not None:
                await self.remote_broker.cancel_all()

    def _record_history(self, status: SafetyStatus, decision: Decision | None) -> None:
        state = self._raw_snapshot()
        report = self.risk_report
        self.history.append({
            "timestamp": state.timestamp,
            "value": state.total_value,
            "cash": state.cash,
            "drawdown": 0.0 if report is None else report.drawdown,
            "portfolio_vol": None if report is None else report.portfolio_vol,
            "leverage": state.leverage,
            "safety": status.state.value,
            "action": None if decision is None else decision.action,
            "weights": {p.symbol: p.weight for p in state.positions},
            "targets": dict(self.portfolio.target_weights),
        })

    def _daily_vols(self, symbols: list[str]) -> dict[str, float]:
        cov = self.risk.covariance(symbols)
        if cov is None:
            return {}
        return {s: math.sqrt(max(cov[i, i], 0.0) / 252) for i, s in enumerate(symbols) if cov[i, i] > 0}

    def _quote(self, symbol: str) -> tuple[float, float] | None:
        ms = self.market_state.get(symbol)
        if ms is None:
            return None
        if ms.bid and ms.ask and ms.ask >= ms.bid:
            return ms.bid, ms.ask
        half = ms.price * self.config.execution.cost.default_spread_bps * 1e-4 / 2
        return ms.price - half, ms.price + half

    def plan_execution(self, decision: Decision, status: SafetyStatus) -> ExecutionPlan:
        """Transforme un UREBALANCE en ordres proposés, contrôlés par les hard controls."""
        symbols = self.universe()
        vols = self._daily_vols(symbols)
        value_total = max(decision.risk_benefit, 0.0) + max(decision.alpha_benefit, 0.0)
        traded = sum(abs(d.notional) for d in decision.symbols.values())
        state = self._raw_snapshot()
        orders, rejected, notes = [], [], []
        # Achats plafonnés au cash disponible (hors cash minimum), sans compter
        # le produit des ventes qui pourraient ne pas être exécutées : pas de
        # levier involontaire. Les ventes passent en premier.
        cash_available = state.cash - self.config.allocation.constraints.min_cash * state.total_value
        by_side = sorted(decision.symbols.items(), key=lambda kv: (kv[1].notional > 0, kv[0]))
        for sym, d in by_side:
            quote = self._quote(sym)
            if abs(d.notional) < 1e-9 or quote is None or decision.timestamp is None:
                continue
            value = value_total * abs(d.notional) / traded if traded else 0.0
            order = self.optimizer.optimize(
                sym, d.notional, value, decision.urgency, bid=quote[0], ask=quote[1],
                daily_vol=vols.get(sym), timestamp=decision.timestamp, decision_id=decision.decision_id,
            )
            if order is None:
                notes.append(f"{sym}: quantité arrondie à zéro")
                continue
            session = None if self.calendar is None else self.calendar.session_at(order.timestamp)
            if session is not None and order.expires_at > session.close:
                order = dataclasses.replace(order, duration=session.close - order.timestamp)
            ms = self.market_state.get(sym)
            recent_volume = self.volume.expected_volume(sym, order.duration)
            # Les plafonds connus (taille d'ordre, participation) sont respectés
            # dès le plan : l'ordre est réduit, le reste sera traité par les
            # décisions suivantes. Les hard controls restent le veto final.
            cap = self.hard_controls.max_order_quantity(
                order.limit_price, state.total_value, recent_volume, order.timestamp
            )
            if abs(order.quantity) > cap:
                qty = float(math.trunc(cap)) if not self.config.execution.optimizer.allow_fractional else cap
                if qty <= 0:
                    notes.append(f"{sym}: plafond d'ordre nul")
                    continue
                notes.append(f"{sym}: ordre réduit de {abs(order.quantity):.0f} à {qty:.0f} (plafonds)")
                order = dataclasses.replace(order, quantity=qty if order.quantity > 0 else -qty)
            if order.quantity > 0:
                affordable = max(0.0, cash_available) / order.limit_price
                if not self.config.execution.optimizer.allow_fractional:
                    affordable = float(math.trunc(affordable))
                if affordable <= 0:
                    notes.append(f"{sym}: achat impossible, cash insuffisant")
                    continue
                if order.quantity > affordable:
                    notes.append(f"{sym}: achat réduit de {order.quantity:.0f} à {affordable:.0f} (cash)")
                    order = dataclasses.replace(order, quantity=affordable)
            verdict = self.hard_controls.check_order(
                order, portfolio_value=state.total_value,
                last_price=None if ms is None else ms.price,
                recent_volume=recent_volume,
                safety_state=status.state,
            )
            if verdict.approved:
                orders.append(order)
                if order.quantity > 0:
                    cash_available -= order.notional
            else:
                rejected.append(RejectedOrder(order, verdict.violations))
                self.safety.record_hard_control_rejection(order.timestamp, "; ".join(verdict.violations))
        plan = ExecutionPlan(
            decision_id=decision.decision_id, timestamp=decision.timestamp, orders=tuple(orders),
            rejected=tuple(rejected), expected_cost=sum(o.expected_cost or 0.0 for o in orders),
            execution_confidence=self.execution_feedback.execution_confidence, notes=tuple(notes),
        )
        self.last_plan = plan
        self.plans.append(plan)
        return plan

    async def execute(self, plan: ExecutionPlan) -> None:
        """Remplace les ordres en cours par ceux du nouveau plan."""
        # Identifiants déterministes : en replay, les mises à jour du journal
        # retrouvent leur ordre à partir des plans recalculés.
        for order in plan.orders:
            self._orders_by_cid[client_order_id(order)] = order
        if self.remote_broker is not None:
            if not self.account_synced:
                if not any(a.kind == "BROKER_UNSYNCED" for a in self.alerts):
                    self.alerts.append(Alert("BROKER_UNSYNCED", None, plan.timestamp,
                                             "compte broker non synchronisé : aucun ordre envoyé", "critical"))
                logger.warning("account not synchronized: plan %s not sent", plan.decision_id)
                return
            await self.remote_broker.cancel_all()
            for order in plan.orders:
                if await self.remote_broker.submit(order) is None:
                    self.remote_errors += 1
                    self.safety.record_hard_control_rejection(
                        order.timestamp, f"BROKER_REJECTED {self.remote_broker.errors[-1]}")
            return
        if self.broker is None:
            return                                   # mode proposals : un humain décide
        for wo in self.broker.cancel_all():
            self.execution_feedback.on_order_closed(wo.order, wo.filled, wo.average_price)
        for order in plan.orders:
            self.broker.submit(order)

    # ------------------------------------------------------------------ compte Alpaca paper

    async def _fetch_account(self, kind: str) -> None:
        """Lit le compte paper ; le résultat entre dans le flux comme un
        PortfolioEvent (journalisé : le replay repart du même état)."""
        client = self.remote_broker.client
        try:
            account = await asyncio.to_thread(client.account)
            positions = await asyncio.to_thread(client.positions)
        except Exception as exc:
            logger.warning("Alpaca account %s failed: %s", kind, exc)
            return
        now = utcnow()
        self._injected.append(PortfolioEvent(
            timestamp=now, received_at=now, symbol=None, source="alpaca_trading",
            payload={
                "kind": kind, "cash": account.cash, "equity": account.equity,
                "status": account.status, "trading_blocked": account.trading_blocked,
                "positions": {p.symbol: [p.quantity, p.avg_price, p.price] for p in positions},
            },
        ))

    def _on_portfolio_event(self, event: PortfolioEvent) -> None:
        data = event.payload
        kind = data.get("kind")
        if kind not in ("sync", "reconcile"):
            return
        if kind == "sync":
            self.account_synced = True
        if data.get("trading_blocked") or data.get("status") not in (None, "ACTIVE"):
            self.safety.halt(f"BROKER_ACCOUNT {data.get('status')} blocked={data.get('trading_blocked')}",
                             event.timestamp)
        positions = {sym: tuple(v) for sym, v in (data.get("positions") or {}).items()}
        engine_qty = {s: p.quantity for s, p in self.portfolio.positions.items() if p.quantity}
        broker_qty = {s: v[0] for s, v in positions.items() if v[0]}
        diffs = {s: broker_qty.get(s, 0.0) - engine_qty.get(s, 0.0)
                 for s in set(engine_qty) | set(broker_qty)
                 if abs(broker_qty.get(s, 0.0) - engine_qty.get(s, 0.0)) > 1e-6}
        cash_diff = float(data["cash"]) - self.portfolio.cash
        if kind == "reconcile" and not diffs and abs(cash_diff) < 1.0:
            return
        if kind == "reconcile":
            self.alerts.append(Alert("RECONCILIATION", None, event.timestamp,
                                     f"positions {diffs or '='} cash {cash_diff:+.2f} : état du broker adopté",
                                     "warning"))
        # Le broker fait foi : positions, prix moyens et cash.
        self.portfolio.cash = float(data["cash"])
        self.portfolio.positions = {
            sym: Position(sym, qty, avg) for sym, (qty, avg, _) in positions.items()
        }
        for sym, (_, _, price) in positions.items():
            if price:
                self.portfolio.update_price(sym, price, event.timestamp)
        if self.tax is not None and self.tax.gains is not None:
            # Les lots du registre (dates et coûts réels) sont gardés ; seuls les
            # écarts avec le compte sont corrigés.
            # - sync : titres en plus -> lot au prix moyen du broker, daté du jour
            #   (date d'achat inconnue) ; titres en moins -> lots retirés sans
            #   plus-value (sortis hors du moteur) ;
            # - rapprochement : fill manqué -> achat / vente au dernier prix.
            adjusted = []
            for sym in sorted(set(positions) | set(self.tax.gains.lots)):
                qty, avg, price = positions.get(sym, (0.0, None, None))
                if kind == "sync":
                    diff = self.tax.align_lots(sym, qty, avg, event.timestamp, realize=False)
                else:
                    diff = self.tax.align_lots(sym, qty, price or self.portfolio.prices.get(sym),
                                               event.timestamp, realize=True)
                if diff:
                    adjusted.append(f"{sym} {diff:+g}")
            if adjusted:
                self.alerts.append(Alert("TAX_LEDGER", None, event.timestamp,
                                         f"lots fiscaux alignés sur le compte : {', '.join(adjusted)} (à vérifier)",
                                         "warning"))
            self._save_ledger()

    def _apply_broker_fill(self, event: OrderUpdateEvent) -> None:
        """Applique la partie **nouvelle** d'un ordre, d'après sa quantité
        cumulée : une mise à jour dupliquée ou arrivée en retard (cumul déjà
        atteint) est ignorée, une mise à jour manquée est rattrapée."""
        cid = event.client_order_id
        done_qty, done_value = self._applied_fills.get(cid, (0.0, 0.0))
        total_qty = event.filled_qty if event.filled_qty > 0 else done_qty + (event.fill_qty or 0.0)
        delta = total_qty - done_qty
        if delta <= 1e-9:
            self.ignored_order_updates += 1
            return
        price = None
        if event.filled_avg_price and event.filled_avg_price > 0:
            price = (total_qty * event.filled_avg_price - done_value) / delta
        if price is None or not math.isfinite(price) or price <= 0:
            price = event.fill_price
        if price is None or not math.isfinite(price) or price <= 0:
            self.ignored_order_updates += 1
            logger.warning("fill without usable price ignored: %s", cid)
            return
        self._applied_fills[cid] = (total_qty, done_value + delta * price)
        signed = delta if event.side == "buy" else -delta
        order = self._orders_by_cid.get(cid)
        self.fills.append(Fill(event.symbol, signed, price, event.timestamp,
                               order.decision_id if order is not None else None))
        self.record_fill(event.symbol, signed, price, event.timestamp, tax_in_cash=False)

    def _on_order_update(self, event: OrderUpdateEvent) -> None:
        if event.update in ("fill", "partial_fill"):
            self._apply_broker_fill(event)
        if event.update == "rejected":
            self.safety.record_hard_control_rejection(event.timestamp, f"BROKER_REJECTED {event.client_order_id}")
        if event.terminal:
            order = self._orders_by_cid.pop(event.client_order_id, None)
            if order is not None:
                signed_filled = event.filled_qty if order.quantity > 0 else -event.filled_qty
                self.execution_feedback.on_order_closed(order, signed_filled, event.filled_avg_price)
        if self.remote_broker is not None:
            self.remote_broker.on_update(event)

    # ------------------------------------------------------------------ calendrier

    def _calendar_source(self) -> str | None:
        """Source du calendrier à charger en live (None : pas de chargement ;
        en replay, le calendrier vient du journal)."""
        mode, provider = self.config.calendar.mode, self.config.feed.provider
        if mode == "off" or provider == "replay":
            return None
        if provider == "alpaca":
            return "alpaca"
        if provider == "dataset":         # backtest sur données réelles : vraies séances
            return "rules"
        return "rules" if mode == "on" else None

    async def _bootstrap(self, now: datetime) -> None:
        """Données de démarrage, injectées comme événements journalisés (le
        replay les relit du journal) : calendrier, taux de change, registre fiscal."""
        self._bootstrapped = True
        if self.config.feed.provider == "replay":
            return
        if self._calendar_source():
            await self._load_calendar(now)
        if self._fx_source():
            await self._load_fx(now)
        await self._drain_injected()          # le registre a besoin des taux
        if self.tax is not None:
            self._load_ledger(now)
        await self._drain_injected()

    # ------------------------------------------------------------------ change

    def _fx_source(self) -> str | None:
        if self.tax is None or not self.tax.needs_fx or self.config.feed.provider == "replay":
            return None
        provider = self.config.fx.provider
        if provider == "auto":
            if self.config.feed.provider == "dataset" and self._dataset_fx_path().exists():
                return "file"             # taux BCE téléchargés avec le dataset
            return "ecb" if self.config.feed.provider == "alpaca" else "fixed"
        return None if provider == "off" else provider

    async def _load_fx(self, now: datetime) -> None:
        cfg = self.config.fx
        base, quote = self.tax.profile.currency, self.tax.portfolio_currency
        rates = None
        self._fx_attempt = utcnow()
        if self._fx_source() == "ecb":
            if base != "EUR":
                raise ValueError(f"ECB rates need EUR as tax currency, got {base}")
            try:
                rates = await asyncio.to_thread(fetch_ecb_rates, quote,
                                                now.date() - timedelta(days=cfg.history_days), now.date())
            except Exception as exc:
                if self.fx is not None:
                    logger.warning("ECB rates refresh failed, keeping previous rates: %s", exc)
                    return
                logger.warning("ECB rates failed, fixed rate %s used: %s", cfg.fixed_rate, exc)
                self.alerts.append(Alert("FX_FALLBACK", None, now,
                                         f"taux BCE indisponibles : {quote}/{base} fixe {cfg.fixed_rate} utilisé",
                                         "warning"))
        if self._fx_source() == "file":
            with open(self._dataset_fx_path(), encoding="utf-8") as fh:
                rates = FxRates.from_payload(json.load(fh))
        if rates is None:
            rates = FxRates.fixed(base, quote, cfg.fixed_rate, now.date() - timedelta(days=cfg.history_days))
        self._injected.append(FxEvent(timestamp=now, received_at=now, symbol=None,
                                      source=f"fx_{rates.source}", payload=rates.to_payload()))

    def _dataset_fx_path(self):
        return dataset_fx_path(resolve_path(self.config.feed.dataset_path or "dataset"))

    def _on_fx(self, event: FxEvent) -> None:
        self.fx = FxRates.from_payload(event.payload)
        if self.tax is not None:
            self.tax.set_fx(self.fx)

    # ------------------------------------------------------------------ registre fiscal

    def _load_ledger(self, now: datetime) -> None:
        """Registre au démarrage : fichier (live Alpaca), sinon vide (le compte
        broker le remplira à la synchronisation), sinon positions initiales."""
        source, ledger = None, None
        if self.ledger_path is not None and self.ledger_path.exists():
            with open(self.ledger_path, encoding="utf-8") as fh:
                ledger, source = json.load(fh), "file"
        elif self.remote_broker is not None and self.config.execution.sync_portfolio:
            ledger, source = self.tax.snapshot(), "empty"
        if ledger is None:
            # Positions initiales de la config : date d'achat inconnue, considérée
            # antérieure à toute taxe (step-up si un prix est fourni) ; coût
            # converti au cours du démarrage.
            for sym, p in self.config.portfolio.positions.items():
                if p.quantity > 0 and self.tax.gains is not None:
                    self.tax.gains.buy(sym, p.quantity, self.tax.to_tax(p.avg_price, now), p.acquired or date.min)
            ledger, source = self.tax.snapshot(), "config"
        self._injected.append(TaxLedgerEvent(timestamp=now, received_at=now, symbol=None,
                                             source="tax_ledger", payload={"source": source, "ledger": ledger}))

    def _on_tax_ledger(self, event: TaxLedgerEvent) -> None:
        if self.tax is not None:
            self.tax.restore(event.payload["ledger"])
            self.ledger_source = event.payload.get("source")

    def _save_ledger(self) -> None:
        """Écriture atomique du registre (fichier temporaire puis remplacement)."""
        if self.ledger_path is None or self.tax is None or self.ledger_source is None:
            return
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.ledger_path.with_name(self.ledger_path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.tax.snapshot(), fh, indent=1)
        os.replace(tmp, self.ledger_path)

    # ------------------------------------------------------------------ calendrier

    async def _load_calendar(self, now: datetime) -> None:
        start = now.date() - timedelta(days=7)
        end = now.date() + timedelta(days=self.config.calendar.horizon_days)
        calendar = None
        if self._calendar_source() == "alpaca":
            try:
                client = self.remote_broker.client if self.remote_broker is not None \
                    else AlpacaTradingClient.from_env()
                calendar = await asyncio.to_thread(client.calendar, start, end)
            except Exception as exc:        # repli : règles NYSE
                logger.warning("Alpaca calendar failed, NYSE rules used: %s", exc)
        if calendar is None:
            calendar = MarketCalendar.from_rules(start, end)
        self._injected.append(CalendarEvent(timestamp=now, received_at=now, symbol=None,
                                            source=f"calendar_{calendar.source}",
                                            payload=calendar.to_payload()))

    def _on_calendar(self, event: CalendarEvent) -> None:
        self.calendar = MarketCalendar.from_payload(event.payload)

    async def _warmup(self) -> None:
        """Historique Alpaca au démarrage (flux live uniquement)."""
        wcfg = self.config.warmup
        if not wcfg.enabled or self.config.feed.provider != "alpaca":
            return
        lookbacks = {tf: d for tf, d in wcfg.lookback_days.items() if tf in self.config.bar_timeframes}
        client = AlpacaHistoricalClient.from_env(data_feed=self.config.feed.alpaca.data_feed)
        for bar in await client.warmup_bars(self.config.feed.symbols, lookbacks):
            await self._ingest(bar)

    def _warmup_bar(self, bar: BarEvent) -> None:
        """Barre historique : initialise l'état, sans décision ni alerte."""
        self.market_state.update(bar)
        self.features.on_bar(bar)
        self.models.on_bar(bar)
        if bar.timeframe == self.volume.timeframe:
            self.volume.update(bar.symbol, bar.volume)
        self.portfolio.update_price(bar.symbol, bar.close, bar.end)

    def decision_context(self, status: SafetyStatus) -> DecisionContext:
        state = self._raw_snapshot()
        symbols = self.universe()
        spreads = {}
        for sym in symbols:
            ms = self.market_state.get(sym)
            if ms is not None and ms.spread is not None and ms.bid and ms.ask:
                spreads[sym] = ms.spread / ((ms.bid + ms.ask) / 2)
        ens = self.models.ensemble
        return DecisionContext(
            market_block=self.market_block,
            timestamp=state.timestamp,
            portfolio_value=state.total_value,
            current={p.symbol: p.weight for p in state.positions},
            target=dict(self.portfolio.target_weights) or None,
            symbols=symbols,
            cov=self.risk.covariance(symbols),
            prices={p.symbol: p.price for p in state.positions},
            spreads=spreads,
            signals={s: next(iter(self.models.signals(s)), None) for s in symbols},
            safety_state=status.state.value,
            frozen=status.frozen_symbols,
            data_scores=self.integrity.scores(),
            robustness=None if self.last_allocation is None else self.last_allocation.robustness,
            portfolio_risk=None if self.risk_report is None else self.risk_report.portfolio_vol,
            model_agreement=None if ens is None else ens.effective_models() / len(ens.names),
            daily_vols=self._daily_vols(symbols),
            adv={s: self.volume.adv(s) for s in symbols},
            execution_confidence=self.execution_feedback.execution_confidence,
        )

    def decide(self, status: SafetyStatus) -> Decision:
        decision = self.decision_engine.decide(self.decision_context(status))
        self.last_decision = decision
        self.decisions.append(decision)
        if self.decision_log is not None:
            self.decision_log.write(decision)
        return decision

    async def _publish_decision(self, decision: Decision) -> None:
        if decision.timestamp is None:
            return
        await self.bus.publish(DecisionEvent(
            timestamp=decision.timestamp, received_at=decision.timestamp, symbol=None,
            source="decision_engine",
            payload={
                "decision_id": decision.decision_id,
                "action": decision.action,
                "fraction": decision.fraction,
                "execution_weights": decision.execution_weights,
                "net_benefit": decision.net_benefit,
                "reasons": list(decision.reasons),
            },
        ))

    def _latest_earnings(self, symbols: list[str]) -> dict[str, tuple[datetime, float | None]]:
        now = self.features.now
        out = {}
        for sym in symbols:
            fact = None if now is None else self.fundamentals.latest_fact(sym, "eps", now)
            if fact is not None:
                surprise = (fact.value - fact.estimate) / abs(fact.estimate) if fact.estimate else None
                out[sym] = (fact.available_at, surprise)
        return out

    def _latest_guidance(self, symbols: list[str]) -> dict[str, tuple[datetime, float]]:
        now = self.features.now
        out = {}
        for sym in symbols:
            hist = [] if now is None else self.fundamentals.store.history(sym, "guidance_eps", now)
            if len(hist) >= 2 and hist[-2].value:
                out[sym] = (hist[-1].available_at, (hist[-1].value - hist[-2].value) / abs(hist[-2].value))
        return out

    async def _publish_alerts(self, status: SafetyStatus, decision: Decision | None) -> None:
        symbols = self.universe()
        cfg = self.config.alerts
        vols = {s: (self.features.volatility.get(s, cfg.vol_fast),
                    self.features.volatility.get(s, cfg.vol_slow)) for s in symbols}
        corr = None
        tf = cfg.correlation_timeframe
        if tf in self.features.covariance_timeframes and self.features.correlation_updates(tf) >= 2:
            corr = self.features.correlation(tf)[1]
        alerts = self.alert_engine.evaluate(
            status.timestamp,
            breaches=list(self.risk_report.breaches) if self.risk_report else [],
            volatilities=vols,
            correlation=corr,
            regimes={s: self.models.regimes(s) for s in symbols},
            signals={s: next(iter(self.models.signals(s)), None) for s in symbols},
            safety_state=status.state.value,
            decision=decision,
            earnings=self._latest_earnings(symbols),
            guidance=self._latest_guidance(symbols),
            important_news=[] if self.features.now is None else self.ai.important_events(self.features.now),
        )
        for alert in alerts:
            self.alerts.append(alert)
            if alert.timestamp is not None:
                await self.bus.publish(AlertEvent(
                    timestamp=alert.timestamp, received_at=alert.timestamp, symbol=alert.symbol,
                    source="alert_engine", kind=alert.kind, severity=alert.severity,
                    message=alert.message,
                ))

    def evaluate_safety(self) -> SafetyStatus:
        now = self.integrity.last_event_time
        report = self.risk_report
        return self.safety.evaluate(
            now,
            data_scores=self.integrity.scores(),
            stale_symbols=self._stale_symbols(now),
            corporate_actions={s: i.timestamp for s, i in self.integrity.corporate_actions.items()},
            risk_breaches=[b.kind for b in report.breaches] if report else (),
            model_health=self.models.health(),
        )

    def _stale_symbols(self, now: datetime | None) -> list[str]:
        if self.calendar is None or now is None:
            return self.integrity.stale_symbols(now)
        session = self.calendar.session_at(now)
        if session is None:
            return []                  # marché fermé : l'absence de données est normale
        # Fraîcheur comptée depuis l'ouverture (pas depuis la clôture de la veille).
        return self.integrity.stale_symbols(now, since=session.open)

    def _reallocate(self, status: SafetyStatus) -> None:
        symbols = self.universe()
        cov = self.risk.covariance(symbols)
        if self.baseline is not None:
            proposal = self.baseline.allocate(self.features, symbols)
            requested = proposal.weights
            attribution = {s: {"baseline": w} for s, w in requested.items()}
        elif cov is not None:
            signals = {s: next(iter(self.models.signals(s)), None) for s in symbols}
            # Crédibilité = fiabilité portée par chaque signal (par contexte).
            requested, attribution, _ = self.allocator.allocate(symbols, cov, signals, None)
        else:
            return  # pas encore de covariance : on garde la cible actuelle

        current = {p.symbol: p.weight for p in self._raw_snapshot().positions}

        # Robustness : une cible instable sous de petites perturbations des
        # estimations est incertaine -> on ne fait qu'une partie du chemin.
        robustness = None
        rob = self.config.robustness
        if rob.enabled and self.allocator is not None and cov is not None:
            self.last_stress = stress_test(
                lambda c, sig: self.allocator.allocate(symbols, c, sig, None)[0],
                symbols, cov, signals, vol_bump=rob.vol_bump,
                include_signals=self.allocator.method == "signal",
            )
            robustness = self.last_stress.score
            if robustness < rob.min_score:
                for sym in set(requested) | set(current):
                    cur = current.get(sym, 0.0)
                    damped = cur + robustness * (requested.get(sym, 0.0) - cur)
                    attribution.setdefault(sym, {})["robustness"] = damped - requested.get(sym, 0.0)
                    requested[sym] = damped
        constrained = self.constraints.apply(requested, current, cov, symbols)
        for sym, delta in constrained.adjustments.items():
            attribution.setdefault(sym, {})["constraints"] = delta
        weights = dict(constrained.weights)

        # Safety : symboles gelés à leur poids courant, pas réduit en DEGRADED.
        if status.state is SafetyState.DEGRADED:
            factor = self.config.safety.degraded_rebalance_factor
            for sym in weights:
                cur = current.get(sym, 0.0)
                safe = cur if sym in status.frozen_symbols else cur + factor * (weights[sym] - cur)
                attribution.setdefault(sym, {})["safety"] = safe - weights[sym]
                weights[sym] = safe
            # Un symbole gelé au-dessus de sa cible ne libère pas son poids :
            # les autres ne doivent pas faire dépasser l'exposition maximale.
            c = self.config.allocation.constraints
            fitted = fit_gross_after_freeze(weights, current, status.frozen_symbols,
                                            min(c.max_gross, 1.0 - c.min_cash))
            for sym, w in fitted.items():
                if w != weights[sym]:
                    attribution.setdefault(sym, {})["safety"] = \
                        attribution.get(sym, {}).get("safety", 0.0) + w - weights[sym]
                    weights[sym] = w

        # Hard controls : veto final, indépendant des modèles.
        verdict = self.hard_controls.validate_targets(weights)
        if not verdict.approved:
            self.rejected_targets += 1
            if status.timestamp is not None:
                self.safety.record_hard_control_rejection(status.timestamp, "; ".join(verdict.violations))
            return  # cible rejetée : on garde la précédente
        vol = None
        if cov is not None:
            vol = portfolio_vol(np.array([weights.get(s, 0.0) for s in symbols]), cov)
        self.last_allocation = TargetAllocation(
            weights=weights, attribution=attribution, portfolio_vol=vol,
            binding=constrained.binding, robustness=robustness,
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

    async def _ingest(self, event) -> None:
        # Le journal garde les données brutes : le replay refait les mêmes
        # contrôles d'intégrité.
        if self.recorder is not None:
            self.recorder.write(event)
        if isinstance(event, BarEvent) and event.payload.get("warmup"):
            # Historique : ordre chronologique par timeframe (le contrôle de
            # retard ne s'applique pas), mais une barre incohérente est rejetée.
            if self.integrity.check_bar(event) is None:
                self._warmup_bar(event)
            return
        if isinstance(event, (OrderUpdateEvent, PortfolioEvent, CalendarEvent, FxEvent, TaxLedgerEvent)):
            # Événements du compte broker et calendrier : pas des données de
            # marché (ils ne doivent pas rafraîchir la fraîcheur d'un symbole).
            await self.bus.publish(event)
            if isinstance(event, (OrderUpdateEvent, PortfolioEvent)):
                await self._check_invariants(event.timestamp)
            return
        checked = self.integrity.check(event)
        for issue in checked.issues:
            self.data_issues.append(issue)
            if issue.severity >= REJECT:
                logger.warning("data rejected: %s", issue)
        for accepted in checked.accepted:
            await self.bus.publish(accepted)
            if self._interval_due:
                await self._on_interval()

    async def _drain_injected(self) -> None:
        while self._injected:
            await self._ingest(self._injected.pop(0))

    async def run(self) -> PortfolioState:
        try:
            if self.config.feed.provider == "alpaca":
                await self._bootstrap(utcnow())
            if self.remote_broker is not None and self.config.execution.sync_portfolio:
                await self._fetch_account("sync")
                await self._drain_injected()
            await self._warmup()
            async for event in self.feed:
                if not self._bootstrapped:          # simulation : horloge du flux
                    await self._bootstrap(event.timestamp)
                # Les analyses IA et les événements produits par le moteur
                # (compte broker) passent d'abord, et sont journalisés : même
                # ordre en live et en replay.
                await self._drain_injected()
                for analysis in self.ai.release(event.received_at):
                    await self._ingest(analysis)
                await self._ingest(event)
                if self.remote_broker is not None:
                    await self.remote_broker.expire(event.timestamp)
            await self.ai.drain()
            for analysis in self.ai.release(None):
                await self._ingest(analysis)
        finally:
            self._save_ledger()
            if self.recorder is not None:
                self.recorder.close()
            if self.decision_log is not None:
                self.decision_log.close()
        return self.snapshot()
