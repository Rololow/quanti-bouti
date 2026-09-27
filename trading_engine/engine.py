"""Assemblage de la boucle principale (README §40, milestone §55 — partie Core).

    Feed → EventBus → MarketState / BarBuilder / FeatureEngine / Portfolio → console
"""

from __future__ import annotations

import dataclasses
import logging
import math
from collections import deque
from datetime import date, datetime, timezone
from typing import Callable

import numpy as np

from trading_engine.alerts.alerts import Alert, AlertEngine
from trading_engine.allocation.allocator import RiskAllocator
from trading_engine.allocation.baseline import BaselineAllocator
from trading_engine.allocation.constraints import ConstraintEngine
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
from trading_engine.features.feature_engine import FeatureEngine
from trading_engine.models.model_engine import ModelEngine
from trading_engine.portfolio.portfolio import Portfolio, PortfolioState
from trading_engine.portfolio.positions import Position
from trading_engine.execution.cost_model import ExecutionCostModel
from trading_engine.execution.feedback import ExecutionFeedback
from trading_engine.execution.fill_model import FillModel
from trading_engine.execution.hard_controls import HardControls
from trading_engine.execution.optimizer import OrderOptimizer
from trading_engine.execution.orders import ExecutionPlan, Fill, RejectedOrder
from trading_engine.execution.paper_broker import PaperBroker
from trading_engine.execution.volume import VolumeTracker
from trading_engine.risk.portfolio_risk import portfolio_vol
from trading_engine.risk.risk_engine import RiskEngine, RiskReport
from trading_engine.robustness.stress import StressReport, stress_test
from trading_engine.safety.safety_engine import SafetyEngine, SafetyState, SafetyStatus
from trading_engine.storage.decision_log import DecisionLogWriter
from trading_engine.storage.event_log import EventLogWriter
from trading_engine.tax.profile import load_tax_profile
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
        self.last_plan: ExecutionPlan | None = None
        self.plans: deque[ExecutionPlan] = deque(maxlen=500)
        self.fills: list[Fill] = []
        self.decision_engine = DecisionEngine(config.decision, self.tax, self.cost_model)
        self.last_decision: Decision | None = None
        self.decisions: deque[Decision] = deque(maxlen=500)
        self.alert_engine = AlertEngine(config.alerts)
        self.alerts: deque[Alert] = deque(maxlen=500)
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

    def _build_tax_model(self) -> TaxModel | None:
        if not self.config.tax.profile:
            return None
        model = TaxModel(
            load_tax_profile(resolve_path(self.config.tax.profile)),
            self.config.instruments,
            self.config.tax.step_up_prices,
        )
        # Positions initiales : lots fiscaux à leur prix moyen. Date inconnue :
        # considérée antérieure à toute taxe (step-up si un prix est fourni).
        for sym, p in self.config.portfolio.positions.items():
            if p.quantity > 0 and model.gains is not None:
                model.gains.buy(sym, p.quantity, p.avg_price, p.acquired or date.min)
        for warning in model.warnings():
            logger.warning("tax: %s", warning)
        return model

    def record_fill(self, symbol: str, quantity: float, price: float, timestamp: datetime) -> None:
        """Applique un fill : position, cash, taxe sur transaction et lots fiscaux."""
        self.portfolio.apply_fill(symbol, quantity, price)
        if self.tax is not None:
            charge, _ = self.tax.record_fill(symbol, quantity, price, timestamp)
            self.portfolio.cash -= charge.amount

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
            annual_vol=feed_cfg.annual_vol,
            max_events=self.config.engine.max_events,
        )

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
        self.risk_report = self.risk.evaluate(self._raw_snapshot())
        await self._publish_breaches(self.risk_report)
        status = self.evaluate_safety()
        if self.config.allocation.method != "static" and status.can_decide:
            self._reallocate(status)
        if not status.can_decide and self.broker is not None:
            for wo in self.broker.cancel_all():       # HALTED : plus aucun ordre en cours
                self.execution_feedback.on_order_closed(wo.order, wo.filled, wo.average_price)
        decision = None
        if self.config.decision.enabled:
            decision = self.decide(status)
            await self._publish_decision(decision)
            if decision.action == "UREBALANCE" and self.config.execution.mode != "off":
                self.execute(self.plan_execution(decision, status))
        await self._publish_alerts(status, decision)

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

    def execute(self, plan: ExecutionPlan) -> None:
        """Paper : remplace les ordres en cours par ceux du nouveau plan."""
        if self.broker is None:
            return                                   # mode proposals : un humain décide
        for wo in self.broker.cancel_all():
            self.execution_feedback.on_order_closed(wo.order, wo.filled, wo.average_price)
        for order in plan.orders:
            self.broker.submit(order)

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
            stale_symbols=self.integrity.stale_symbols(now),
            corporate_actions={s: i.timestamp for s, i in self.integrity.corporate_actions.items()},
            risk_breaches=[b.kind for b in report.breaches] if report else (),
            model_health=self.models.health(),
        )

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

    async def run(self) -> PortfolioState:
        try:
            async for event in self.feed:
                # Le journal garde les données brutes : le replay refait les
                # mêmes contrôles d'intégrité.
                if self.recorder is not None:
                    self.recorder.write(event)
                checked = self.integrity.check(event)
                for issue in checked.issues:
                    self.data_issues.append(issue)
                    if issue.severity >= REJECT:
                        logger.warning("data rejected: %s", issue)
                for accepted in checked.accepted:
                    await self.bus.publish(accepted)
                    if self._interval_due:
                        await self._on_interval()
        finally:
            if self.recorder is not None:
                self.recorder.close()
            if self.decision_log is not None:
                self.decision_log.close()
        return self.snapshot()
