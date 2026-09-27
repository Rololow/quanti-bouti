"""Data Integrity (README §50.1) : contrôles avant tout modèle.

Chaque événement brut passe par `DataIntegrity.check`, qui renvoie les
événements à transmettre au moteur (0, 1 ou 2 : un trade en quarantaine peut
être libéré en même temps que celui qui le confirme) et les anomalies vues.

Contrôles :

- timestamp dans le futur (au-delà de `max_clock_skew`) ou très en retard ;
- trade de taille négative ;
- saut de prix |log(p / p_prev)| > max(min_jump, jump_k * échelle), où
  l'échelle est la moyenne EWMA des |rendements| tick du symbole :
    * le trade est mis en quarantaine ;
    * le trade suivant le **confirme** (proche du nouveau niveau) -> les deux
      sont acceptés ; si le ratio de prix correspond à un split (2:1, 3:1,
      1:2…), une corporate action probable est signalée ;
    * le trade suivant le **dément** (retour à l'ancien niveau) -> glitch
      rejeté ;
- quote croisée (bid > ask) ou négative : rejet ; spread anormalement large :
  accepté mais signalé ;
- barre incohérente (high < low, open/close hors [low, high]) : rejet ;
- flux figé : un symbole silencieux depuis `stale_after` alors que les autres
  vivent (`stale_symbols`).

Un `DataIntegrityScore` par symbole (EWMA de 1 - sévérité) résume la qualité
récente des données. Tout ne dépend que des timestamps des événements :
le replay reproduit exactement les mêmes décisions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from trading_engine.data.events import (
    BarEvent,
    Event,
    FundamentalEvent,
    MarketEvent,
    NewsEvent,
    QuoteEvent,
    TradeEvent,
)

# Sévérités (0 = information, 1 = donnée rejetée)
INFO, WARNING, REJECT = 0.1, 0.5, 1.0

SPLIT_RATIOS = tuple(sorted({n / d for n in range(1, 11) for d in range(1, 11) if n != d and max(n, d) / min(n, d) >= 1.5}))


@dataclass(frozen=True)
class IntegrityConfig:
    max_clock_skew: float = 2.0          # secondes d'avance tolérées sur received_at
    max_lateness: float = 300.0          # secondes de retard par rapport au dernier événement du symbole
    jump_k: float = 12.0                 # saut = jump_k × |rendement| moyen...
    min_jump: float = 0.02               # ... et au moins 2 %
    warmup_ticks: int = 20               # pas de détection de saut avant N trades
    max_spread: float = 0.05             # spread relatif signalé au-delà (5 %)
    split_tolerance: float = 0.01
    stale_after: float = 300.0           # secondes sans donnée pour un symbole
    score_lambda: float = 0.98


@dataclass(frozen=True)
class DataIssue:
    kind: str
    symbol: str | None
    timestamp: datetime
    severity: float
    detail: str = ""

    def __str__(self) -> str:
        return f"{self.kind} {self.symbol or ''} {self.detail}".strip()


@dataclass
class IntegrityResult:
    accepted: list[Event] = field(default_factory=list)
    issues: list[DataIssue] = field(default_factory=list)


@dataclass
class _SymbolState:
    last_price: float | None = None
    last_timestamp: datetime | None = None
    abs_return: float | None = None      # échelle EWMA des |log-rendements|
    n_ticks: int = 0
    quarantined: TradeEvent | None = None
    score: float = 1.0


class DataIntegrity:
    def __init__(self, config: IntegrityConfig | None = None) -> None:
        self.config = config or IntegrityConfig()
        self._state: dict[str, _SymbolState] = {}
        self.last_event_time: datetime | None = None
        self.counts: dict[str, int] = {}
        self.corporate_actions: dict[str, DataIssue] = {}
        self.withheld = 0   # événements non transmis (rejetés ou en quarantaine)
        self.accepted = 0

    # ----------------------------------------------------------------- public

    def score(self, symbol: str) -> float:
        state = self._state.get(symbol)
        return 1.0 if state is None else state.score

    def scores(self) -> dict[str, float]:
        return {sym: st.score for sym, st in sorted(self._state.items())}

    def global_score(self) -> float:
        return min((st.score for st in self._state.values()), default=1.0)

    def stale_symbols(self, now: datetime | None = None) -> list[str]:
        now = now or self.last_event_time
        if now is None:
            return []
        limit = timedelta(seconds=self.config.stale_after)
        return sorted(
            sym for sym, st in self._state.items()
            if st.last_timestamp is not None and now - st.last_timestamp > limit
        )

    def check(self, event: Event) -> IntegrityResult:
        result = IntegrityResult()
        symbol = event.symbol
        state = self._state.setdefault(symbol, _SymbolState()) if symbol else None

        issue = self._check_timing(event, state)
        if issue is None:
            if isinstance(event, TradeEvent):
                self._check_trade(event, state, result)
                self._finish(result, state, event)
                return result
            if isinstance(event, QuoteEvent):
                issue = self._check_quote(event)
            elif isinstance(event, BarEvent):
                issue = self._check_bar(event)
            elif isinstance(event, FundamentalEvent):
                issue = self._check_fundamental(event)
            elif isinstance(event, NewsEvent):
                issue = self._check_news(event)

        if issue is not None:
            result.issues.append(issue)
        if issue is None or issue.severity < REJECT:
            result.accepted.append(event)
        self._finish(result, state, event)
        return result

    # ----------------------------------------------------------------- checks

    def _issue(self, kind: str, event: Event, severity: float, detail: str = "") -> DataIssue:
        return DataIssue(kind, event.symbol, event.timestamp, severity, detail)

    def _check_timing(self, event: Event, state: _SymbolState | None) -> DataIssue | None:
        cfg = self.config
        # Une donnée fondamentale peut être reçue avant sa publication (base
        # historique chargée à l'avance) : ce n'est pas une anomalie, le store
        # point-in-time ne la rendra visible qu'à `available_at`.
        if not isinstance(event, FundamentalEvent) and \
                event.timestamp - event.received_at > timedelta(seconds=cfg.max_clock_skew):
            return self._issue("FUTURE_TIMESTAMP", event, REJECT,
                               f"{(event.timestamp - event.received_at).total_seconds():.1f}s ahead")
        # Le retard ne concerne que les données de marché : une news ou une donnée
        # fondamentale historique est normalement publiée avant d'être reçue.
        if isinstance(event, MarketEvent) and state is not None and state.last_timestamp is not None:
            late = state.last_timestamp - event.timestamp
            if late > timedelta(seconds=cfg.max_lateness):
                return self._issue("LATE_EVENT", event, REJECT, f"{late.total_seconds():.0f}s late")
        return None

    def _check_trade(self, trade: TradeEvent, state: _SymbolState, result: IntegrityResult) -> None:
        cfg = self.config
        if trade.size < 0:
            result.issues.append(self._issue("INVALID_SIZE", trade, REJECT, f"size={trade.size}"))
            return

        suspect = state.quarantined
        if suspect is not None:
            state.quarantined = None
            to_suspect = abs(math.log(trade.price / suspect.price))
            to_previous = abs(math.log(trade.price / state.last_price))
            if to_suspect <= to_previous:
                # Nouveau niveau confirmé : on accepte le trade en quarantaine.
                ratio = suspect.price / state.last_price
                split = self._split_ratio(ratio)
                if split is not None:
                    issue = self._issue("POSSIBLE_CORPORATE_ACTION", suspect, WARNING,
                                        f"price ratio {ratio:.3f} ≈ {split:.3f}")
                    self.corporate_actions[trade.symbol] = issue
                else:
                    issue = self._issue("CONFIRMED_JUMP", suspect, INFO, f"{math.log(ratio):+.2%}")
                result.issues.append(issue)
                self._accept_trade(suspect, state, result, update_scale=False)
            else:
                result.issues.append(self._issue(
                    "GLITCH", suspect, REJECT,
                    f"{suspect.price} not confirmed (back to {trade.price})"))

        if state.last_price is not None and state.n_ticks >= cfg.warmup_ticks:
            move = abs(math.log(trade.price / state.last_price))
            threshold = max(cfg.min_jump, cfg.jump_k * (state.abs_return or 0.0))
            if move > threshold:
                state.quarantined = trade
                result.issues.append(self._issue("PRICE_JUMP_QUARANTINED", trade, INFO,
                                                 f"{move:.2%} > {threshold:.2%}"))
                return
        self._accept_trade(trade, state, result)

    def _accept_trade(self, trade: TradeEvent, state: _SymbolState, result: IntegrityResult,
                      update_scale: bool = True) -> None:
        if state.last_price is not None and update_scale:
            r = abs(math.log(trade.price / state.last_price))
            state.abs_return = r if state.abs_return is None else 0.99 * state.abs_return + 0.01 * r
        state.last_price = trade.price
        state.n_ticks += 1
        result.accepted.append(trade)

    def _split_ratio(self, ratio: float) -> float | None:
        for split in SPLIT_RATIOS:
            if abs(ratio / split - 1.0) <= self.config.split_tolerance:
                return split
        return None

    def _check_quote(self, quote: QuoteEvent) -> DataIssue | None:
        if quote.bid <= 0 or quote.ask <= 0:
            return self._issue("INVALID_QUOTE", quote, REJECT, f"bid={quote.bid} ask={quote.ask}")
        if quote.bid > quote.ask:
            return self._issue("CROSSED_QUOTE", quote, REJECT, f"bid={quote.bid} > ask={quote.ask}")
        spread = quote.spread / quote.mid
        if spread > self.config.max_spread:
            return self._issue("WIDE_SPREAD", quote, WARNING, f"{spread:.2%}")
        return None

    def check_bar(self, bar: BarEvent) -> DataIssue | None:
        """Cohérence d'une barre seule (utilisé pour l'historique de démarrage)."""
        issue = self._check_bar(bar)
        if issue is not None:
            self.counts[issue.kind] = self.counts.get(issue.kind, 0) + 1
        return issue

    def _check_bar(self, bar: BarEvent) -> DataIssue | None:
        if bar.low <= 0 or bar.high < bar.low or not (
            bar.low <= bar.open <= bar.high and bar.low <= bar.close <= bar.high
        ):
            return self._issue("INCONSISTENT_BAR", bar, REJECT,
                               f"o={bar.open} h={bar.high} l={bar.low} c={bar.close}")
        if bar.volume < 0:
            return self._issue("INCONSISTENT_BAR", bar, REJECT, f"volume={bar.volume}")
        return None

    def _check_fundamental(self, ev: FundamentalEvent) -> DataIssue | None:
        if not math.isfinite(ev.value) or (ev.estimate is not None and not math.isfinite(ev.estimate)):
            return self._issue("INVALID_FUNDAMENTAL", ev, REJECT, f"{ev.name}={ev.value}")
        if not ev.name or not ev.period:
            return self._issue("INVALID_FUNDAMENTAL", ev, REJECT, "missing name or period")
        return None

    def _check_news(self, ev: NewsEvent) -> DataIssue | None:
        if not ev.headline.strip():
            return self._issue("EMPTY_NEWS", ev, REJECT)
        return None

    # ----------------------------------------------------------------- bookkeeping

    def _finish(self, result: IntegrityResult, state: _SymbolState | None, event: Event) -> None:
        if self.last_event_time is None or event.timestamp > self.last_event_time:
            self.last_event_time = event.timestamp
        if state is not None:
            if state.last_timestamp is None or event.timestamp > state.last_timestamp:
                state.last_timestamp = event.timestamp
            severity = max((i.severity for i in result.issues if i.symbol == event.symbol), default=0.0)
            lam = self.config.score_lambda
            state.score = lam * state.score + (1 - lam) * (1.0 - severity)
        for issue in result.issues:
            self.counts[issue.kind] = self.counts.get(issue.kind, 0) + 1
        self.accepted += len(result.accepted)
        if not result.accepted:
            self.withheld += 1
