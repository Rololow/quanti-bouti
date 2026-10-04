"""News Intelligence : de la news brute au signal événementiel (README §22-27, §47-48, §50.7).

    News -> analyse IA (cascade d'effort) -> validation -> événement dédoublonné
         -> score par événement -> features ai_* (+ fondamentaux extraits)

Garde-fous :

- **anti-look-ahead** : l'analyse est un `NewsAnalysisEvent` reçu après la
  latence du modèle ; elle n'est utilisée qu'à partir de ce moment. Les
  chiffres extraits deviennent des faits fondamentaux disponibles à cette
  même date (pas à la publication de la news) ;
- **reproductibilité** : les analyses passent par le journal ; le replay les
  relit sans rappeler le modèle ;
- **coût** : une news déjà vue (même id) n'est pas ré-analysée ; budget
  d'appels par heure ; cascade `low` -> `high` seulement si l'extraction est
  douteuse (validation échouée ou confiance basse) ;
- **double comptage** : un événement = un signal, quel que soit le nombre
  d'articles ; les reprises augmentent la **confirmation** (diversité et
  qualité des sources), pas l'amplitude.

    score_événement = impact × confiance × nouveauté × confirmation
    confirmation    = 1 - Π(1 - qualité_source)       (sources distinctes)
"""

from __future__ import annotations

import asyncio
import dataclasses
import heapq
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Mapping

from trading_engine.ai.news_analyzer import AnalysisFailed, AnalysisRequest, NewsAnalyzer
from trading_engine.ai.validation import DEFAULT_PROVIDER_QUALITY, ValidatedAnalysis, validate_analysis
from trading_engine.data.events import FundamentalEvent, NewsAnalysisEvent, NewsEvent
from trading_engine.news.news_engine import jaccard

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AIConfig:
    fast_effort: str = "low"
    escalated_effort: str = "high"
    escalate_below_confidence: float = 0.6
    min_confidence: float = 0.5
    max_calls_per_hour: int = 120
    known_events_in_prompt: int = 10
    event_window: timedelta = timedelta(hours=48)
    signal_half_life_cap: timedelta = timedelta(days=5)
    similarity: float = 0.5
    important_threshold: float = 0.7
    extract_fundamentals: bool = True
    min_source_quality_for_numbers: float = 0.7
    simulated_latency: timedelta = timedelta(seconds=5)
    provider_quality: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_PROVIDER_QUALITY))


@dataclass
class AIEvent:
    event_id: str
    symbol: str
    first_seen: datetime
    event_type: str
    canonical: frozenset[str]
    description: str
    analyses: list[ValidatedAnalysis] = field(default_factory=list)
    qualities: dict[str, float] = field(default_factory=dict)   # éditeur -> qualité
    alerted: bool = False

    @property
    def confirmation(self) -> float:
        miss = 1.0
        for q in self.qualities.values():
            miss *= 1.0 - max(0.0, min(q, 1.0))
        return 1.0 - miss

    def _mean(self, attr: str) -> float:
        return sum(getattr(a, attr) for a in self.analyses) / len(self.analyses)

    @property
    def impact(self) -> float:
        return sum(a.impact for a in self.analyses) / len(self.analyses)

    @property
    def confidence(self) -> float:
        return self._mean("confidence")

    @property
    def importance(self) -> float:
        return max(a.importance for a in self.analyses)

    @property
    def horizon_days(self) -> int:
        return max(a.horizon_days for a in self.analyses)

    def score(self) -> float:
        # nouveauté = 1 par événement : les reprises n'ajoutent pas d'amplitude
        return self.impact * self.confidence * self.confirmation


def _run_to_completion(coro) -> None:
    """Exécute une coroutine qui ne suspend jamais (analyseur simulé)."""
    try:
        coro.send(None)
    except StopIteration:
        return
    coro.close()
    raise RuntimeError("a simulated analyzer must not await")


class NewsIntelligence:
    def __init__(
        self,
        analyzer: NewsAnalyzer | None,
        config: AIConfig | None = None,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        last_eps: Callable[[str, datetime], float | None] | None = None,
    ) -> None:
        self.analyzer = analyzer
        self.config = config or AIConfig()
        self._clock = clock
        self._last_eps = last_eps or (lambda sym, t: None)
        self._pending: list[tuple] = []                  # tas (received_at, n, event)
        self._tasks: set[asyncio.Task] = set()
        self._seen_news: set[tuple[str, str]] = set()
        self._calls: deque[datetime] = deque()
        self._events: dict[str, list[AIEvent]] = {}
        self._counter = 0
        self._n = 0
        self.stats = {"submitted": 0, "skipped_duplicate": 0, "skipped_budget": 0, "calls": 0,
                      "escalations": 0, "failures": 0, "accepted": 0, "rejected": 0}

    # ------------------------------------------------------------------ analyse

    def known_events(self, symbol: str, as_of: datetime) -> list[tuple[str, str]]:
        recent = [e for e in self._events.get(symbol, []) if as_of - e.first_seen <= self.config.event_window]
        return [(e.event_id, e.description) for e in recent[-self.config.known_events_in_prompt:]]

    def _budget_ok(self, now: datetime) -> bool:
        while self._calls and now - self._calls[0] > timedelta(hours=1):
            self._calls.popleft()
        return len(self._calls) < self.config.max_calls_per_hour

    def submit(self, news: NewsEvent) -> None:
        """Demande l'analyse d'une news (asynchrone pour un vrai modèle)."""
        if self.analyzer is None:
            return
        key = (news.symbol, news.news_id or news.headline)
        if key in self._seen_news:
            self.stats["skipped_duplicate"] += 1
            return
        self._seen_news.add(key)
        simulated = self.analyzer.name == "simulated"
        now = news.received_at if simulated else self._clock()
        if not self._budget_ok(now):
            self.stats["skipped_budget"] += 1
            logger.warning("AI call budget reached: news %s not analysed", news.news_id)
            return
        # Le premier appel est réservé dès maintenant : une rafale de news ne
        # peut pas dépasser le budget avant que les appels ne partent.
        self._calls.append(now)
        self.stats["submitted"] += 1
        request = AnalysisRequest(news, tuple(self.known_events(news.symbol, news.timestamp)),
                                  self._last_eps(news.symbol, news.received_at))
        coro = self._analyze(request, simulated)
        if simulated:
            # Déterministe : exécuté immédiatement (l'analyseur simulé n'attend
            # jamais), disponible après une latence simulée en temps de marché.
            _run_to_completion(coro)
        else:
            task = asyncio.ensure_future(coro)
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def _analyze(self, request: AnalysisRequest, simulated: bool) -> None:
        news = request.news
        cfg = self.config
        output = None
        result = None
        issues: list[str] = []
        escalated = False
        for effort in (cfg.fast_effort, cfg.escalated_effort):
            now = news.received_at if simulated else self._clock()
            if escalated:
                self._calls.append(now)          # le 1er appel a été réservé à la soumission
            self.stats["calls"] += 1
            try:
                output = await self.analyzer.analyze(request, effort)
            except AnalysisFailed as exc:
                issues.append(f"ANALYZER: {exc} (effort {effort})")
                output = None
            except Exception as exc:          # réseau, quota, API : on dégrade, on ne plante pas
                issues.append(f"ANALYZER: {type(exc).__name__}: {exc} (effort {effort})")
                output = None
            if output is not None:
                result = validate_analysis(
                    output.raw, symbol=news.symbol, news_id=news.news_id, provider=news.provider,
                    known_event_ids={eid for eid, _ in request.known_events}, last_eps=request.last_eps,
                    min_confidence=cfg.min_confidence, provider_table=cfg.provider_quality,
                )
                issues.extend(result.issues)
                good = result.accepted and result.analysis.confidence >= cfg.escalate_below_confidence
                if good:
                    break
            if effort == cfg.escalated_effort or not self._budget_ok(now):
                break
            escalated = True
            self.stats["escalations"] += 1

        if output is None:
            self.stats["failures"] += 1
        accepted = result is not None and result.accepted
        self.stats["accepted" if accepted else "rejected"] += 1
        received = news.received_at + cfg.simulated_latency if simulated else self._clock()
        event = NewsAnalysisEvent(
            timestamp=news.timestamp, received_at=max(received, news.received_at),
            symbol=news.symbol, source=f"ai_{self.analyzer.name}", news_id=news.news_id,
            headline=news.headline, model="" if output is None else output.model,
            payload={
                "provider": news.provider,
                "effort": None if output is None else output.effort,
                "escalated": escalated,
                "validated": dataclasses.asdict(result.analysis) if accepted else None,
                "issues": list(issues),
            },
        )
        self._n += 1
        heapq.heappush(self._pending, (event.received_at, self._n, event))

    async def drain(self) -> None:
        """Attend les analyses en cours (fin de flux)."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    def release(self, until: datetime | None) -> list[NewsAnalysisEvent]:
        """Analyses devenues disponibles avant `until` (toutes si None), dans l'ordre."""
        out = []
        while self._pending and (until is None or self._pending[0][0] <= until):
            out.append(heapq.heappop(self._pending)[2])
        return out

    # ------------------------------------------------------------------ événements

    def on_analysis(self, ev: NewsAnalysisEvent) -> tuple[AIEvent | None, list[FundamentalEvent]]:
        """Intègre une analyse (live ou rejouée) ; retourne l'événement et les faits extraits."""
        data = ev.payload.get("validated")
        if not data:
            return None, []
        data = dict(data)
        data["key_facts"] = tuple(data.get("key_facts") or ())
        data["issues"] = tuple(data.get("issues") or ())
        analysis = ValidatedAnalysis(**data)
        if not analysis.relevant:
            return None, []
        events = self._events.setdefault(ev.symbol, [])
        tokens = frozenset(analysis.canonical_event.lower().split())
        target = None
        if analysis.duplicate_of:
            target = next((e for e in events if e.event_id == analysis.duplicate_of), None)
        if target is None:
            best = 0.0
            for e in events:
                if ev.timestamp - e.first_seen > self.config.event_window or e.event_type != analysis.event_type:
                    continue
                sim = jaccard(tokens, e.canonical)
                if sim >= self.config.similarity and sim > best:
                    target, best = e, sim
        if target is None:
            self._counter += 1
            target = AIEvent(f"E{self._counter:05d}", ev.symbol, ev.timestamp, analysis.event_type,
                             tokens, analysis.canonical_event)
            events.append(target)
        target.analyses.append(analysis)
        provider = str(ev.payload.get("provider") or analysis.source_type)
        target.qualities[provider] = max(target.qualities.get(provider, 0.0), analysis.source_quality)
        return target, self._extract_fundamentals(ev, analysis)

    def _extract_fundamentals(self, ev: NewsAnalysisEvent, a: ValidatedAnalysis) -> list[FundamentalEvent]:
        cfg = self.config
        if (not cfg.extract_fundamentals or a.is_rumor or a.period is None
                or a.source_quality < cfg.min_source_quality_for_numbers):
            return []
        out = []
        # Disponible quand l'analyse l'est, pas à la publication de la news.
        available = ev.received_at
        if a.reported_eps is not None:
            out.append(FundamentalEvent(
                timestamp=available, received_at=available, symbol=ev.symbol, source="ai_extraction",
                name="eps", value=a.reported_eps, estimate=a.eps_estimate, period=a.period,
                available_at=available, payload={"news_id": ev.news_id}))
        if a.guidance_eps is not None:
            out.append(FundamentalEvent(
                timestamp=available, received_at=available, symbol=ev.symbol, source="ai_extraction",
                name="guidance_eps", value=a.guidance_eps, period=a.period, available_at=available,
                payload={"news_id": ev.news_id}))
        return out

    def events(self, symbol: str) -> list[AIEvent]:
        return list(self._events.get(symbol, []))

    def snapshot(self, symbol: str, as_of: datetime | None) -> dict[str, float]:
        """Features ai_* à `as_of` : score événementiel atténué, importance récente."""
        if as_of is None:
            return {}
        events = [e for e in self._events.get(symbol, []) if e.first_seen <= as_of]
        if not events:
            return {}
        score = importance = 0.0
        for e in events:
            half_life = min(timedelta(days=e.horizon_days), self.config.signal_half_life_cap)
            decay = 0.5 ** ((as_of - e.first_seen) / half_life)
            score += e.score() * decay
            importance = max(importance, e.importance * decay)
        return {"ai_news_score": score, "ai_importance": importance,
                "ai_events": float(sum(1 for e in events if as_of - e.first_seen <= timedelta(days=1)))}

    def important_events(self, as_of: datetime) -> list[AIEvent]:
        """Événements importants, confiants et récents pas encore signalés (une seule alerte chacun)."""
        out = []
        for events in self._events.values():
            for e in events:
                if (not e.alerted and e.importance >= self.config.important_threshold
                        and e.confidence >= self.config.important_threshold
                        and as_of - e.first_seen <= timedelta(days=1)):
                    e.alerted = True
                    out.append(e)
        return sorted(out, key=lambda e: (e.first_seen, e.symbol))
