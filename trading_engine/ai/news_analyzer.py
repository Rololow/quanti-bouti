"""Analyseurs de news : texte -> JSON structuré (README §23, §47-48).

- `ClaudeNewsAnalyzer` : API Claude avec sortie structurée (JSON schema) ;
- `SimulatedNewsAnalyzer` : déterministe, sans réseau, pour la simulation
  et les tests (même contrat de sortie).

L'analyseur ne décide rien : il extrait. La validation, le dédoublonnage et
la transformation en signal sont faits ensuite, côté moteur.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, Sequence

from trading_engine.ai.schemas import NEWS_EXTRACTION_SCHEMA, SYSTEM_PROMPT
from trading_engine.data.events import NewsEvent
from trading_engine.news.news_engine import jaccard, normalize

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalysisRequest:
    news: NewsEvent
    known_events: Sequence[tuple[str, str]] = ()   # (id, description canonique) récents du symbole
    last_eps: float | None = None


@dataclass
class AnalyzerOutput:
    raw: Any                       # JSON décodé (non validé)
    model: str
    effort: str | None = None
    usage: dict = field(default_factory=dict)


class AnalysisFailed(Exception):
    pass


class NewsAnalyzer(Protocol):
    name: str

    async def analyze(self, request: AnalysisRequest, effort: str) -> AnalyzerOutput: ...


def build_user_prompt(request: AnalysisRequest) -> str:
    news = request.news
    known = "\n".join(f"- {eid}: {desc}" for eid, desc in request.known_events) or "- (none)"
    return (
        f"Asset: {news.symbol}\n"
        f"Published: {news.timestamp.isoformat()}\n"
        f"Publisher: {news.provider or 'unknown'}\n"
        f"Known recent events for {news.symbol} (id: description):\n{known}\n\n"
        "<article>\n"
        f"Headline: {news.headline}\n"
        f"Summary: {news.summary}\n"
        "</article>"
    )


class ClaudeNewsAnalyzer:
    """Extraction par Claude avec sortie structurée.

    - modèle par défaut : claude-opus-5 ; l'effort fixe le coût (la cascade
      commence en `low` et monte en `high` si nécessaire) ;
    - `fallbacks="default"` (beta server-side-fallback) : si le modèle décline
      la requête, l'API la rejoue sur un modèle de repli dans le même appel ;
    - `stop_reason` vérifié avant toute lecture (refusal, max_tokens).

    Nécessite le paquet `anthropic` et des identifiants (ANTHROPIC_API_KEY ou
    `ant auth login`).
    """

    name = "claude"

    def __init__(
        self,
        client: Any = None,
        *,
        model: str = "claude-opus-5",
        max_tokens: int = 4096,
        server_fallbacks: bool = True,
    ) -> None:
        if client is None:
            import anthropic

            client = anthropic.AsyncAnthropic()
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.server_fallbacks = server_fallbacks

    async def analyze(self, request: AnalysisRequest, effort: str) -> AnalyzerOutput:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_user_prompt(request)}],
            output_config={
                "effort": effort,
                "format": {"type": "json_schema", "schema": NEWS_EXTRACTION_SCHEMA},
            },
        )
        if self.server_fallbacks:
            kwargs.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        response = await self.client.beta.messages.create(**kwargs)
        if response.stop_reason in ("refusal", "max_tokens"):
            raise AnalysisFailed(f"stop_reason={response.stop_reason}")
        text = next((b.text for b in response.content if getattr(b, "type", None) == "text"), None)
        if text is None:
            raise AnalysisFailed("no text block in response")
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise AnalysisFailed(f"invalid JSON: {exc}") from exc
        usage = getattr(response, "usage", None)
        return AnalyzerOutput(
            raw=raw, model=getattr(response, "model", self.model), effort=effort,
            usage={} if usage is None else {
                "input_tokens": getattr(usage, "input_tokens", None),
                "output_tokens": getattr(usage, "output_tokens", None),
            },
        )


KEYWORDS = {
    "earnings": ("earnings", 0.8), "beats": ("earnings", 0.8), "miss": ("earnings", 0.8),
    "outlook": ("guidance", 0.8), "revenue": ("earnings", 0.6), "probe": ("regulation", 0.7),
    "contract": ("product", 0.5), "demand": ("macro", 0.5),
}


class SimulatedNewsAnalyzer:
    """Analyseur déterministe pour la simulation : dérive l'analyse du titre
    (et du ton simulé), et reconnaît les reprises d'un même événement."""

    name = "simulated"

    async def analyze(self, request: AnalysisRequest, effort: str) -> AnalyzerOutput:
        news = request.news
        tokens = normalize(news.headline, news.symbol)
        tone = news.payload.get("tone")
        if tone is None:
            tone = -1 if tokens & {"miss", "cuts", "probe", "weaker", "warns"} else 1
        event_type, importance = "other", 0.3
        for word in sorted(tokens):
            if word in KEYWORDS and KEYWORDS[word][1] > importance:
                event_type, importance = KEYWORDS[word]
        duplicate_of = None
        best = 0.0
        for eid, desc in request.known_events:
            sim = jaccard(tokens, frozenset(desc.split()))
            if sim >= 0.5 and sim > best:
                duplicate_of, best = eid, sim
        raw = {
            "relevant": True, "event_type": event_type,
            "canonical_event": " ".join(sorted(tokens)), "duplicate_of": duplicate_of,
            "sentiment": 0.6 * tone, "relevance": 0.9, "importance": importance,
            "revenue_impact": 0.4 * tone if event_type in ("earnings", "product") else 0.0,
            "earnings_impact": 0.5 * tone if event_type in ("earnings", "guidance") else 0.0,
            "risk_impact": -0.5 if event_type == "regulation" else 0.0,
            "horizon_days": 20 if event_type in ("earnings", "guidance") else 5,
            "is_rumor": False, "source_type": "press", "key_facts": [news.headline],
            "reported_eps": None, "eps_estimate": None, "guidance_eps": None, "period": None,
            "confidence": 0.8 if effort != "high" else 0.9,
        }
        return AnalyzerOutput(raw=raw, model="simulated", effort=effort)
