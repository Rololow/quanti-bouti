"""News Engine (README §22, §25, §50.7) : un événement, pas N articles.

Chaque news est rattachée à un **cluster** (événement) du même symbole si
elle lui ressemble assez dans une fenêtre de temps :

- même `news_id`, ou
- similarité de Jaccard des mots normalisés >= `similarity`, après
  normalisation des synonymes financiers courants (beats/tops,
  raises/lifts, outlook/forecast/guidance…).

La Phase 11 (IA) remplacera la similarité lexicale par une similarité
sémantique ; la structure (clusters, nouveauté, activité) reste la même.

Features (préfixe news_) :

    news_activity        nombre d'**événements** récents, atténué (demi-vie)
    news_echo            articles par événement récent (reprises)
    news_hours_since     ancienneté du dernier événement
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from trading_engine.data.events import NewsEvent

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "by", "as", "at",
    "is", "are", "be", "its", "it", "from", "after", "amid", "over", "new", "says",
}
SYNONYMS = {
    "tops": "beats", "beat": "beats", "exceeds": "beats", "surpasses": "beats",
    "estimates": "expectations", "consensus": "expectations",
    "lifts": "raises", "raise": "raises", "boosts": "raises", "hikes": "raises", "ups": "raises",
    "forecast": "outlook", "guidance": "outlook", "projections": "outlook",
    "sales": "revenue", "revenues": "revenue", "posts": "reports", "report": "reports",
    "lowers": "cuts", "cut": "cuts", "slashes": "cuts", "reduces": "cuts",
    "investigation": "probe", "inquiry": "probe", "under": "faces",
    "sees": "warns", "softer": "weaker", "weak": "weaker",
    "misses": "miss", "missed": "miss",
}


def normalize(text: str, symbol: str | None = None) -> frozenset[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    sym = (symbol or "").lower()
    return frozenset(SYNONYMS.get(w, w) for w in words if w not in STOPWORDS and w != sym)


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


@dataclass
class NewsCluster:
    cluster_id: str
    symbol: str
    first_seen: datetime
    last_seen: datetime
    tokens: frozenset[str]
    headlines: list[str] = field(default_factory=list)
    providers: set[str] = field(default_factory=set)
    news_ids: set[str] = field(default_factory=set)

    @property
    def articles(self) -> int:
        return len(self.headlines)


class NewsEngine:
    def __init__(
        self,
        *,
        similarity: float = 0.5,
        window: timedelta = timedelta(hours=24),
        activity_half_life: timedelta = timedelta(hours=6),
    ) -> None:
        self.similarity = similarity
        self.window = window
        self.half_life = activity_half_life
        self._clusters: dict[str, list[NewsCluster]] = {}
        self._counter = 0
        self.articles = 0

    def on_news(self, ev: NewsEvent) -> tuple[NewsCluster, bool]:
        """Rattache la news à un événement ; retourne (cluster, nouveau ?)."""
        self.articles += 1
        tokens = normalize(f"{ev.headline}", ev.symbol)
        clusters = self._clusters.setdefault(ev.symbol, [])
        best, best_sim = None, 0.0
        for c in clusters:
            if ev.timestamp - c.last_seen > self.window:
                continue
            sim = 1.0 if ev.news_id and ev.news_id in c.news_ids else jaccard(tokens, c.tokens)
            if sim > best_sim:
                best, best_sim = c, sim
        if best is not None and best_sim >= self.similarity:
            best.headlines.append(ev.headline)
            best.providers.add(ev.provider)
            best.news_ids.add(ev.news_id)
            best.last_seen = max(best.last_seen, ev.timestamp)
            best.tokens = best.tokens | tokens
            return best, False
        self._counter += 1
        cluster = NewsCluster(f"C{self._counter:05d}", ev.symbol, ev.timestamp, ev.timestamp, tokens,
                              [ev.headline], {ev.provider}, {ev.news_id})
        clusters.append(cluster)
        # purge des événements anciens
        horizon = ev.timestamp - 4 * self.window
        self._clusters[ev.symbol] = [c for c in clusters if c.last_seen >= horizon]
        return cluster, True

    def clusters(self, symbol: str) -> list[NewsCluster]:
        return list(self._clusters.get(symbol, []))

    def snapshot(self, symbol: str, as_of: datetime | None) -> dict[str, float]:
        if as_of is None:
            return {}
        recent = [c for c in self._clusters.get(symbol, []) if c.first_seen <= as_of]
        if not recent:
            return {}
        hl = self.half_life.total_seconds()
        weights = [0.5 ** ((as_of - c.first_seen).total_seconds() / hl) for c in recent]
        activity = sum(weights)
        echo = sum(w * c.articles for w, c in zip(weights, recent)) / activity if activity > 0 else 0.0
        last = max(c.first_seen for c in recent)
        return {
            "news_activity": activity,
            "news_echo": echo,
            "news_hours_since": (as_of - last).total_seconds() / 3600,
        }
