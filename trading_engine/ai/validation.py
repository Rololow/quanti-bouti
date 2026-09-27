"""Validation des sorties de l'IA (README §49.5, §50.7).

JSON valide ≠ information correcte ≠ information utile. Chaque analyse
passe par :

    1. schéma      : champs requis, types, énumérations
    2. bornes      : sentiment et impacts dans [-1, 1], scores dans [0, 1],
                     horizon 1-365 jours ; hors bornes -> rejet (pas de
                     correction silencieuse)
    3. cohérence   : news non pertinente -> importance plafonnée ; doublon
                     annoncé qui ne correspond à aucun événement connu -> ignoré
    4. source      : qualité de l'éditeur (table) ou type de source déclaré ;
                     rumeur -> confiance plafonnée
    5. chiffres    : EPS / guidance finis, positifs, et plausibles par rapport
                     à la dernière valeur connue (pas un facteur 10)
    6. confiance   : sous `min_confidence`, l'analyse n'alimente aucun signal

Le résultat garde la liste des problèmes rencontrés, pour l'audit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from trading_engine.ai.schemas import EVENT_TYPES, NEWS_EXTRACTION_SCHEMA, SOURCE_TYPES

# Qualité par type de source déclaré (si l'éditeur n'est pas dans la table).
SOURCE_TYPE_QUALITY = {"primary": 1.0, "major_wire": 0.9, "press": 0.7, "blog_social": 0.3, "unknown": 0.5}

DEFAULT_PROVIDER_QUALITY = {
    "sec": 1.0, "company": 1.0, "businesswire": 0.95, "prnewswire": 0.95, "globenewswire": 0.95,
    "reuters": 0.9, "bloomberg": 0.9, "ap": 0.9, "dow jones": 0.9, "wsj": 0.9, "financial times": 0.9,
    "benzinga": 0.7, "marketwatch": 0.7, "cnbc": 0.7, "barron's": 0.7, "seeking alpha": 0.5,
    "reddit": 0.2, "twitter": 0.2, "x": 0.2, "stocktwits": 0.2,
}

_UNIT = ("sentiment", "revenue_impact", "earnings_impact", "risk_impact")
_PROB = ("relevance", "importance", "confidence")


@dataclass(frozen=True)
class ValidatedAnalysis:
    symbol: str
    news_id: str
    relevant: bool
    event_type: str
    canonical_event: str
    duplicate_of: str | None
    sentiment: float
    relevance: float
    importance: float
    revenue_impact: float
    earnings_impact: float
    risk_impact: float
    horizon_days: int
    is_rumor: bool
    source_type: str
    source_quality: float
    key_facts: tuple[str, ...]
    reported_eps: float | None
    eps_estimate: float | None
    guidance_eps: float | None
    period: str | None
    confidence: float
    issues: tuple[str, ...] = ()

    @property
    def impact(self) -> float:
        """Impact signé : les impacts économiques priment sur le ton."""
        economic = [v for v in (self.earnings_impact, self.revenue_impact) if v != 0]
        base = sum(economic) / len(economic) if economic else self.sentiment
        return max(-1.0, min(1.0, 0.7 * base + 0.3 * self.risk_impact))


@dataclass
class ValidationResult:
    analysis: ValidatedAnalysis | None
    issues: list[str] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return self.analysis is not None


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def provider_quality(provider: str, table: Mapping[str, float]) -> float | None:
    p = (provider or "").strip().lower()
    if not p:
        return None
    for name, q in table.items():
        if p == name or p.startswith(name + " ") or name in p.split():
            return q
    return None


def validate_analysis(
    raw: Any,
    *,
    symbol: str,
    news_id: str,
    provider: str = "",
    known_event_ids: set[str] | frozenset[str] = frozenset(),
    last_eps: float | None = None,
    min_confidence: float = 0.5,
    rumor_confidence_cap: float = 0.5,
    provider_table: Mapping[str, float] = DEFAULT_PROVIDER_QUALITY,
) -> ValidationResult:
    issues: list[str] = []
    # 1. schéma
    if not isinstance(raw, Mapping):
        return ValidationResult(None, ["SCHEMA: not an object"])
    required = NEWS_EXTRACTION_SCHEMA["required"]
    missing = [k for k in required if k not in raw]
    extra = [k for k in raw if k not in NEWS_EXTRACTION_SCHEMA["properties"]]
    if missing:
        return ValidationResult(None, [f"SCHEMA: missing {missing}"])
    if extra:
        issues.append(f"SCHEMA: ignored extra fields {extra}")
    if raw["event_type"] not in EVENT_TYPES or raw["source_type"] not in SOURCE_TYPES:
        return ValidationResult(None, [f"SCHEMA: bad enum {raw['event_type']!r}/{raw['source_type']!r}"])
    if not isinstance(raw["relevant"], bool) or not isinstance(raw["is_rumor"], bool):
        return ValidationResult(None, ["SCHEMA: booleans expected"])
    if not isinstance(raw["key_facts"], list) or not all(isinstance(f, str) for f in raw["key_facts"]):
        return ValidationResult(None, ["SCHEMA: key_facts must be a list of strings"])

    # 2. bornes
    for key in _UNIT + _PROB:
        v = raw[key]
        if not _is_number(v):
            return ValidationResult(None, [f"SCHEMA: {key} not a number"])
        lo = -1.0 if key in _UNIT else 0.0
        if not lo <= v <= 1.0:
            return ValidationResult(None, [f"RANGE: {key}={v} outside [{lo}, 1]"])
    horizon = raw["horizon_days"]
    if not isinstance(horizon, int) or isinstance(horizon, bool) or not 1 <= horizon <= 365:
        return ValidationResult(None, [f"RANGE: horizon_days={horizon!r}"])

    canonical = str(raw["canonical_event"]).strip()[:200]
    if not canonical:
        return ValidationResult(None, ["SCHEMA: empty canonical_event"])

    # 3. cohérence
    importance = float(raw["importance"])
    if not raw["relevant"] and importance > 0.3:
        issues.append("CONSISTENCY: irrelevant news with high importance, capped at 0.3")
        importance = 0.3
    duplicate_of = raw["duplicate_of"]
    if duplicate_of is not None and duplicate_of not in known_event_ids:
        issues.append(f"CONSISTENCY: duplicate_of={duplicate_of!r} is not a known event, ignored")
        duplicate_of = None

    # 4. source
    quality = provider_quality(provider, provider_table)
    if quality is None:
        quality = SOURCE_TYPE_QUALITY[raw["source_type"]]
    confidence = float(raw["confidence"])
    if raw["is_rumor"] and confidence > rumor_confidence_cap:
        issues.append(f"SOURCE: rumor, confidence capped at {rumor_confidence_cap}")
        confidence = rumor_confidence_cap

    # 5. chiffres fondamentaux
    numbers: dict[str, float | None] = {}
    for key in ("reported_eps", "eps_estimate", "guidance_eps"):
        v = raw[key]
        if v is None:
            numbers[key] = None
            continue
        if not _is_number(v) or v <= 0:
            issues.append(f"NUMBERS: {key}={v!r} dropped (not a positive finite number)")
            numbers[key] = None
            continue
        reference = last_eps * (4 if key == "guidance_eps" else 1) if last_eps else None
        if reference and not reference / 10 <= v <= reference * 10:
            issues.append(f"NUMBERS: {key}={v} implausible vs last known {reference:.2f}, dropped")
            numbers[key] = None
            continue
        numbers[key] = float(v)
    period = raw["period"] if isinstance(raw["period"], str) and raw["period"].strip() else None

    analysis = ValidatedAnalysis(
        symbol=symbol, news_id=news_id, relevant=raw["relevant"], event_type=raw["event_type"],
        canonical_event=canonical, duplicate_of=duplicate_of,
        sentiment=float(raw["sentiment"]), relevance=float(raw["relevance"]), importance=importance,
        revenue_impact=float(raw["revenue_impact"]), earnings_impact=float(raw["earnings_impact"]),
        risk_impact=float(raw["risk_impact"]), horizon_days=horizon, is_rumor=raw["is_rumor"],
        source_type=raw["source_type"], source_quality=quality,
        key_facts=tuple(str(f)[:300] for f in raw["key_facts"][:5]),
        reported_eps=numbers["reported_eps"], eps_estimate=numbers["eps_estimate"],
        guidance_eps=numbers["guidance_eps"], period=period, confidence=confidence,
        issues=tuple(issues),
    )
    # 6. confiance
    if confidence < min_confidence:
        return ValidationResult(None, issues + [f"CONFIDENCE: {confidence:.2f} < {min_confidence:.2f}"])
    return ValidationResult(analysis, issues)
