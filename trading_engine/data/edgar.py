"""Fondamentaux SEC EDGAR (API XBRL `companyfacts`) — README §19-21.

    https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json

Chaque fait porte la date de dépôt (`filed`). Seule la date est connue, pas
l'heure d'acceptation : par prudence (anti-look-ahead), un fait n'est
considéré disponible que le **lendemain** du dépôt à 00:00 UTC.

La SEC exige un User-Agent identifiant le demandeur (nom + contact) : il
doit être fourni explicitement par l'utilisateur (`user_agent`), aucune
valeur par défaut n'est envoyée.
"""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.request
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, AsyncIterator, Callable, Mapping

from trading_engine.data.events import FundamentalEvent
from trading_engine.data.market_feed import MarketFeed

logger = logging.getLogger(__name__)

URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

# concept XBRL -> nom de métrique du moteur
DEFAULT_CONCEPTS = {
    "Revenues": "revenue",
    "RevenueFromContractWithCustomerExcludingAssessedTax": "revenue",
    "EarningsPerShareDiluted": "eps",
    "NetIncomeLoss": "net_income",
    "OperatingIncomeLoss": "operating_income",
    "NetCashProvidedByUsedInOperatingActivities": "operating_cash_flow",
    "LongTermDebt": "long_term_debt",
}
QUARTERLY_FORMS = {"10-Q", "10-K"}


def available_from_filed(filed: str) -> datetime:
    d = date.fromisoformat(filed) + timedelta(days=1)
    return datetime.combine(d, time(0, 0), tzinfo=timezone.utc)


def parse_companyfacts(
    data: Mapping[str, Any],
    symbol: str,
    concepts: Mapping[str, str] = DEFAULT_CONCEPTS,
) -> list[FundamentalEvent]:
    """Faits trimestriels / annuels -> FundamentalEvent, un par (métrique, période, dépôt)."""
    events: list[FundamentalEvent] = []
    seen: set[tuple[str, str, str]] = set()
    facts = (data.get("facts") or {}).get("us-gaap") or {}
    for concept, metric in concepts.items():
        units = (facts.get(concept) or {}).get("units") or {}
        for unit, rows in units.items():
            for row in rows:
                if row.get("form") not in QUARTERLY_FORMS or "filed" not in row or "val" not in row:
                    continue
                fy, fp = row.get("fy"), row.get("fp")
                period = f"{fy}{fp}" if fy and fp else row.get("end", "")
                key = (metric, period, row["filed"])
                if key in seen:
                    continue
                seen.add(key)
                available = available_from_filed(row["filed"])
                events.append(FundamentalEvent(
                    timestamp=available, received_at=available, symbol=symbol, source="sec_edgar",
                    name=metric, value=float(row["val"]), period=period, available_at=available,
                    unit=unit, payload={"form": row.get("form"), "accn": row.get("accn"),
                                        "period_end": row.get("end"), "concept": concept},
                ))
    events.sort(key=lambda e: (e.available_at, e.name, e.period))
    return events


def _http_fetch(url: str, user_agent: str) -> Mapping[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": user_agent, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


class EdgarFundamentalFeed(MarketFeed):
    """Interroge EDGAR périodiquement et émet les nouveaux faits (reçus maintenant,
    disponibles à `available_at`)."""

    def __init__(
        self,
        ciks: Mapping[str, int],
        *,
        user_agent: str,
        poll_every: float = 3600.0,
        fetch: Callable[[str, str], Mapping[str, Any]] = _http_fetch,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        sleep: Callable[[float], Any] = asyncio.sleep,
        max_polls: int | None = None,
    ) -> None:
        if not user_agent or "@" not in user_agent:
            raise ValueError("SEC EDGAR requires a user_agent with a contact email, e.g. 'Name name@example.com'")
        self.ciks = dict(ciks)
        self.user_agent = user_agent
        self.poll_every = poll_every
        self._fetch = fetch
        self._clock = clock
        self._sleep = sleep
        self.max_polls = max_polls
        self._seen: set[tuple[str, str, str, datetime]] = set()

    async def __aiter__(self) -> AsyncIterator[FundamentalEvent]:
        polls = 0
        while self.max_polls is None or polls < self.max_polls:
            for symbol, cik in sorted(self.ciks.items()):
                try:
                    data = await asyncio.to_thread(self._fetch, URL.format(cik=cik), self.user_agent)
                except Exception as exc:  # réseau : on réessaiera au prochain passage
                    logger.warning("EDGAR fetch failed for %s: %s", symbol, exc)
                    continue
                now = self._clock()
                for ev in parse_companyfacts(data, symbol):
                    key = (ev.symbol, ev.name, ev.period, ev.available_at)
                    if key in self._seen:
                        continue
                    self._seen.add(key)
                    # reçu maintenant ; utilisable seulement à partir de available_at
                    yield FundamentalEvent(**{**vars(ev), "received_at": max(now, ev.timestamp),
                                              "timestamp": ev.timestamp})
            polls += 1
            if self.max_polls is None or polls < self.max_polls:
                await self._sleep(self.poll_every)
