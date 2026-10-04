"""Taux de change pour la fiscalité (README §49 : taxes dans la devise du pays).

Le compte (Alpaca) est en USD, l'impôt belge en EUR. En Belgique, valeur
d'acquisition et prix de vente sont convertis au cours du jour de chaque
opération : la plus-value taxable inclut donc l'effet de change.

Source : taux de référence de la BCE (publiés vers 16h CET les jours ouvrés).
Pour une date donnée, on prend le dernier taux publié ce jour-là ou avant
(week-ends et fériés : taux du dernier jour ouvré).

Les taux entrent dans le moteur comme un `FxEvent` journalisé : le replay
utilise exactement les taux vus en live.
"""

from __future__ import annotations

import bisect
import csv
import io
import logging
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from typing import Callable, Mapping

logger = logging.getLogger(__name__)

ECB_URL = "https://data-api.ecb.europa.eu/service/data/EXR/D.{quote}.EUR.SP00.A"


class FxRates:
    """Cours `quote` par unité de `base` (BCE : USD par EUR), datés."""

    def __init__(self, base: str, quote: str, rates: Mapping[date, float], source: str = "unknown") -> None:
        if any(not (v > 0) for v in rates.values()):
            raise ValueError("FX rates must be positive")
        self.base, self.quote, self.source = base, quote, source
        self._days = sorted(rates)
        self._values = [float(rates[d]) for d in self._days]
        self.before_first = 0          # demandes antérieures au premier taux (taux le plus ancien utilisé)

    def __len__(self) -> int:
        return len(self._days)

    @property
    def first(self) -> date | None:
        return self._days[0] if self._days else None

    @property
    def last(self) -> date | None:
        return self._days[-1] if self._days else None

    def rate(self, when: date | datetime) -> float:
        """Unités de `quote` pour 1 `base` au jour `when`."""
        if not self._days:
            raise LookupError(f"no {self.base}/{self.quote} rate available")
        day = when.date() if isinstance(when, datetime) else when
        i = bisect.bisect_right(self._days, day) - 1
        if i < 0:
            self.before_first += 1
            i = 0
        return self._values[i]

    def to_base(self, amount: float, when: date | datetime) -> float:
        """Montant en `quote` (ex. USD) -> `base` (ex. EUR)."""
        return amount / self.rate(when)

    def to_quote(self, amount: float, when: date | datetime) -> float:
        return amount * self.rate(when)

    def to_payload(self) -> dict:
        return {"base": self.base, "quote": self.quote, "source": self.source,
                "rates": [[d.isoformat(), v] for d, v in zip(self._days, self._values)]}

    @classmethod
    def from_payload(cls, payload: Mapping) -> "FxRates":
        return cls(payload["base"], payload["quote"],
                   {date.fromisoformat(d): float(v) for d, v in payload.get("rates", [])},
                   payload.get("source", "unknown"))

    @classmethod
    def fixed(cls, base: str, quote: str, rate: float, day: date) -> "FxRates":
        return cls(base, quote, {day: rate}, source="fixed")


def _http_get(url: str) -> str:
    req = urllib.request.Request(url, headers={"Accept": "text/csv"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8")


def fetch_ecb_rates(
    quote: str, start: date, end: date, *, http_get: Callable[[str], str] = _http_get,
    chunk_years: int = 5, retries: int = 3, sleep: Callable[[float], None] = time.sleep,
) -> FxRates:
    """Taux de référence BCE : `quote` par EUR entre deux dates incluses.

    Par tranches de `chunk_years` ans (une requête sur 20 ans dépasse souvent
    le délai de l'API), chaque tranche retentée `retries` fois."""
    if quote == "EUR":
        raise ValueError("quote currency must differ from EUR")
    rates: dict[date, float] = {}
    cursor = start
    while cursor <= end:
        stop = min(end, date(cursor.year + chunk_years, 1, 1) - timedelta(days=1))
        query = urllib.parse.urlencode({"startPeriod": cursor.isoformat(), "endPeriod": stop.isoformat(),
                                        "format": "csvdata"})
        for attempt in range(1, retries + 1):
            try:
                text = http_get(f"{ECB_URL.format(quote=quote)}?{query}")
                break
            except Exception as exc:
                if attempt == retries:
                    raise
                logger.warning("ECB %s..%s attempt %d failed: %s", cursor, stop, attempt, exc)
                sleep(2.0 * attempt)
        for row in csv.DictReader(io.StringIO(text)):
            value = (row.get("OBS_VALUE") or "").strip()
            period = (row.get("TIME_PERIOD") or "").strip()
            if value and period:
                rates[date.fromisoformat(period)] = float(value)
        cursor = stop + timedelta(days=1)
    if not rates:
        raise LookupError(f"ECB returned no EUR/{quote} rate between {start} and {end}")
    return FxRates("EUR", quote, rates, source="ecb")
