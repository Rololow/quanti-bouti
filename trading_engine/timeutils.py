"""Helpers de temps : tous les timestamps du moteur sont en UTC et tz-aware."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

_TIMEFRAME_RE = re.compile(r"^(\d+)(s|m|h|d)$")
_UNITS = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days"}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def ensure_utc(ts: datetime) -> datetime:
    """Refuse les datetimes naïfs : un timestamp sans fuseau est ambigu."""
    if ts.tzinfo is None:
        raise ValueError(f"naive datetime not allowed: {ts!r}")
    return ts.astimezone(timezone.utc)


def parse_timeframe(timeframe: str) -> timedelta:
    """'5m' -> 5 minutes, '1h' -> 1 heure, '1d' -> 1 jour."""
    match = _TIMEFRAME_RE.match(timeframe)
    if match is None:
        raise ValueError(f"invalid timeframe: {timeframe!r}")
    value, unit = match.groups()
    delta = timedelta(**{_UNITS[unit]: int(value)})
    if delta <= timedelta(0):
        raise ValueError(f"timeframe must be positive: {timeframe!r}")
    return delta


def floor_time(ts: datetime, step: timedelta) -> datetime:
    """Arrondit `ts` au début de l'intervalle `step` (aligné sur l'epoch UTC)."""
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    return epoch + ((ensure_utc(ts) - epoch) // step) * step


def parse_rfc3339(value: str) -> datetime:
    """Parse un timestamp RFC 3339 (ex. Alpaca, précision nanoseconde).

    Les chiffres au-delà de la microseconde sont tronqués.
    """
    return ensure_utc(datetime.fromisoformat(value))


def format_timeframe(delta: timedelta) -> str:
    """Inverse de `parse_timeframe` : 30 minutes -> '30m', 2 jours -> '2d'."""
    seconds = int(delta.total_seconds())
    if seconds <= 0 or seconds != delta.total_seconds():
        raise ValueError(f"unsupported timeframe: {delta!r}")
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if seconds % size == 0:
            return f"{seconds // size}{unit}"
    return f"{seconds}s"


TRADING_DAYS_PER_YEAR = 252
TRADING_HOURS_PER_DAY = 6.5


def periods_per_year(timeframe: str) -> float:
    """Nombre de barres par an (séance actions de 6h30 pour l'intraday)."""
    step = parse_timeframe(timeframe)
    if step >= timedelta(days=1):
        return TRADING_DAYS_PER_YEAR / (step / timedelta(days=1))
    return TRADING_DAYS_PER_YEAR * timedelta(hours=TRADING_HOURS_PER_DAY) / step
