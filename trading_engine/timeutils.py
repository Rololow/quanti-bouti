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
