"""Calendrier de marché (NYSE / Nasdaq) : séances, jours fériés, clôtures anticipées.

Deux sources :
- `MarketCalendar.from_rules` : règles NYSE (fériés, observés, 13h00 les
  veilles de fête), sans réseau ; sert de repli et pour la simulation ;
- le calendrier Alpaca (`/v2/calendar`) en live : il fait foi (fermetures
  exceptionnelles, deuils nationaux…).

Le calendrier entre dans le moteur comme un `CalendarEvent` journalisé : le
replay utilise exactement les séances vues en live.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Iterable
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)
EARLY_CLOSE = time(13, 0)


@dataclass(frozen=True)
class Session:
    day: date
    open: datetime            # UTC
    close: datetime           # UTC

    @classmethod
    def local(cls, day: date, open_: time, close: time) -> "Session":
        return cls(day, datetime.combine(day, open_, NY).astimezone(timezone.utc),
                   datetime.combine(day, close, NY).astimezone(timezone.utc))

    @property
    def early_close(self) -> bool:
        return self.close.astimezone(NY).time() < REGULAR_CLOSE


# ------------------------------------------------------------------ règles NYSE

def _easter(year: int) -> date:
    """Dimanche de Pâques (algorithme grégorien anonyme)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    return date(year, month, (h + l - 7 * m + 114) % 31 + 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    last = (date(year, month + 1, 1) if month < 12 else date(year + 1, 1, 1)) - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _observed(d: date) -> date:
    """Samedi -> vendredi, dimanche -> lundi."""
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def nyse_holidays(year: int) -> set[date]:
    days = {
        _nth_weekday(year, 1, 0, 3),              # Martin Luther King Jr. Day
        _nth_weekday(year, 2, 0, 3),              # Washington's Birthday
        _easter(year) - timedelta(days=2),        # Good Friday
        _last_weekday(year, 5, 0),                # Memorial Day
        _observed(date(year, 7, 4)),              # Independence Day
        _nth_weekday(year, 9, 0, 1),              # Labor Day
        _nth_weekday(year, 11, 3, 4),             # Thanksgiving
        _observed(date(year, 12, 25)),            # Christmas
    }
    new_year = date(year, 1, 1)
    if new_year.weekday() != 5:                   # samedi : pas reporté au 31/12
        days.add(_observed(new_year))
    if year >= 2022:
        days.add(_observed(date(year, 6, 19)))    # Juneteenth
    return days


def nyse_early_closes(year: int, holidays: set[date]) -> set[date]:
    candidates = {
        date(year, 7, 3),                                   # veille de l'Independence Day
        _nth_weekday(year, 11, 3, 4) + timedelta(days=1),   # lendemain de Thanksgiving
        date(year, 12, 24),                                 # veille de Noël
    }
    return {d for d in candidates if d.weekday() < 5 and d not in holidays}


# ------------------------------------------------------------------ calendrier

class MarketCalendar:
    def __init__(self, sessions: Iterable[Session], *, start: date, end: date, source: str = "rules") -> None:
        self.sessions = sorted(sessions, key=lambda s: s.day)
        self._by_day = {s.day: s for s in self.sessions}
        self.start, self.end = start, end            # période couverte (jours sans séance inclus)
        self.source = source

    @classmethod
    def from_rules(cls, start: date, end: date) -> "MarketCalendar":
        sessions = []
        holidays: dict[int, set[date]] = {}
        early: dict[int, set[date]] = {}
        d = start
        while d <= end:
            if d.year not in holidays:
                holidays[d.year] = nyse_holidays(d.year)
                early[d.year] = nyse_early_closes(d.year, holidays[d.year])
            if d.weekday() < 5 and d not in holidays[d.year]:
                close = EARLY_CLOSE if d in early[d.year] else REGULAR_CLOSE
                sessions.append(Session.local(d, REGULAR_OPEN, close))
            d += timedelta(days=1)
        return cls(sessions, start=start, end=end, source="rules")

    def covers(self, day: date) -> bool:
        return self.start <= day <= self.end

    def session(self, day: date) -> Session | None:
        """Séance du jour (règles NYSE si le jour n'est pas couvert)."""
        if self.covers(day):
            return self._by_day.get(day)
        return MarketCalendar.from_rules(day, day)._by_day.get(day)

    def session_at(self, ts: datetime) -> Session | None:
        s = self.session(ts.astimezone(NY).date())
        return s if s is not None and s.open <= ts < s.close else None

    def is_open(self, ts: datetime) -> bool:
        return self.session_at(ts) is not None

    def next_open(self, ts: datetime, max_days: int = 15) -> datetime | None:
        day = ts.astimezone(NY).date()
        for i in range(max_days):
            s = self.session(day + timedelta(days=i))
            if s is not None and s.open > ts:
                return s.open
        return None

    def last_open(self, ts: datetime, max_days: int = 15) -> datetime | None:
        """Ouverture de la séance en cours ou de la dernière séance commencée."""
        day = ts.astimezone(NY).date()
        for i in range(max_days):
            s = self.session(day - timedelta(days=i))
            if s is not None and s.open <= ts:
                return s.open
        return None

    def trading_block(self, ts: datetime, avoid_open: float = 0.0, avoid_close: float = 0.0) -> str | None:
        """Raison de ne pas trader à `ts` (None = séance ouverte et tradable).
        `avoid_open` / `avoid_close` : minutes évitées après l'ouverture et avant
        la clôture (spreads larges, enchères)."""
        s = self.session_at(ts)
        if s is None:
            nxt = self.next_open(ts)
            when = "inconnue" if nxt is None else nxt.astimezone(NY).strftime("%Y-%m-%d %H:%M ET")
            return f"marché fermé (prochaine ouverture {when})"
        if ts < s.open + timedelta(minutes=avoid_open):
            return f"ouverture : {avoid_open:g} premières minutes évitées"
        if ts >= s.close - timedelta(minutes=avoid_close):
            return f"clôture dans moins de {avoid_close:g} min"
        return None

    # ------------------------------------------------------------ sérialisation

    def to_payload(self) -> dict:
        return {
            "source": self.source, "start": self.start.isoformat(), "end": self.end.isoformat(),
            "sessions": [[s.day.isoformat(), s.open.isoformat(), s.close.isoformat()] for s in self.sessions],
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "MarketCalendar":
        sessions = [Session(date.fromisoformat(d), datetime.fromisoformat(o), datetime.fromisoformat(c))
                    for d, o, c in payload.get("sessions", [])]
        return cls(sessions, start=date.fromisoformat(payload["start"]), end=date.fromisoformat(payload["end"]),
                   source=payload.get("source", "unknown"))
