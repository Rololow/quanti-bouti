import asyncio
import dataclasses
from datetime import date, datetime, timedelta, timezone

import pytest

from trading_engine.config import CalendarConfig, load_config
from trading_engine.data.calendar import MarketCalendar, nyse_early_closes, nyse_holidays
from trading_engine.data.events import CalendarEvent
from trading_engine.data.integrity import DataIntegrity
from trading_engine.data.events import TradeEvent
from trading_engine.api.state import build_state
from trading_engine.engine import Engine
from trading_engine.execution.alpaca_trading import AlpacaTradingClient
from trading_engine.storage.event_log import dumps, loads, read_events

UTC = timezone.utc


def test_nyse_holidays_2026():
    assert sorted(nyse_holidays(2026)) == [
        date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3), date(2026, 5, 25),
        date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7), date(2026, 11, 26), date(2026, 12, 25),
    ]
    assert nyse_early_closes(2026, nyse_holidays(2026)) == {date(2026, 11, 27), date(2026, 12, 24)}


def test_nyse_observed_rules():
    # Noël 2027 un samedi : férié le vendredi 24, donc pas de clôture anticipée ce jour-là
    h2027 = nyse_holidays(2027)
    assert date(2027, 12, 24) in h2027 and date(2027, 12, 24) not in nyse_early_closes(2027, h2027)
    # 1er janvier 2028 un samedi : pas reporté au vendredi 31 décembre 2027
    assert date(2027, 12, 31) not in h2027 and date(2027, 12, 31) not in nyse_holidays(2028)
    assert nyse_early_closes(2025, nyse_holidays(2025)) == {date(2025, 7, 3), date(2025, 11, 28),
                                                           date(2025, 12, 24)}


def test_sessions_follow_new_york_time_and_dst():
    cal = MarketCalendar.from_rules(date(2026, 1, 1), date(2026, 12, 31))
    jan = cal.session(date(2026, 1, 5))
    jul = cal.session(date(2026, 7, 6))
    assert (jan.open, jan.close) == (datetime(2026, 1, 5, 14, 30, tzinfo=UTC), datetime(2026, 1, 5, 21, tzinfo=UTC))
    assert (jul.open, jul.close) == (datetime(2026, 7, 6, 13, 30, tzinfo=UTC), datetime(2026, 7, 6, 20, tzinfo=UTC))
    early = cal.session(date(2026, 11, 27))
    assert early.early_close and early.close == datetime(2026, 11, 27, 18, tzinfo=UTC)
    assert cal.session(date(2026, 1, 3)) is None and cal.session(date(2026, 4, 3)) is None
    assert len(cal.sessions) == 251


def test_trading_block_and_next_open():
    cal = MarketCalendar.from_rules(date(2026, 1, 1), date(2026, 1, 31))
    fri_close = datetime(2026, 1, 16, 21, 0, tzinfo=UTC)
    assert cal.is_open(fri_close - timedelta(seconds=1)) and not cal.is_open(fri_close)
    # vendredi soir -> mardi (lundi 19 = MLK Day)
    assert cal.next_open(fri_close) == datetime(2026, 1, 20, 14, 30, tzinfo=UTC)
    assert "2026-01-20 09:30 ET" in cal.trading_block(fri_close)
    assert "ouverture" in cal.trading_block(datetime(2026, 1, 20, 14, 32, tzinfo=UTC), 5, 15)
    assert cal.trading_block(datetime(2026, 1, 20, 14, 35, tzinfo=UTC), 5, 15) is None
    assert "clôture" in cal.trading_block(datetime(2026, 1, 20, 20, 50, tzinfo=UTC), 5, 15)
    assert cal.last_open(datetime(2026, 1, 17, 3, tzinfo=UTC)) == datetime(2026, 1, 16, 14, 30, tzinfo=UTC)
    # hors de la période couverte : règles NYSE
    assert cal.session(date(2026, 3, 2)) is not None


def test_calendar_payload_and_event_roundtrip():
    cal = MarketCalendar.from_rules(date(2026, 1, 1), date(2026, 1, 10))
    back = MarketCalendar.from_payload(cal.to_payload())
    assert back.sessions == cal.sessions and (back.start, back.end, back.source) == (cal.start, cal.end, "rules")
    t = datetime(2026, 1, 2, tzinfo=UTC)
    ev = CalendarEvent(timestamp=t, received_at=t, symbol=None, source="calendar_rules", payload=cal.to_payload())
    assert MarketCalendar.from_payload(loads(dumps(ev)).payload).sessions == cal.sessions


def test_alpaca_calendar_is_authoritative():
    """Fermeture exceptionnelle (deuil national du 9 janvier 2025) absente des règles."""
    def http(method, url, headers, body=None):
        assert method == "GET" and url.endswith("/v2/calendar?start=2025-01-08&end=2025-01-10")
        return [{"date": "2025-01-08", "open": "09:30", "close": "16:00"},
                {"date": "2025-01-10", "open": "09:30", "close": "16:00"}]

    cal = AlpacaTradingClient(key_id="k", secret_key="s", http=http).calendar(date(2025, 1, 8), date(2025, 1, 10))
    assert cal.source == "alpaca" and cal.session(date(2025, 1, 9)) is None
    assert MarketCalendar.from_rules(date(2025, 1, 9), date(2025, 1, 9)).session(date(2025, 1, 9)) is not None


def test_staleness_counted_from_session_open():
    integ = DataIntegrity()
    friday = datetime(2026, 1, 16, 20, 59, tzinfo=UTC)
    integ.check(TradeEvent(timestamp=friday, received_at=friday, symbol="SPY", source="t", price=100, size=1))
    tuesday_open = datetime(2026, 1, 20, 14, 30, tzinfo=UTC)
    now = tuesday_open + timedelta(minutes=2)
    assert integ.stale_symbols(now) == ["SPY"]
    assert integ.stale_symbols(now, since=tuesday_open) == []
    assert integ.stale_symbols(tuesday_open + timedelta(minutes=10), since=tuesday_open) == ["SPY"]


def test_calendar_config_validation():
    with pytest.raises(ValueError):
        CalendarConfig(mode="sometimes")
    with pytest.raises(ValueError):
        CalendarConfig(avoid_close_minutes=-1)


# ------------------------------------------------------------------ moteur

def _cfg(calendar_mode, max_events=12000, **storage):
    cfg = load_config()
    return dataclasses.replace(
        cfg,
        engine=dataclasses.replace(cfg.engine, max_events=max_events, report_every=0),
        allocation=dataclasses.replace(cfg.allocation, method="hrp"),
        risk=dataclasses.replace(cfg.risk, min_observations=5),
        decision=dataclasses.replace(cfg.decision, risk_aversion=50.0, holding_period="20d"),
        execution=dataclasses.replace(cfg.execution, mode="paper"),
        calendar=dataclasses.replace(cfg.calendar, mode=calendar_mode),
        storage=dataclasses.replace(cfg.storage, **storage),
    )


def test_simulation_ignores_calendar_by_default():
    e = Engine(_cfg("auto"))
    asyncio.run(e.run())
    assert e.calendar is None and e.market_block is None


def test_engine_trades_only_during_session_and_replays(tmp_path):
    log = tmp_path / "events.jsonl"
    # ~ 30 h simulées : fin de séance, nuit, puis séance du lendemain
    live = Engine(_cfg("on", max_events=22000, event_log=str(log)))
    asyncio.run(live.run())
    cal = live.calendar
    assert cal is not None and cal.source == "rules"
    stamps = [d.timestamp for d in live.decisions]
    assert any(not cal.is_open(t) for t in stamps), "la simulation doit dépasser la clôture"
    avoid = timedelta(minutes=live.config.calendar.avoid_close_minutes)
    for d in live.decisions:
        session = cal.session_at(d.timestamp)
        tradable = session is not None and d.timestamp < session.close - avoid \
            and d.timestamp >= session.open + timedelta(minutes=live.config.calendar.avoid_open_minutes)
        if not tradable:
            assert d.action == "UDONOTHING" and d.reasons
    closed = [d for d in live.decisions if cal.session_at(d.timestamp) is None]
    assert closed and all("marché fermé" in d.reasons[0] for d in closed)
    assert any(d.action == "UREBALANCE" for d in live.decisions)
    for plan in live.plans:
        for order in plan.orders:
            assert order.expires_at <= cal.session_at(order.timestamp).close
    assert all(cal.is_open(f.timestamp) for f in live.fills)
    assert sum(isinstance(ev, CalendarEvent) for ev in read_events(log)) == 1
    market = build_state(live)["meta"]["market"]
    assert market["calendar"] == "rules" and market["open"] == cal.is_open(live.features.now)

    # replay : le calendrier vient du journal (mode auto, aucun chargement)
    cfg = _cfg("auto", max_events=22000)
    replay = Engine(dataclasses.replace(cfg, feed=dataclasses.replace(cfg.feed, provider="replay",
                                                                      replay_path=str(log))))
    asyncio.run(replay.run())
    assert replay.calendar.sessions == cal.sessions
    assert list(replay.decisions) == list(live.decisions)
    assert replay.fills == live.fills and replay.snapshot() == live.snapshot()
