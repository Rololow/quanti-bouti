import asyncio
import dataclasses
import json
from datetime import datetime, timedelta, timezone

import pytest

from trading_engine.alerts.alerts import AlertEngine
from trading_engine.config import load_config
from trading_engine.data.alpaca_feed import AlpacaNewsFeed
from trading_engine.data.edgar import EdgarFundamentalFeed, available_from_filed, parse_companyfacts
from trading_engine.data.event_file_feed import EventFileFeed
from trading_engine.data.events import FundamentalEvent, NewsEvent, TradeEvent
from trading_engine.data.market_feed import MarketFeed
from trading_engine.data.merge import ConcurrentFeed, TimeMergedFeed
from trading_engine.data.simulated_qualitative import SimulatedQualitativeFeed
from trading_engine.engine import Engine
from trading_engine.features.fundamentals import FundamentalFeatures, FundamentalStore, previous_year_period
from trading_engine.news.news_engine import NewsEngine
from trading_engine.storage.event_log import EventLogWriter

T0 = datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc)


def fund(name, value, period, available, received=None, estimate=None, sym="AAPL"):
    received = received or available
    return FundamentalEvent(timestamp=available, received_at=received, symbol=sym, source="t",
                            name=name, value=value, period=period, available_at=available,
                            estimate=estimate)


def news(headline, minutes, sym="AAPL", news_id="", provider="Reuters"):
    ts = T0 + timedelta(minutes=minutes)
    return NewsEvent(timestamp=ts, received_at=ts, symbol=sym, source="t", headline=headline,
                     news_id=news_id, provider=provider)


class ListFeed(MarketFeed):
    def __init__(self, events):
        self.events = events

    async def __aiter__(self):
        for ev in self.events:
            yield ev


# ------------------------------------------------------------------ point-in-time

def test_fact_is_invisible_before_publication():
    store = FundamentalStore()
    # base historique : la valeur est « reçue » avant d'être publiée
    store.add(fund("eps", 2.0, "2026Q1", T0 + timedelta(days=10), received=T0))
    assert store.latest("AAPL", "eps", T0 + timedelta(days=9)) is None
    assert store.latest("AAPL", "eps", T0 + timedelta(days=10)).value == 2.0


def test_revisions_are_point_in_time():
    store = FundamentalStore()
    store.add(fund("revenue", 100.0, "2026Q1", T0))
    store.add(fund("revenue", 90.0, "2026Q1", T0 + timedelta(days=30)))     # révision
    assert store.value_for_period("AAPL", "revenue", "2026Q1", T0 + timedelta(days=1)) == 100.0
    assert store.value_for_period("AAPL", "revenue", "2026Q1", T0 + timedelta(days=31)) == 90.0


def test_derived_fundamental_features():
    ff = FundamentalFeatures(half_life_days=30)
    ff.store.add(fund("eps", 2.0, "2025Q1", T0 - timedelta(days=365)))
    ff.store.add(fund("eps", 1.8, "2025Q4", T0 - timedelta(days=90)))
    ff.store.add(fund("eps", 2.2, "2026Q1", T0, estimate=2.0))
    ff.store.add(fund("guidance_eps", 8.0, "2026FY", T0 - timedelta(days=90)))
    ff.store.add(fund("guidance_eps", 8.4, "2026FY", T0))
    snap = ff.snapshot("AAPL", T0)
    assert snap["fund_eps_surprise"] == pytest.approx(0.1)
    assert snap["fund_eps_growth"] == pytest.approx(0.1)            # vs 2025Q1, pas 2025Q4
    assert snap["fund_guidance_change"] == pytest.approx(0.05)
    assert snap["qual_score"] > 0
    later = ff.snapshot("AAPL", T0 + timedelta(days=60))
    assert later["qual_score"] == pytest.approx(snap["qual_score"] / 4, rel=1e-6)   # 2 demi-vies
    assert ff.snapshot("AAPL", T0 - timedelta(days=400)) == {}
    assert previous_year_period("2026Q3") == "2025Q3" and previous_year_period("x") is None


def test_engine_never_uses_a_fact_before_publication():
    """Un fait reçu au début mais publié plus tard n'apparaît dans les features
    qu'une fois l'horloge de marché passée à sa date de publication."""
    publish = T0 + timedelta(minutes=10)
    events = [fund("eps", 2.2, "2026Q1", publish, received=T0, estimate=2.0, sym="SPY")]
    seen = []
    for i in range(1, 30):
        ts = T0 + timedelta(minutes=i)
        events.append(TradeEvent(timestamp=ts, received_at=ts, symbol="SPY", source="t",
                                 price=100.0 + 0.01 * (i % 2), size=10))
    cfg = load_config()
    engine = Engine(dataclasses.replace(cfg, engine=dataclasses.replace(cfg.engine, report_every=0)),
                    feed=ListFeed(events))

    def spy(ev):
        seen.append((ev.timestamp, "fund_eps_surprise" in engine.features.snapshot("SPY")))
    engine.bus.subscribe("trade", spy)
    asyncio.run(engine.run())
    assert all(not visible for ts, visible in seen if ts < publish)
    assert all(visible for ts, visible in seen if ts >= publish)


# ------------------------------------------------------------------ news

def test_rephrased_articles_are_one_event():
    ne = NewsEngine()
    c1, new1 = ne.on_news(news("AAPL beats expectations and raises outlook", 0))
    c2, new2 = ne.on_news(news("AAPL tops estimates, lifts forecast", 5, provider="Bloomberg"))
    c3, new3 = ne.on_news(news("AAPL faces regulatory probe", 6))
    assert new1 and not new2 and new3
    assert c1 is c2 and c1.articles == 2 and c1.providers == {"Reuters", "Bloomberg"}
    assert c3 is not c1
    snap = ne.snapshot("AAPL", T0 + timedelta(minutes=10))
    decay = [0.5 ** (10 / 360), 0.5 ** (4 / 360)]                  # demi-vie 6 h
    assert snap["news_activity"] == pytest.approx(sum(decay))       # 2 événements, pas 3 articles
    assert snap["news_echo"] == pytest.approx((2 * decay[0] + decay[1]) / sum(decay))


def test_same_news_id_and_time_window():
    ne = NewsEngine(window=timedelta(hours=1))
    c1, _ = ne.on_news(news("AAPL something", 0, news_id="42"))
    c2, new = ne.on_news(news("totally different words", 1, news_id="42"))
    assert c1 is c2 and not new
    _, new_later = ne.on_news(news("AAPL something", 180))           # hors fenêtre
    assert new_later


def test_news_activity_decays_and_is_point_in_time():
    ne = NewsEngine(activity_half_life=timedelta(hours=6))
    ne.on_news(news("AAPL announces contract", 0))
    assert ne.snapshot("AAPL", T0 - timedelta(minutes=1)) == {}
    assert ne.snapshot("AAPL", T0 + timedelta(hours=6))["news_activity"] == pytest.approx(0.5)


# ------------------------------------------------------------------ sources

def test_time_merged_feed_orders_by_reception_and_stops_with_market():
    market = ListFeed([TradeEvent(timestamp=T0 + timedelta(minutes=m), received_at=T0 + timedelta(minutes=m),
                                  symbol="A", source="t", price=1.0) for m in (0, 10, 20)])
    extra = ListFeed([news("A one", 5, sym="A"), news("A two", 15, sym="A"), news("A late", 60, sym="A")])
    out = asyncio.run(_collect(TimeMergedFeed([market, extra])))
    assert [e.received_at for e in out] == sorted(e.received_at for e in out)
    assert [type(e).__name__ for e in out] == ["TradeEvent", "NewsEvent", "TradeEvent", "NewsEvent", "TradeEvent"]


def test_concurrent_feed_merges_everything():
    a = ListFeed([news("A one", 1, sym="A")])
    b = ListFeed([news("B one", 2, sym="B"), news("B two", 3, sym="B")])
    out = asyncio.run(_collect(ConcurrentFeed([a, b])))
    assert sorted(e.headline for e in out) == ["A one", "B one", "B two"]


async def _collect(feed):
    return [ev async for ev in feed]


def test_event_file_feed(tmp_path):
    path = tmp_path / "fund.jsonl"
    with EventLogWriter(path) as w:
        w.write(fund("eps", 2.0, "2026Q1", T0 + timedelta(days=1)))
        w.write(fund("eps", 1.0, "2025Q4", T0))
    out = asyncio.run(_collect(EventFileFeed(path)))
    assert [e.period for e in out] == ["2025Q4", "2026Q1"]


def test_simulated_qualitative_feed_is_deterministic():
    def run():
        feed = SimulatedQualitativeFeed(["A", "B"], start=T0, until=T0 + timedelta(days=3), seed=1)
        return asyncio.run(_collect(feed))
    a, b = run(), run()
    assert a == b and a
    assert any(isinstance(e, FundamentalEvent) and e.estimate for e in a)
    assert any(isinstance(e, NewsEvent) and "duplicate_of" in e.payload for e in a)
    assert [e.received_at for e in a] == sorted(e.received_at for e in a)


EDGAR_FIXTURE = {
    "cik": 320193,
    "facts": {"us-gaap": {
        "EarningsPerShareDiluted": {"units": {"USD/shares": [
            {"end": "2025-12-27", "val": 2.4, "fy": 2026, "fp": "Q1", "form": "10-Q", "filed": "2026-01-30", "accn": "x1"},
            {"end": "2025-12-27", "val": 2.4, "fy": 2026, "fp": "Q1", "form": "10-Q", "filed": "2026-01-30", "accn": "x1"},
            {"end": "2025-12-27", "val": 2.4, "fy": 2026, "fp": "Q1", "form": "8-K", "filed": "2026-01-29"},
        ]}},
        "Revenues": {"units": {"USD": [
            {"end": "2025-12-27", "val": 1.2e11, "fy": 2026, "fp": "Q1", "form": "10-Q", "filed": "2026-01-30"},
        ]}},
    }},
}


def test_edgar_parser_is_conservative_about_availability():
    events = parse_companyfacts(EDGAR_FIXTURE, "AAPL")
    assert {(e.name, e.period) for e in events} == {("eps", "2026Q1"), ("revenue", "2026Q1")}
    assert all(e.available_at == datetime(2026, 1, 31, tzinfo=timezone.utc) for e in events)
    assert available_from_filed("2026-01-30") == datetime(2026, 1, 31, tzinfo=timezone.utc)


def test_edgar_feed_requires_contact_and_deduplicates():
    with pytest.raises(ValueError):
        EdgarFundamentalFeed({"AAPL": 320193}, user_agent="")
    calls = []

    def fetch(url, ua):
        calls.append((url, ua))
        return EDGAR_FIXTURE

    async def no_sleep(_):
        return None

    feed = EdgarFundamentalFeed({"AAPL": 320193}, user_agent="Test test@example.com", fetch=fetch,
                                clock=lambda: datetime(2026, 3, 1, tzinfo=timezone.utc),
                                sleep=no_sleep, max_polls=2)
    out = asyncio.run(_collect(feed))
    assert len(calls) == 2 and "CIK0000320193" in calls[0][0]
    assert len(out) == 2                                  # rien de nouveau au 2e passage
    assert all(e.received_at > e.available_at for e in out)


def test_alpaca_news_message_fans_out_per_symbol():
    feed = AlpacaNewsFeed(["*"], key_id="k", secret_key="s", clock=lambda: T0)
    assert feed.channels == {"news": True} and "v1beta1/news" in feed.url
    events = feed.parse_message({
        "T": "n", "id": 24918784, "headline": "Apple and Microsoft announce partnership",
        "summary": "…", "author": "Benzinga Newsdesk", "created_at": "2026-01-05T14:25:00Z",
        "updated_at": "2026-01-05T14:25:00Z", "url": "https://example.com/n",
        "symbols": ["AAPL", "MSFT"], "source": "benzinga",
    })
    assert [e.symbol for e in events] == ["AAPL", "MSFT"]
    assert {e.news_id for e in events} == {"24918784"}
    assert events[0].timestamp < events[0].received_at


# ------------------------------------------------------------------ alertes

def test_earnings_and_guidance_alerts_once():
    ae = AlertEngine()
    first = ae.evaluate(T0, earnings={"AAPL": (T0, 0.12)}, guidance={"AAPL": (T0, 0.05)})
    assert {a.kind for a in first} == {"NEW_EARNINGS", "GUIDANCE_CHANGE"}
    assert next(a for a in first if a.kind == "NEW_EARNINGS").severity == "warning"
    assert ae.evaluate(T0, earnings={"AAPL": (T0, 0.12)}, guidance={"AAPL": (T0, 0.05)}) == []
    small = ae.evaluate(T0, guidance={"AAPL": (T0 + timedelta(days=1), 0.001)})
    assert small == []                                      # sous le seuil


# ------------------------------------------------------------------ simulation complète

def test_simulation_includes_qualitative_data_by_default_only_when_simulated():
    cfg = load_config()
    engine = Engine(dataclasses.replace(cfg, engine=dataclasses.replace(cfg.engine, max_events=20000,
                                                                        report_every=0)))
    asyncio.run(engine.run())
    assert engine.fundamentals.store.count > 0 and engine.news.articles > 0
    clusters = sum(len(engine.news.clusters(s)) for s in engine.features.symbols())
    assert clusters < engine.news.articles                  # des reprises ont été regroupées
    assert any(a.kind == "NEW_EARNINGS" for a in engine.alerts)
    replay_cfg = dataclasses.replace(cfg, feed=dataclasses.replace(cfg.feed, provider="replay",
                                                                   replay_path="unused"))
    assert Engine.__dict__["_qualitative_provider"](type("E", (), {"config": replay_cfg})(), "auto") == "none"
