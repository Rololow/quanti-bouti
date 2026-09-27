from datetime import timedelta

import pytest

from trading_engine.data.events import BarEvent, QuoteEvent, TradeEvent
from trading_engine.data.integrity import REJECT, DataIntegrity, IntegrityConfig


def _trade(t0, i, price, sym="SPY", size=10, **kw):
    ts = t0 + timedelta(seconds=i)
    return TradeEvent(timestamp=ts, received_at=kw.pop("received_at", ts), symbol=sym,
                      source="t", price=price, size=size, **kw)


def _warm(di, t0, n=30, price=100.0):
    for i in range(n):
        di.check(_trade(t0, i, price * (1 + 0.0005 * (-1) ** i)))
    return n


def _kinds(result):
    return [i.kind for i in result.issues]


def test_clean_trade_passes(t0):
    di = DataIntegrity()
    res = di.check(_trade(t0, 0, 100.0))
    assert res.accepted and not res.issues
    assert di.score("SPY") == 1.0


def test_glitch_is_quarantined_then_rejected(t0):
    di = DataIntegrity()
    n = _warm(di, t0)
    spike = di.check(_trade(t0, n, 1000.0))            # fat finger ×10
    assert spike.accepted == [] and _kinds(spike) == ["PRICE_JUMP_QUARANTINED"]
    back = di.check(_trade(t0, n + 1, 100.02))
    assert [e.price for e in back.accepted] == [100.02]  # le glitch n'est jamais transmis
    assert "GLITCH" in _kinds(back)
    assert di.score("SPY") < 1.0


def test_confirmed_jump_is_released(t0):
    di = DataIntegrity()
    n = _warm(di, t0)
    di.check(_trade(t0, n, 90.0))                      # -10 % : quarantaine
    res = di.check(_trade(t0, n + 1, 89.8))            # confirmé
    assert [e.price for e in res.accepted] == [90.0, 89.8]
    assert "CONFIRMED_JUMP" in _kinds(res)
    assert not di.corporate_actions


def test_split_detected_as_corporate_action(t0):
    di = DataIntegrity()
    n = _warm(di, t0)
    di.check(_trade(t0, n, 50.0))                      # 2:1
    res = di.check(_trade(t0, n + 1, 50.01))
    assert "POSSIBLE_CORPORATE_ACTION" in _kinds(res)
    assert "SPY" in di.corporate_actions


def test_no_jump_detection_during_warmup(t0):
    di = DataIntegrity()
    di.check(_trade(t0, 0, 100.0))
    res = di.check(_trade(t0, 1, 150.0))
    assert res.accepted and not res.issues


def test_future_and_late_timestamps(t0):
    di = DataIntegrity(IntegrityConfig(max_clock_skew=2, max_lateness=60))
    future = _trade(t0, 10, 100.0, received_at=t0)      # 10 s d'avance
    assert "FUTURE_TIMESTAMP" in _kinds(di.check(future))
    di.check(_trade(t0, 200, 100.0))
    late = di.check(_trade(t0, 100, 100.0))              # 100 s de retard
    assert late.accepted == [] and "LATE_EVENT" in _kinds(late)


def test_invalid_quotes_and_bars(t0):
    di = DataIntegrity()
    crossed = di.check(QuoteEvent(timestamp=t0, symbol="SPY", source="t", bid=101, ask=100))
    assert crossed.accepted == [] and _kinds(crossed) == ["CROSSED_QUOTE"]
    wide = di.check(QuoteEvent(timestamp=t0, symbol="SPY", source="t", bid=90, ask=110))
    assert wide.accepted and _kinds(wide) == ["WIDE_SPREAD"]
    bad_bar = BarEvent(timestamp=t0, end=t0 + timedelta(minutes=1), symbol="SPY", source="t",
                       timeframe="1m", open=100, high=99, low=101, close=100)
    res = di.check(bad_bar)
    assert res.accepted == [] and res.issues[0].severity == REJECT


def test_stale_symbols(t0):
    di = DataIntegrity(IntegrityConfig(stale_after=60))
    di.check(_trade(t0, 0, 100.0, sym="TLT"))
    di.check(_trade(t0, 120, 100.0, sym="SPY"))
    assert di.stale_symbols() == ["TLT"]
