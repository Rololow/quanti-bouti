"""Change EUR/USD dans la fiscalité et registre fiscal persistant."""

import asyncio
import dataclasses
import json
from datetime import date, datetime, timezone

import pytest

from trading_engine.config import PROJECT_ROOT, FxConfig, load_config
from trading_engine.data.events import FxEvent, PortfolioEvent, TaxLedgerEvent
from trading_engine.engine import Engine
from trading_engine.storage.event_log import read_events
from trading_engine.tax.fx import FxRates, fetch_ecb_rates
from trading_engine.tax.profile import Instrument, load_tax_profile
from trading_engine.tax.tax_model import TaxModel

BE = PROJECT_ROOT / "config" / "taxes" / "BE.toml"
SPY = {"SPY": Instrument("SPY", "etf", "US", "distributing")}
UTC = timezone.utc


def _fx(rates):
    return FxRates("EUR", "USD", {date.fromisoformat(d): v for d, v in rates.items()}, source="test")


# ------------------------------------------------------------------ taux

def test_rate_lookup_uses_last_published_rate():
    fx = _fx({"2026-01-02": 1.10, "2026-01-05": 1.20})
    assert fx.rate(date(2026, 1, 2)) == 1.10
    assert fx.rate(datetime(2026, 1, 4, 12, tzinfo=UTC)) == 1.10      # week-end : taux du vendredi
    assert fx.rate(date(2026, 1, 9)) == 1.20
    assert fx.before_first == 0 and fx.rate(date(2025, 12, 1)) == 1.10 and fx.before_first == 1
    assert fx.to_base(120.0, date(2026, 1, 5)) == pytest.approx(100.0)
    assert fx.to_quote(100.0, date(2026, 1, 5)) == pytest.approx(120.0)
    back = FxRates.from_payload(fx.to_payload())
    assert back.to_payload() == fx.to_payload()
    with pytest.raises(ValueError):
        FxRates("EUR", "USD", {date(2026, 1, 1): 0.0})
    with pytest.raises(LookupError):
        FxRates("EUR", "USD", {}).rate(date(2026, 1, 1))


def test_fetch_ecb_rates_parses_csv():
    seen = []
    csv_text = ("KEY,FREQ,CURRENCY,CURRENCY_DENOM,EXR_TYPE,EXR_SUFFIX,TIME_PERIOD,OBS_VALUE,OBS_STATUS\n"
                "EXR.D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,2026-01-02,1.1034,A\n"
                "EXR.D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,2026-01-05,1.0988,A\n"
                "EXR.D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,2026-01-06,,M\n")

    def get(url):
        seen.append(url)
        return csv_text

    fx = fetch_ecb_rates("USD", date(2026, 1, 1), date(2026, 1, 6), http_get=get)
    assert seen[0].startswith("https://data-api.ecb.europa.eu/service/data/EXR/D.USD.EUR.SP00.A?")
    assert "startPeriod=2026-01-01" in seen[0] and "format=csvdata" in seen[0]
    assert (fx.base, fx.quote, fx.source, len(fx)) == ("EUR", "USD", "ecb", 2)
    assert fx.rate(date(2026, 1, 6)) == 1.0988
    with pytest.raises(LookupError):
        fetch_ecb_rates("USD", date(2026, 1, 1), date(2026, 1, 6), http_get=lambda url: "TIME_PERIOD,OBS_VALUE\n")


def test_fx_config_validation():
    with pytest.raises(ValueError):
        FxConfig(provider="bank")
    with pytest.raises(ValueError):
        FxConfig(fixed_rate=0)


# ------------------------------------------------------------------ fiscalité en EUR

def _model(**kw):
    m = TaxModel(load_tax_profile(BE), SPY, portfolio_currency="USD", **kw)
    return m


def test_transaction_tax_computed_in_euros():
    m = _model()
    m.set_fx(_fx({"2026-03-02": 1.25}))
    charge, _ = m.record_fill("SPY", 10, 500.0, datetime(2026, 3, 2, 15, tzinfo=UTC))
    # 5 000 USD = 4 000 EUR ; TOB 0,35 % = 14 EUR = 17,50 USD
    assert charge.amount == pytest.approx(14.0) and charge.portfolio_amount == pytest.approx(17.5)
    assert m.transaction_taxes_paid == pytest.approx(14.0)
    rec = m.transactions[0]
    assert (rec.day, rec.symbol, rec.side, rec.notional, rec.amount) == (date(2026, 3, 2), "SPY", "buy",
                                                                         pytest.approx(4000.0), pytest.approx(14.0))
    # 1 000 000 USD = 800 000 EUR -> 2 800 EUR plafonnés à 1 600 EUR (= 2 000 USD)
    big, _ = m.record_fill("SPY", 2000, 500.0, datetime(2026, 3, 2, 16, tzinfo=UTC))
    assert big.capped and big.amount == 1600.0 and big.portfolio_amount == pytest.approx(2000.0)


def test_capital_gain_includes_currency_effect():
    m = _model()
    m.set_fx(_fx({"2026-02-02": 1.00, "2026-06-01": 1.25}))
    m.record_fill("SPY", 10, 100.0, datetime(2026, 2, 2, tzinfo=UTC))       # 1 000 EUR
    _, realized = m.record_fill("SPY", -10, 100.0, datetime(2026, 6, 1, tzinfo=UTC))  # 800 EUR
    # même prix en USD, mais le dollar a baissé : moins-value de 200 EUR
    assert realized.gain == pytest.approx(-200.0)


def test_step_up_price_converted_at_reference_date():
    m = _model(step_up_prices={"SPY": 660.0})
    m.set_fx(_fx({"2025-12-31": 1.10, "2026-06-01": 1.20}))
    assert m.gains.step_up_prices["SPY"] == pytest.approx(600.0)
    m.gains.buy("SPY", 1, 500.0, date(2024, 5, 1))                          # avant la taxe
    _, realized = m.record_fill("SPY", -1, 780.0, datetime(2026, 6, 1, tzinfo=UTC))
    assert realized.tax_basis == pytest.approx(600.0) and realized.proceeds == pytest.approx(650.0)


def test_rebalance_cost_returned_in_portfolio_currency():
    m = _model()
    m.set_fx(_fx({"2026-03-02": 1.25}))
    cost = m.rebalance_cost({"SPY": 5000.0}, {"SPY": 500.0}, datetime(2026, 3, 2, tzinfo=UTC))
    assert cost.transaction_tax == pytest.approx(17.5) and cost.per_symbol["SPY"] == pytest.approx(17.5)


def test_missing_rates_are_reported():
    m = _model()
    m.record_fill("SPY", 1, 100.0, datetime(2026, 3, 2, tzinfo=UTC))
    assert m.fx_missing > 0 and any("pas de taux" in w for w in m.warnings())
    with pytest.raises(ValueError):
        m.set_fx(FxRates("EUR", "GBP", {date(2026, 1, 1): 0.9}))


# ------------------------------------------------------------------ registre

def test_ledger_snapshot_restore_roundtrip():
    m = _model()
    m.set_fx(_fx({"2026-02-02": 1.00}))
    m.record_fill("SPY", 10, 100.0, datetime(2026, 2, 2, tzinfo=UTC))
    m.record_fill("SPY", 5, 110.0, datetime(2026, 2, 9, tzinfo=UTC))
    m.record_fill("SPY", -12, 120.0, datetime(2026, 2, 16, tzinfo=UTC))
    snap = json.loads(json.dumps(m.snapshot()))
    other = _model()
    other.restore(snap)
    assert other.snapshot() == m.snapshot()
    assert other.transaction_taxes_paid == pytest.approx(m.transaction_taxes_paid)
    assert other.gains.net_gain(2026) == pytest.approx(m.gains.net_gain(2026))
    lots = other.gains.lots["SPY"]
    assert [(lot.quantity, lot.acquired) for lot in lots] == [(3.0, date(2026, 2, 9))]
    with pytest.raises(ValueError):
        other.restore(dict(snap, version=99))
    with pytest.raises(ValueError):
        other.restore(dict(snap, country="FR"))


def test_align_lots():
    m = _model()
    m.set_fx(_fx({"2026-01-02": 1.25}))
    m.gains.buy("SPY", 10, 80.0, date(2026, 1, 2))
    t = datetime(2026, 3, 2, tzinfo=UTC)
    assert m.align_lots("SPY", 10, 100.0, t, realize=False) == 0.0
    assert m.align_lots("SPY", 12, 125.0, t, realize=False) == 2.0          # lot ajouté à 100 EUR
    assert [(lot.quantity, lot.unit_cost) for lot in m.gains.lots["SPY"]] == [(10, 80.0), (2, 100.0)]
    assert m.align_lots("SPY", 9, 125.0, t, realize=False) == -3.0          # retiré sans plus-value
    assert m.gains.realized == [] and m.lot_quantity("SPY") == 9
    assert m.align_lots("SPY", 4, 150.0, t, realize=True) == -5.0           # fill manqué : vente
    assert m.gains.realized[0].gain == pytest.approx(5 * 120.0 - 5 * 80.0)


# ------------------------------------------------------------------ moteur

def _cfg(**kw):
    cfg = load_config()
    return dataclasses.replace(
        cfg,
        engine=dataclasses.replace(cfg.engine, max_events=12000, report_every=0),
        allocation=dataclasses.replace(cfg.allocation, method="hrp"),
        risk=dataclasses.replace(cfg.risk, min_observations=5),
        decision=dataclasses.replace(cfg.decision, risk_aversion=5_000.0, holding_period="20d"),
        execution=dataclasses.replace(cfg.execution, mode="paper"),
        **kw,
    )


def test_simulation_uses_fixed_rate_and_replays(tmp_path):
    log = tmp_path / "events.jsonl"
    cfg = _cfg()
    assert cfg.tax.ledger_path                      # configuré, mais jamais écrit en simulation
    live = Engine(dataclasses.replace(cfg, storage=dataclasses.replace(cfg.storage, event_log=str(log))))
    assert live.ledger_path is None
    asyncio.run(live.run())
    assert live.fx.source == "fixed" and live.tax.fx is live.fx and live.ledger_source == "config"
    assert live.fills and live.tax.transaction_taxes_paid > 0 and live.tax.fx_missing == 0
    # TOB en EUR, déduite du cash en USD
    rec = live.tax.transactions[0]
    assert rec.amount == pytest.approx(rec.notional * 0.0035) or rec.amount == 1600.0
    kinds = [type(ev).__name__ for ev in read_events(log)]
    assert kinds.index("FxEvent") < kinds.index("TaxLedgerEvent") < kinds.index("TradeEvent")

    replay = Engine(dataclasses.replace(cfg, feed=dataclasses.replace(cfg.feed, provider="replay",
                                                                      replay_path=str(log))))
    asyncio.run(replay.run())
    assert replay.fx.to_payload() == live.fx.to_payload()
    assert replay.tax.snapshot() == live.tax.snapshot()
    assert replay.snapshot() == live.snapshot()


def test_fx_off_keeps_amounts_unconverted():
    cfg = _cfg(fx=FxConfig(provider="off"))
    e = Engine(cfg)
    asyncio.run(e.run())
    assert e.fx is None and not e.tax.needs_fx and e.tax.fx_missing == 0


def test_ledger_persists_across_restarts(tmp_path):
    path = tmp_path / "ledger.json"
    first = Engine(_cfg())
    first.ledger_path = path                        # comme en live Alpaca
    asyncio.run(first.run())
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved == first.tax.snapshot() and saved["transactions"]
    assert not list(tmp_path.glob("*.tmp"))

    second = Engine(_cfg())
    second.ledger_path = path
    asyncio.run(second._bootstrap(datetime(2026, 1, 7, 14, 30, tzinfo=UTC)))
    assert second.ledger_source == "file"
    assert second.tax.snapshot() == saved           # lots datés, TOB et plus-values repris


def test_sync_keeps_ledger_lots_and_flags_differences(tmp_path):
    e = Engine(_cfg())
    t0 = datetime(2026, 1, 7, 14, 30, tzinfo=UTC)
    asyncio.run(e._bootstrap(t0))                   # lots des positions initiales (config)
    before = {s: [(lot.quantity, lot.acquired) for lot in lots] for s, lots in e.tax.gains.lots.items()}
    positions = {s: [e.tax.lot_quantity(s), 100.0, 110.0] for s in e.tax.gains.lots}
    ev = PortfolioEvent(timestamp=t0, received_at=t0, symbol=None, source="alpaca_trading",
                        payload={"kind": "sync", "cash": 1000.0, "equity": 1000.0, "status": "ACTIVE",
                                 "trading_blocked": False, "positions": positions})
    asyncio.run(e._ingest(ev))
    after = {s: [(lot.quantity, lot.acquired) for lot in lots] for s, lots in e.tax.gains.lots.items()}
    assert after == before and not [a for a in e.alerts if a.kind == "TAX_LEDGER"]
    # le compte a 3 SPY de plus et plus aucun TLT
    positions["SPY"][0] += 3
    del positions["TLT"]
    asyncio.run(e._ingest(dataclasses.replace(ev, payload=dict(ev.payload, positions=positions))))
    alert = [a for a in e.alerts if a.kind == "TAX_LEDGER"]
    assert len(alert) == 1 and "SPY +3" in alert[0].message and "TLT -20" in alert[0].message
    assert e.tax.lot_quantity("TLT") == 0 and e.tax.gains.realized == []
    new_lot = e.tax.gains.lots["SPY"][-1]
    assert new_lot.quantity == 3 and new_lot.acquired == t0.date()
    assert new_lot.unit_cost == pytest.approx(100.0 / e.config.fx.fixed_rate)


def test_ledger_events_roundtrip():
    from trading_engine.storage.event_log import dumps, loads
    t = datetime(2026, 1, 2, tzinfo=UTC)
    for ev in (FxEvent(timestamp=t, received_at=t, symbol=None, source="fx_test", payload=_fx({"2026-01-02": 1.1}).to_payload()),
               TaxLedgerEvent(timestamp=t, received_at=t, symbol=None, source="tax_ledger",
                              payload={"source": "config", "ledger": _model().snapshot()})):
        assert dumps(loads(dumps(ev))) == dumps(ev)
