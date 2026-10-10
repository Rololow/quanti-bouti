"""Portefeuille vu en EUR : conversion au cours BCE, couverture de change des
obligations, taux courts en euro."""

import json
from datetime import date

import pytest

from trading_engine.backtest.__main__ import main
from trading_engine.backtest.dataset import eur_rates_path, fx_path, import_daily_csv, meta_path, rates_path
from trading_engine.backtest.runner import load_fx
from trading_engine.backtest.sources import _ecb_period, ecb_euro_short_rates
from trading_engine.config import PROJECT_ROOT, load_config
from trading_engine.portfolio.financing import RateSeries
from trading_engine.storage.event_log import read_events
from trading_engine.tax.fx import FxRates

CSV = ("Date,Open,High,Low,Close,Adj Close,Volume,Dividend\n"
       "2025-03-03,100,100,100,100,100,10,0\n"
       "2025-03-04,110,110,110,110,110,10,1\n"
       "2025-03-05,110,110,110,110,110,10,0\n")
FX = {date(2025, 3, 3): 1.00, date(2025, 3, 4): 1.10, date(2025, 3, 5): 1.25}   # USD par EUR


def _closes(path):
    return [b.close for b in read_events(path)]


def test_eur_conversion_and_hedged_share_class(tmp_path):
    f = tmp_path / "x.csv"
    f.write_text(CSV, encoding="utf-8")
    fx = FxRates("EUR", "USD", FX)
    spot = import_daily_csv({"TLT": f}, tmp_path / "spot.jsonl", keep={"TLT": 1.0}, interest={"TLT": 1.0}, to_eur=fx)
    assert spot["currency"] == "EUR"
    c = _closes(tmp_path / "spot.jsonl")
    # indice USD (dividende réinvesti) : 100, 111, 111 ; en EUR : / cours du jour
    assert c == pytest.approx([100 / 1.00, 111 / 1.10, 111 / 1.25])
    carry = 0.0365                                     # 0,01 % par jour
    import_daily_csv({"TLT": f}, tmp_path / "h.jsonl", keep={"TLT": 1.0}, interest={"TLT": 1.0},
                     to_eur=fx, hedged=["TLT"], carry=lambda d: carry)
    h = _closes(tmp_path / "h.jsonl")
    # couvert : rendement USD + différentiel de taux, le change ne joue pas
    assert h[0] == pytest.approx(100.0)
    assert h[1] / h[0] == pytest.approx(1.11 * (1 + carry / 365))
    assert h[2] / h[1] == pytest.approx(1.0 * (1 + carry / 365))
    # intérêts accumulés dans l'unité du prix publié (EUR)
    meta = json.loads(meta_path(tmp_path / "h.jsonl").read_text(encoding="utf-8"))
    assert meta["hedged"] == ["TLT"]
    assert meta["interest_index"]["TLT"][0][1] == pytest.approx(1.0 * 100 / 100 * h[1] / 111)
    with pytest.raises(ValueError, match="hedged"):
        import_daily_csv({"TLT": f}, tmp_path / "z.jsonl", hedged=["TLT"])


def test_ecb_euro_rates_fall_back_and_avoid_look_ahead():
    assert _ecb_period("2006-12") == date(2007, 1, 1)          # moyenne de décembre connue fin décembre
    assert _ecb_period("2016-03-16") == date(2016, 3, 16)
    calls = []

    def http_get(url):
        calls.append(url)
        if "EURIBOR" in url:
            raise OSError("timeout")
        return "TIME_PERIOD,OBS_VALUE\n2016-03-16,-0.40\n2019-09-18,-0.50\n"

    rates, source = ecb_euro_short_rates(date(2016, 1, 1), date(2020, 1, 1), http_get=http_get)
    assert rates == {date(2016, 3, 16): pytest.approx(-0.004), date(2019, 9, 18): pytest.approx(-0.005)}
    assert "DFR" in source and len(calls) == 2


def test_import_csv_cli_in_eur_writes_euro_rates(tmp_path):
    src = tmp_path / "daily.jsonl"
    fx_path(src).write_text(json.dumps(FxRates("EUR", "USD", FX).to_payload()), encoding="utf-8")
    rates_path(src).write_text(json.dumps(RateSeries({date(2025, 1, 1): 0.04}, "usd").to_payload()),
                               encoding="utf-8")
    eur_rates_path(src).write_text(json.dumps(RateSeries({date(2025, 1, 1): 0.02}, "eur").to_payload()),
                                   encoding="utf-8")
    f = tmp_path / "TLT.csv"
    f.write_text(CSV, encoding="utf-8")
    out = tmp_path / "eurh.jsonl"
    code = main(["--overlay", str(PROJECT_ROOT / "config" / "tax_ucits.yaml"), "import-csv", f"TLT={f}",
                 "--net-dividends", "--currency", "EUR", "--hedge", "TLT", "--hedge-cost", "0",
                 "--rates-from", str(src), "--out", str(out)])
    assert code == 0
    assert RateSeries.from_payload(json.loads(rates_path(out).read_text(encoding="utf-8"))).rate(date(2025, 3, 4)) == 0.02
    assert not fx_path(out).exists()                     # rien à convertir
    h = _closes(out)
    assert h[2] / h[1] == pytest.approx(1 + (0.02 - 0.04) / 365)   # couverture : coûte le différentiel


def test_eur_portfolio_needs_no_conversion():
    cfg = load_config("config/config.yaml", (PROJECT_ROOT / "config" / "eur.yaml",))
    fx = load_fx(cfg, PROJECT_ROOT / "does-not-exist.jsonl")
    assert (fx.base, fx.quote) == ("EUR", "EUR") and fx.to_base(123.0, date(2020, 1, 1)) == 123.0


def test_eur_report_caveat_replaces_usd_cash_warning():
    from trading_engine.backtest.runner import caveats

    cfg = load_config("config/config.yaml", (PROJECT_ROOT / "config" / "eur.yaml",))
    out = caveats(cfg, load_fx(cfg, PROJECT_ROOT / "x.jsonl"), {"currency": "EUR", "hedged": ["TLT", "IEF"]})
    assert out[0].startswith("Portefeuille en EUR") and "TLT, IEF" in out[0]
    assert not any(c.startswith("Point de vue EUR") for c in out)
