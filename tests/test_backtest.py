"""Backtest : dataset, rejeu en trades synthétiques, références, métriques,
walk-forward et rapport (données synthétiques, sans réseau)."""

import asyncio
import json
from datetime import date, datetime, timedelta, timezone
from urllib.parse import parse_qs, urlparse

import pytest

from trading_engine.allocation.constraints import fit_gross_after_freeze
from trading_engine.backtest import metrics
from trading_engine.backtest.benchmarks import StrategyResult, buy_and_hold, daily_closes, monthly_inverse_vol
from trading_engine.backtest.dataset import (
    BarDatasetFeed,
    download_bars,
    fx_path,
    read_closes,
    synthetic_trades,
    write_synthetic_dataset,
)
from trading_engine.backtest.runner import (
    format_report,
    label_of,
    parse_grid,
    run_backtest,
    to_json,
    walk_forward,
    with_params,
)
from trading_engine.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, deep_merge, load_config
from trading_engine.data.alpaca_history import AlpacaHistoricalClient
from trading_engine.data.events import BarEvent
from trading_engine.storage.event_log import read_events

UTC = timezone.utc
OVERLAY = PROJECT_ROOT / "config" / "backtest.yaml"


# ------------------------------------------------------------------ config

def test_deep_merge_and_overlay():
    base = {"a": {"x": 1, "y": 2}, "l": [1, 2], "p": {"SPY": 1}}
    assert deep_merge(base, {"a": {"y": 3}, "l": [9], "p": {}}) == {"a": {"x": 1, "y": 3}, "l": [9], "p": {}}
    cfg = load_config(DEFAULT_CONFIG_PATH, (OVERLAY,))
    assert cfg.feed.provider == "dataset" and cfg.bar_timeframes == ("30m", "1h", "1d")
    assert dict(cfg.portfolio.positions) == {} and cfg.portfolio.cash == 100000.0
    assert cfg.execution.volume_timeframe == "30m" and cfg.decision.holding_period == "60d"
    assert load_config().feed.provider == "simulated"          # la base n'est pas modifiée


def test_with_params_and_grid():
    cfg = load_config()
    out = with_params(cfg, {"decision.holding_period": "120d", "decision.risk_aversion": "20",
                            "decision.include_alpha": "false"})
    assert out.decision.holding_period == "120d" and out.decision.risk_aversion == 20.0
    assert out.decision.include_alpha is False
    with pytest.raises(KeyError):
        with_params(cfg, {"decision.nope": 1})
    grid = parse_grid(["decision.holding_period=20d,60d", "decision.risk_aversion=5,10"])
    assert len(grid) == 4 and grid[0] == {"decision.holding_period": "20d", "decision.risk_aversion": "5"}
    assert parse_grid([]) == [{}] and label_of({"decision.holding_period": "20d"}) == "holding_period=20d"
    with pytest.raises(ValueError):
        parse_grid(["oops"])


# ------------------------------------------------------------------ dataset

def _bar(t, sym="SPY", o=100.0, h=102.0, low=99.0, c=101.0, v=400.0):
    return BarEvent(timestamp=t, end=t + timedelta(minutes=30), received_at=t + timedelta(minutes=30),
                    symbol=sym, source="t", timeframe="30m", open=o, high=h, low=low, close=c, volume=v)


def test_synthetic_trades_follow_ohlc_path():
    t = datetime(2026, 3, 2, 14, 30, tzinfo=UTC)
    up = synthetic_trades(_bar(t))
    assert [x.price for x in up] == [100.0, 99.0, 102.0, 101.0] and all(x.size == 100.0 for x in up)
    assert [x.timestamp - t for x in up] == [timedelta(0), timedelta(minutes=7.5), timedelta(minutes=15),
                                             timedelta(minutes=27)]
    down = synthetic_trades(_bar(t, c=99.5))
    assert [x.price for x in down] == [100.0, 102.0, 99.0, 99.5]


def test_dataset_feed_orders_trades_globally(tmp_path):
    path = tmp_path / "bars.jsonl"
    n = write_synthetic_dataset(path, ["SPY", "TLT"], date(2026, 3, 2), date(2026, 3, 3))
    assert n == 2 * 2 * 13                                   # 2 séances x 13 barres x 2 symboles

    async def collect(feed):
        return [ev async for ev in feed]

    trades = asyncio.run(collect(BarDatasetFeed(path)))
    assert len(trades) == 4 * n
    assert [t.timestamp for t in trades] == sorted(t.timestamp for t in trades)
    assert trades[0].timestamp == datetime(2026, 3, 2, 14, 30, tzinfo=UTC)
    assert len(asyncio.run(collect(BarDatasetFeed(path, max_events=10)))) == 10
    closes = read_closes(path)
    assert set(closes) == {"SPY", "TLT"} and len(closes["SPY"]) == 26
    with pytest.raises(FileNotFoundError):
        BarDatasetFeed(tmp_path / "missing.jsonl")


def test_download_keeps_regular_session_and_writes_fx(tmp_path):
    calls = []

    def get(url, headers):
        q = parse_qs(urlparse(url).query)
        calls.append(q)
        start = datetime.fromisoformat(q["start"][0].replace("Z", "+00:00"))
        rows = []
        for d in range(3):
            day = (start + timedelta(days=d)).date()
            for hh, mm in ((14, 0), (14, 30), (20, 30), (21, 0)):   # 9h00 pré-marché ... 16h00 après-bourse (hiver)
                t = datetime(day.year, day.month, day.day, hh, mm, tzinfo=UTC)
                rows.append({"t": t.isoformat().replace("+00:00", "Z"), "o": 1, "h": 1, "l": 1, "c": 1, "v": 10, "n": 1})
        return {"bars": {"SPY": rows}}

    client = AlpacaHistoricalClient(key_id="k", secret_key="s", http_get=get,
                                    clock=lambda: datetime(2026, 9, 30, tzinfo=UTC))
    csv = "TIME_PERIOD,OBS_VALUE\n2026-01-02,1.10\n2026-01-05,1.12\n"
    path = tmp_path / "bars.jsonl"
    summary = download_bars(client, ["SPY"], date(2026, 1, 5), date(2026, 1, 7), path,
                            chunk_days=2, fx_get=lambda url: csv)
    assert len(calls) == 2 and all(q["timeframe"] == ["30Min"] for q in calls)
    bars = list(read_events(path))
    # seules 14h30 et 20h30 UTC (9h30 et 15h30 ET) sont dans la séance régulière
    assert {(b.timestamp.hour, b.timestamp.minute) for b in bars} == {(14, 30), (20, 30)}
    assert all(not b.payload.get("warmup") and b.source == "alpaca_dataset" for b in bars)
    assert summary["dropped_outside_session"] > 0 and summary["fx"]["rates"] == 2
    assert json.loads(fx_path(path).read_text(encoding="utf-8"))["source"] == "ecb"


# ------------------------------------------------------------------ métriques et références

def test_metrics_on_known_series():
    days = [date(2025, 1, 1) + timedelta(days=i) for i in range(366)]
    growth = [(d, 100.0 * 2 ** (i / 365)) for i, d in enumerate(days)]
    m = metrics.compute(growth)
    assert m["total_return"] == pytest.approx(1.0, rel=1e-6) and m["cagr"] == pytest.approx(1.0, rel=1e-2)
    assert m["max_drawdown"] == 0.0 and m["vol"] == pytest.approx(0.0, abs=1e-9)
    assert metrics.max_drawdown([100, 120, 60, 130, 65]) == pytest.approx(-0.5)
    flat = metrics.compute([(days[0], 1.0), (days[1], 1.0)])
    assert flat["sharpe"] is None and flat["total_return"] == 0.0
    assert metrics.compute([])["cagr"] is None
    assert metrics.turnover([(days[10], 50.0), (days[400 % 366], 50.0)], growth, days[0], days[-1]) > 0


def test_benchmarks():
    days = [date(2026, 1, 1) + timedelta(days=i) for i in range(70)]
    prices = {"A": {d: 100.0 * (1.01 ** i) for i, d in enumerate(days)}, "B": {d: 50.0 for d in days}}
    bh = buy_and_hold(days, prices, cash=1000.0, cost_bps=0.0, min_cash=0.0)
    assert bh.equity[-1][1] == pytest.approx(500 * 1.01 ** 69 + 500) and len(bh.trades) == 2
    taxed = buy_and_hold(days, prices, cash=1000.0, cost_bps=10.0, min_cash=0.0,
                         tax=lambda s, n, side, d: (n * 0.0035, n * 0.0035 / 1.1))
    assert taxed.equity[0][1] == pytest.approx(1000 - 1000 * (0.001 + 0.0035))
    assert sum(a for _, a in taxed.taxes) == pytest.approx(1000 * 0.0035 / 1.1)
    rp = monthly_inverse_vol(days, prices, cash=1000.0, cost_bps=0.0, min_cash=0.0)
    rebalance_days = sorted({d for d, _ in rp.trades})
    assert rebalance_days[0] == days[0] and {d.day for d in rebalance_days} == {1}   # 1er de chaque mois
    closes = {"A": [(datetime(2026, 1, 2, 15, tzinfo=UTC), 1.0), (datetime(2026, 1, 2, 21, tzinfo=UTC), 2.0)]}
    d, p = daily_closes(closes)
    assert d == [date(2026, 1, 2)] and p["A"][date(2026, 1, 2)] == 2.0


def test_walk_forward_picks_best_past_sharpe():
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(3 * 365)]

    def series(year_rates):
        v, out = 100.0, []
        for i, d in enumerate(days):
            if i:
                v *= 1 + year_rates[d.year] + (0.001 if i % 2 else -0.001)
            out.append((d, v))
        return out

    a = StrategyResult("A", series({2024: 0.001, 2025: -0.001, 2026: 0.0}), params={"p": "a"})
    b = StrategyResult("B", series({2024: 0.0, 2025: 0.002, 2026: 0.0005}), params={"p": "b"})
    wf, folds = walk_forward([a, b], date(2024, 1, 1), days[-1], default=b)
    assert [f["chosen"] for f in folds] == ["B", "A", "B"]       # défaut, puis meilleur passé
    assert folds[0]["reason"].startswith("défaut")
    # la série recollée suit A en 2025 (en perte)
    v_2025 = [v for d, v in wf.equity if d.year == 2025]
    assert v_2025[-1] < v_2025[0]


def test_fit_gross_after_freeze():
    current = {"A": 0.40, "B": 0.56}
    target = {"A": 0.40, "B": 0.67}          # A gelé au-dessus de sa cible (0,20)
    fitted = fit_gross_after_freeze(target, current, {"A"}, 0.98)
    assert fitted["A"] == 0.40 and sum(fitted.values()) == pytest.approx(0.98)
    assert fit_gross_after_freeze({"A": 0.3, "B": 0.3}, current, {"A"}, 0.98) == {"A": 0.3, "B": 0.3}
    over = fit_gross_after_freeze({"A": 0.7, "B": 0.6}, {"A": 0.7, "B": 0.6}, {"A"}, 0.98)
    assert over["A"] == 0.7 and over["B"] == pytest.approx(0.28)


# ------------------------------------------------------------------ bout en bout

def test_backtest_end_to_end_on_synthetic_data(tmp_path):
    data = tmp_path / "bars.jsonl"
    write_synthetic_dataset(data, ["SPY", "TLT", "GLD"], date(2025, 1, 2), date(2025, 3, 31), seed=3)
    kwargs = dict(config=DEFAULT_CONFIG_PATH, overlays=[OVERLAY], eval_start=date(2025, 3, 1),
                  grid=parse_grid(["decision.holding_period=60d,240d"]), variants=["sans_modeles"], workers=1,
                  bootstrap_samples=100)
    report = run_backtest(data, **kwargs)
    names = [r["name"] for r in report["rows"]]
    assert names[:3] == ["Buy & hold équipondéré", "Risk parity mensuelle (1/vol)", "60/40 SPY/TLT (mensuel)"]
    assert len(names) == 6 and "walk-forward" in names[-1]
    stress = report["cost_stress"]
    assert stress["multiplier"] == 2.0 and len(stress["rows"]) == 4          # 3 références + 1 moteur
    assert "coûts ×2" in stress["rows"][-1]["name"] and stress["rows"][-1]["stats"]["fills"] >= 0
    prov = report["provenance"]
    assert len(prov["dataset_sha256"]) == 64 and prov["numpy"] and prov["grid"]
    engine_row = [r for r in report["rows"] if r["kind"] == "engine:sans_modeles"][-1]
    assert set(engine_row["vs"]) == set(names[:3]) and "deflated_sharpe" in engine_row
    assert engine_row["sharpe_ci"]["sharpe"] == engine_row["metrics"]["sharpe"]
    assert 2025 in engine_row["years"]
    for r in report["rows"]:
        m = r["metrics"]
        assert m["start"] >= date(2025, 3, 1) and m["days"] >= 15 and m["cagr_eur"] is not None
    engine_rows = [r for r in report["rows"] if r["kind"] == "engine:sans_modeles"]
    for r in engine_rows:
        assert r["stats"]["handler_errors"] == 0 and r["stats"]["halts"] == 0
        assert r["stats"]["decisions"] > 100
    # Le moteur sort du cash (le bénéfice de risque couvre la TOB). Le niveau et la
    # date exacts des trades dépendent des arrondis de la plateforme (données
    # synthétiques comprises) : on vérifie seulement qu'il s'expose au marché.
    for r in engine_rows:
        assert r["stats"]["fills"] > 0 and r["metrics"]["vol"] > 0
        assert r["metrics"]["transaction_tax"] is not None
    assert report["fx"]["source"] == "fixed" and report["tax_currency"] == "EUR"
    text = format_report(report)
    assert "| Moteur sans modèles" in text and "Walk-forward" in text and "Limites" in text
    for section in ("## Le moteur bat-il les références ?", "## Rendement par année", "## Stress des coûts",
                    "## Traçabilité"):
        assert section in text
    assert json.loads(to_json(report))["eval_start"] == "2025-03-01"
    # déterministe
    again = run_backtest(data, **kwargs)
    assert [r["metrics"] for r in again["rows"]] == [r["metrics"] for r in report["rows"]]


# ------------------------------------------------------------------ statistiques de robustesse

from trading_engine.backtest import stats
from trading_engine.backtest.benchmarks import monthly_fixed
from trading_engine.backtest.dataset import import_daily_csv
from trading_engine.backtest.sources import download_csvs, stooq_daily, yahoo_daily


def test_bootstrap_and_deflated_sharpe():
    import numpy as np
    rng = np.random.default_rng(0)
    good = rng.normal(0.001, 0.01, 2000)                  # Sharpe ≈ 1,6
    ci = stats.bootstrap_sharpe(good, samples=300)
    assert ci["low"] < ci["sharpe"] < ci["high"] and ci["low"] > 0
    noise = rng.normal(0.0, 0.01, 2000)
    days = [date(2020, 1, 1) + timedelta(days=i) for i in range(2000)]
    same = stats.bootstrap_difference(dict(zip(days, good)), dict(zip(days, good)), samples=200)
    assert same["difference"] == 0 and same["p_better"] == 0
    better = stats.bootstrap_difference(dict(zip(days, good)), dict(zip(days, noise)), samples=300)
    assert better["difference"] > 0 and better["p_better"] > 0.9
    one = stats.deflated_sharpe(good, [1.6])
    many = stats.deflated_sharpe(good, [1.6, 0.5, -0.4, 1.0, 0.2, 1.3, -0.8, 0.7])
    assert one["dsr"] > 0.99 and many["dsr"] < one["dsr"] and many["sr0"] > 0
    assert stats.deflated_sharpe(noise[:5], [0.1])["dsr"] is None


def test_yearly_and_period_returns():
    series = [(date(2020, 12, 31), 100.0), (date(2021, 6, 30), 110.0), (date(2021, 12, 31), 120.0),
              (date(2022, 3, 1), 90.0), (date(2022, 12, 30), 108.0)]
    assert stats.yearly_returns(series, date(2020, 12, 31), date(2022, 12, 30)) == \
        pytest.approx({2020: 0.0, 2021: 0.2, 2022: -0.1})
    p = stats.period_stats(series, date(2022, 1, 1), date(2022, 3, 31))
    assert p["return"] == pytest.approx(-0.25) and p["max_drawdown"] == pytest.approx(-0.25)
    assert stats.period_stats(series, date(2030, 1, 1), date(2030, 2, 1)) is None


def test_sixty_forty_benchmark():
    days = [date(2026, 1, 1) + timedelta(days=i) for i in range(70)]
    prices = {"SPY": {d: 100.0 for d in days}, "TLT": {d: 50.0 for d in days}}
    r = monthly_fixed("60/40", days, prices, {"SPY": 0.6, "TLT": 0.4}, cash=1000.0, cost_bps=0.0, min_cash=0.0)
    first = [n for d, n in r.trades if d == days[0]]
    assert sorted(first) == pytest.approx([400.0, 600.0])
    assert {d.day for d, _ in r.trades} == {1}
    assert monthly_fixed("x", days, {"SPY": prices["SPY"]}, {"SPY": 0.6, "TLT": 0.4}, cash=1.0, cost_bps=0.0) is None


# ------------------------------------------------------------------ données quotidiennes

def test_import_daily_csv_adjusts_and_uses_sessions(tmp_path):
    f = tmp_path / "spy.csv"
    f.write_text("Date,Open,High,Low,Close,Adj Close,Volume\n"
                 "2025-11-28,100,102,99,101,50.5,1000\n"       # lendemain de Thanksgiving : clôture 13h
                 "2025-11-29,1,1,1,1,1,1\n"                     # samedi : ignoré
                 "2025-12-01,null,null,null,null,null,null\n"    # ligne vide
                 "2025-12-02,101,103,100,102,51,2000\n", encoding="utf-8")
    out = tmp_path / "daily.jsonl"
    summary = import_daily_csv({"SPY": f}, out)
    bars = list(read_events(out))
    assert summary["bars"] == 2 and summary["skipped"] == {"SPY": 2}
    first = bars[0]
    assert first.timeframe == "1d" and first.end.hour == 18 and first.end - first.timestamp == timedelta(hours=3, minutes=30)
    assert first.close == 50.5 and first.open == pytest.approx(50.0) and first.high == pytest.approx(51.0)
    with pytest.raises(ValueError, match="missing columns"):
        bad = tmp_path / "bad.csv"
        bad.write_text("Day,Price\n2025-01-02,1\n", encoding="utf-8")
        import_daily_csv({"X": bad}, tmp_path / "x.jsonl")


def test_yahoo_and_stooq_parsers():
    ts = int(datetime(2025, 3, 3, 14, 30, tzinfo=UTC).timestamp())
    payload = {"chart": {"error": None, "result": [{
        "timestamp": [ts, ts + 86400],
        "indicators": {"quote": [{"open": [1.0, None], "high": [2.0, None], "low": [0.5, None],
                                  "close": [1.5, None], "volume": [10, None]}],
                       "adjclose": [{"adjclose": [1.2, None]}]}}]}}
    urls = []

    def fake(url):
        urls.append(url)
        return json.dumps(payload)

    rows = yahoo_daily("SPY", date(2025, 3, 1), date(2025, 3, 5), http_get=fake)
    assert rows == [{"Date": "2025-03-03", "Open": 1.0, "High": 2.0, "Low": 0.5, "Close": 1.5,
                     "Adj Close": 1.2, "Volume": 10}]
    assert "query1.finance.yahoo.com/v8/finance/chart/SPY" in urls[0] and "interval=1d" in urls[0]
    with pytest.raises(LookupError):
        yahoo_daily("X", date(2025, 3, 1), date(2025, 3, 5),
                    http_get=lambda u: json.dumps({"chart": {"error": {"code": "Not Found"}}}))
    csv_text = "Date,Open,High,Low,Close,Volume\n2025-03-03,1,2,0.5,1.5,10\n"
    assert stooq_daily("SPY", date(2025, 3, 1), date(2025, 3, 5), http_get=lambda u: csv_text)[0]["Close"] == "1.5"
    with pytest.raises(LookupError):
        stooq_daily("SPY", date(2025, 3, 1), date(2025, 3, 5), http_get=lambda u: "No data")


def test_download_csvs_retries(tmp_path):
    calls = {"n": 0}

    def flaky(url):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("429 Too Many Requests")
        return "Date,Open,High,Low,Close,Volume\n2025-03-03,1,2,0.5,1.5,10\n"

    files = download_csvs(["SPY"], date(2025, 3, 1), date(2025, 3, 5), tmp_path, source="stooq",
                          http_get=flaky, sleep=lambda s: None)
    assert calls["n"] == 2 and files["SPY"].read_text(encoding="utf-8").startswith("Date,Open")


# ------------------------------------------------------------------ une décision par séance

def test_session_rebalancing_decides_once_per_session():
    import dataclasses
    cfg = load_config()
    cfg = dataclasses.replace(
        cfg, engine=dataclasses.replace(cfg.engine, max_events=22000, report_every=0),
        allocation=dataclasses.replace(cfg.allocation, method="hrp", rebalance_timeframe="session"),
        calendar=dataclasses.replace(cfg.calendar, mode="on"),
        risk=dataclasses.replace(cfg.risk, min_observations=5))
    from trading_engine.engine import Engine
    e = Engine(cfg)
    asyncio.run(e.run())
    sessions = [e.calendar.session_at(d.timestamp) for d in e.decisions]
    assert len(e.decisions) == 2 and all(s is not None for s in sessions)
    assert len({s.day for s in sessions}) == 2
    assert all(d.timestamp >= s.open + timedelta(minutes=5) for d, s in zip(e.decisions, sessions))
