"""Impôts différés : Reynders à la revente, taxe annuelle sur les plus-values,
récolte de l'exonération, coût de liquidation."""

import dataclasses
import json
from datetime import date, datetime, timezone

import pytest

from trading_engine.backtest.benchmarks import simulate
from trading_engine.backtest.dataset import import_daily_csv
from trading_engine.config import PROJECT_ROOT, load_config, resolve_path
from trading_engine.engine import Engine
from trading_engine.tax.profile import Instrument, load_tax_profile
from trading_engine.tax.tax_model import TaxModel, as_if_current_rules

UTC = timezone.utc
BE = PROJECT_ROOT / "config" / "taxes" / "BE.toml"
INSTRUMENTS = {
    "BOND": Instrument("BOND", asset_class="etf", domicile="IE", distribution="accumulating", bond_share=1.0),
    "EQ": Instrument("EQ", asset_class="etf", domicile="IE", distribution="accumulating", fund_withholding=0.15),
}


def _model(as_if=True):
    profile = load_tax_profile(BE)
    return TaxModel(as_if_current_rules(profile) if as_if else profile, INSTRUMENTS)


def test_reynders_due_at_sale_on_interest_accrued_while_held():
    m = _model()
    m.set_interest_index({"BOND": [["2020-01-01", 1.0], ["2021-01-01", 3.0], ["2022-01-01", 6.0]]})
    m.record_fill("BOND", 10, 100.0, date(2020, 6, 1))          # intérêts cumulés à l'achat : 1
    m.record_fill("BOND", 10, 100.0, date(2021, 6, 1))          # 3
    assert m.distribution_keep("BOND") == 1.0                    # pas de Reynders annuelle
    # Vente de 15 parts en 2022 (cumul 6), FIFO : 10 × (6 - 1) + 5 × (6 - 3) = 65
    assert m.interest_tax_on_sale("BOND", 15, 90.0, date(2022, 6, 1)) == pytest.approx(0.30 * 65)
    charge, _ = m.record_fill("BOND", -15, 90.0, date(2022, 6, 1))
    assert charge.interest_tax == pytest.approx(19.5) and m.interest_taxes_paid == pytest.approx(19.5)
    assert m.interest_tax_on_sale("EQ", 10, 100.0, date(2022, 6, 1)) == 0.0
    # Sans index : repli prudent sur la plus-value × part en créances
    m2 = _model()
    m2.record_fill("BOND", 10, 100.0, date(2020, 6, 1))
    assert m2.interest_tax_on_sale("BOND", 10, 120.0, date(2022, 6, 1)) == pytest.approx(0.30 * 200)
    assert m2.interest_tax_on_sale("BOND", 10, 80.0, date(2022, 6, 1)) == 0.0
    # La décision voit la Reynders d'une vente
    cost = m.rebalance_cost({"BOND": -500.0}, {"BOND": 100.0}, date(2022, 7, 1))
    assert cost.interest_tax == pytest.approx(0.30 * 5 * 3) and cost.total > cost.transaction_tax


def test_gains_tax_rules_and_year_cache():
    m = _model(as_if=False)
    assert m.gains_tax_due(2024) == 0.0                          # avant 2026 : hors régime
    m = _model()
    m.record_fill("EQ", 100, 100.0, date(2020, 1, 2))
    m.record_fill("EQ", -100, 250.0, date(2020, 6, 1))           # +15 000
    assert m.gains.net_gain(2020) == pytest.approx(15_000)
    assert m.gains_tax_due(2020) == pytest.approx(0.10 * 5_000)  # exonération 10 000
    snap = m.snapshot()
    m2 = _model()
    m2.restore(snap)
    assert m2.gains.net_gain(2020) == pytest.approx(15_000)


def test_harvest_plan_uses_remaining_exemption_only():
    m = _model()
    m.record_fill("EQ", 1_000, 100.0, date(2020, 1, 2))
    m.record_fill("BOND", 1_000, 100.0, date(2020, 1, 2))
    positions = {"EQ": (1_000, 150.0), "BOND": (1_000, 130.0)}   # +50 000 latents ; BOND exclu (Reynders)
    plan = m.harvest_plan(positions, date(2020, 12, 15))
    assert [(s, round(q, 6), round(g, 6)) for s, q, g in plan] == [("EQ", 200.0, 10_000.0)]
    m.record_fill("EQ", -200, 150.0, date(2020, 12, 15))
    m.record_fill("EQ", 200, 150.0, date(2020, 12, 15))
    assert m.gains_tax_due(2020) == 0.0                          # gain exonéré, base remontée
    assert m.harvest_plan(positions, date(2020, 12, 16)) == []   # exonération épuisée
    # Gain trop faible pour payer la TOB aller-retour : pas de récolte
    m3 = _model()
    m3.record_fill("EQ", 1_000, 100.0, date(2020, 1, 2))
    assert m3.harvest_plan({"EQ": (1_000, 100.1)}, date(2020, 12, 15)) == []


def test_liquidation_cost_includes_all_sale_taxes():
    m = _model()
    m.set_interest_index({"BOND": [["2020-01-01", 0.0], ["2023-01-01", 9.0]]})
    m.record_fill("EQ", 1_000, 100.0, date(2020, 1, 2))
    m.record_fill("BOND", 100, 100.0, date(2020, 1, 2))
    cost = m.liquidation_cost({"EQ": (1_000, 130.0), "BOND": (100, 100.0)}, date(2023, 6, 1))
    tob = 0.0012 * 130_000 + 0.0012 * 10_000
    assert cost == pytest.approx(tob + 0.10 * (30_000 - 10_000) + 0.30 * 900)


def test_benchmark_pays_gains_tax_and_reports_liquidation():
    days = [date(2020, 1, 2), date(2020, 6, 1), date(2021, 1, 4), date(2021, 2, 1)]
    prices = {"EQ": dict(zip(days, [100.0, 200.0, 200.0, 200.0]))}
    targets = {0: {"EQ": 1.0}, 1: {"EQ": 0.5}}                   # vente de la moitié à 200
    r = simulate("x", days, prices, lambda i: targets.get(i), cash=100_000, cost_bps=0.0, tax_model=_model())
    # Achat : TOB 120 ; vente de (200 000 - 99 940) / 200 = 500,3 parts, gain 50 030.
    assert r.stats["gains_tax_paid"] == pytest.approx(0.10 * (50_030 - 10_000))
    assert r.stats["liquidation_cost"] > 0


def test_dataset_interest_index_from_distributions(tmp_path):
    f = tmp_path / "b.csv"
    f.write_text("Date,Open,High,Low,Close,Adj Close,Volume,Dividend\n"
                 "2025-03-03,100,100,100,100,100,10,0\n"
                 "2025-03-04,100,100,100,100,100,10,1\n"
                 "2025-03-05,100,100,100,100,100,10,1\n", encoding="utf-8")
    import_daily_csv({"BOND": f}, tmp_path / "b.jsonl", keep={"BOND": 1.0}, interest={"BOND": 1.0})
    meta = json.loads((tmp_path / "b.jsonl.meta.json").read_text(encoding="utf-8"))
    idx = meta["interest_index"]["BOND"]
    assert idx[0] == ["2025-03-04", pytest.approx(1.0)]
    assert idx[1][1] == pytest.approx(1.0 + 1.0 * 101 / 100)      # par part de l'indice réinvesti


def test_engine_settles_gains_tax_at_year_change_and_harvests():
    cfg = load_config("config/config.yaml", (PROJECT_ROOT / "config" / "tax_ucits.yaml",))
    cfg = dataclasses.replace(cfg, tax=dataclasses.replace(cfg.tax, as_if_current_rules=True,
                                                           harvest_exemption=True),
                              fx=dataclasses.replace(cfg.fx, provider="off"))
    e = Engine(cfg)
    assert e.tax.profile.capital_gains.effective_from is None
    sym = "SPY"
    e.portfolio.cash = 0.0
    t0 = datetime(2020, 1, 2, 15, tzinfo=UTC)
    e.record_fill(sym, 1_000, 100.0, t0)
    e.record_fill(sym, -500, 150.0, datetime(2020, 6, 1, 15, tzinfo=UTC))     # +25 000
    e._settle_tax_year(t0)
    cash = e.portfolio.cash
    e._settle_tax_year(datetime(2021, 1, 4, 15, tzinfo=UTC))
    assert cash - e.portfolio.cash == pytest.approx(0.10 * 15_000) and e.gains_tax_paid == pytest.approx(1_500)
    # Décembre 2021 : récolte sur les 500 parts restantes (+25 000 latents, 10 000 exonérés)
    e.portfolio.update_price(sym, 150.0, datetime(2021, 12, 13, 15, tzinfo=UTC))
    e._maybe_harvest(datetime(2021, 12, 13, 16, tzinfo=UTC))
    assert e.harvested_gains == pytest.approx(10_000) and e.harvest_cost > 0
    assert e.tax.gains_tax_due(2021) == 0.0
    assert e.tax.lot_quantity(sym) == pytest.approx(500)
    e._maybe_harvest(datetime(2021, 12, 14, 16, tzinfo=UTC))   # une fois par an
    assert e.harvested_gains == pytest.approx(10_000)
