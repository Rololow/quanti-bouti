from datetime import date, datetime, timezone

import pytest

from trading_engine.config import PROJECT_ROOT
from trading_engine.tax.capital_gains import CapitalGainsTracker
from trading_engine.tax.profile import CapitalGainsRules, Instrument, TaxProfile, load_tax_profile
from trading_engine.tax.tax_model import TaxModel

BE = PROJECT_ROOT / "config" / "taxes" / "BE.toml"

INSTRUMENTS = {
    "IWDA": Instrument("IWDA", "etf", "IE", "accumulating"),
    "VWRL": Instrument("VWRL", "etf", "IE", "distributing"),
    "SPY": Instrument("SPY", "etf", "US", "distributing"),
    "ABI": Instrument("ABI", "stock", "BE"),
    "OLO": Instrument("OLO", "bond", "BE"),
    "BEVEK": Instrument("BEVEK", "fund", "LU", "accumulating", registered_locally=True),
}


@pytest.fixture
def be():
    return TaxModel(load_tax_profile(BE), INSTRUMENTS)


# ------------------------------------------------------------------ profil

def test_belgian_profile_loads():
    p = load_tax_profile(BE)
    assert p.country == "BE" and p.currency == "EUR"
    assert not p.verified                     # à vérifier avant usage réel
    assert "IE" in p.regions["EEA"] and "US" not in p.regions["EEA"]
    assert p.capital_gains.rate == 0.10
    assert p.capital_gains.effective_from == date(2026, 1, 1)


def test_profile_validation():
    with pytest.raises(ValueError):
        TaxProfile.from_dict({})
    with pytest.raises(ValueError):
        TaxProfile.from_dict({"meta": {"country": "XX"},
                              "transaction_tax": {"rules": [{"id": "r", "rate": 35}]}})


def test_other_country_profile_is_just_data():
    """Un autre pays = un autre fichier : ici un pays sans taxe de bourse."""
    p = TaxProfile.from_dict({
        "meta": {"country": "XX"},
        "transaction_tax": {"rules": [{"id": "none", "rate": 0.0, "match": {}}]},
        "capital_gains": {"rate": 0.3},
    })
    m = TaxModel(p)
    assert m.transaction_tax("ANY", 10_000, "buy").amount == 0.0
    m.record_fill("ANY", 10, 100, date(2026, 1, 1))
    m.record_fill("ANY", -10, 150, date(2026, 6, 1))
    assert m.gains.tax_due(2026) == pytest.approx(500 * 0.3)


# ------------------------------------------------------------------ TOB

@pytest.mark.parametrize("symbol,rule,rate", [
    ("IWDA", "etf_eea", 0.0012),
    ("VWRL", "etf_eea", 0.0012),
    ("SPY", "shares_and_other_etf", 0.0035),
    ("ABI", "shares_and_other_etf", 0.0035),
    ("OLO", "bond", 0.0012),
    ("BEVEK", "accumulating_fund_registered_in_belgium", 0.0132),
])
def test_tob_rules(be, symbol, rule, rate):
    charge = be.transaction_tax(symbol, 10_000, "buy")
    assert charge.rule_id == rule
    assert charge.amount == pytest.approx(10_000 * rate)
    assert be.transaction_tax(symbol, 10_000, "sell").amount == pytest.approx(charge.amount)


def test_tob_caps(be):
    assert be.transaction_tax("ABI", 1_000_000, "buy").amount == 1600.0
    assert be.transaction_tax("IWDA", 2_000_000, "buy").amount == 1300.0
    assert be.transaction_tax("BEVEK", 1_000_000, "buy").amount == 4000.0
    assert be.transaction_tax("ABI", 1_000_000, "buy").capped


def test_unknown_instrument_uses_prudent_default(be):
    charge = be.transaction_tax("MYSTERY", 10_000, "buy")
    assert charge.rule_id == "default" and charge.rate == 0.0035
    assert any("MYSTERY" in w for w in be.warnings())


# ------------------------------------------------------------------ revenus

def test_foreign_dividend(be):
    div = be.dividend_tax("SPY", 100.0)
    assert div.foreign_withholding == pytest.approx(15.0)       # traité US
    assert div.local_tax == pytest.approx(0.30 * 85.0)          # précompte sur le net
    assert div.net == pytest.approx(59.5)


def test_domestic_dividend(be):
    div = be.dividend_tax("ABI", 100.0)
    assert div.foreign_withholding == 0.0 and div.local_tax == pytest.approx(30.0)


def test_account_tax(be):
    assert be.account_tax(900_000) == 0.0
    assert be.account_tax(1_200_000) == pytest.approx(1_800.0)


# ------------------------------------------------------------------ plus-values

RULES = CapitalGainsRules(
    rate=0.10, effective_from=date(2026, 1, 1), annual_exemption=10_000,
    carry_forward_per_year=1_000, carry_forward_max_years=5,
    step_up_date=date(2025, 12, 31), step_up_keeps_higher_cost=True,
)


def test_fifo_and_same_year_loss_offset():
    t = CapitalGainsTracker(RULES)
    t.buy("A", 100, 100, date(2026, 1, 10))
    t.buy("A", 100, 200, date(2026, 2, 10))
    g = t.sell("A", 150, 300, date(2026, 3, 1))        # 100 @100 + 50 @200
    assert g.tax_basis == pytest.approx(100 * 100 + 50 * 200)
    assert g.gain == pytest.approx(45_000 - 20_000)
    t.buy("B", 100, 100, date(2026, 1, 10))
    t.sell("B", 100, 50, date(2026, 4, 1))              # perte de 5 000
    assert t.net_gain(2026) == pytest.approx(20_000)
    assert t.tax_due(2026) == pytest.approx(0.10 * (20_000 - 10_000))


def test_historic_gains_are_frozen_by_step_up():
    t = CapitalGainsTracker(RULES, step_up_prices={"A": 180, "B": 80})
    t.buy("A", 10, 100, date(2020, 1, 1))              # basis = 180 (valeur au 31/12/2025)
    t.buy("B", 10, 100, date(2020, 1, 1))              # coût réel plus élevé conservé
    assert t.sell("A", 10, 200, date(2026, 6, 1)).gain == pytest.approx(10 * 20)
    assert t.sell("B", 10, 90, date(2026, 6, 1)).gain == pytest.approx(-10 * 10)


def test_sales_before_effective_date_are_not_taxed():
    t = CapitalGainsTracker(RULES)
    t.buy("A", 10, 100, date(2025, 1, 1))
    t.sell("A", 10, 200, date(2025, 6, 1))
    assert t.net_gain(2025) == 0.0 and t.tax_due(2025) == 0.0


def test_exemption_carry_forward():
    t = CapitalGainsTracker(RULES)
    assert t.exemption(2026) == 10_000
    assert t.exemption(2027) == 11_000                  # 2026 inutilisé : +1 000
    assert t.exemption(2032) == 15_000                  # plafond 5 ans
    t.buy("A", 100, 100, date(2027, 1, 1))
    t.sell("A", 100, 210, date(2027, 6, 1))             # gain 11 000 : tout l'exo 2027 consommé
    # 2027 : 10 000 d'exo annuelle + 1 000 de réserve consommés, rien d'inutilisé
    assert t.exemption(2028) == 10_000


def test_marginal_tax_of_sale_does_not_modify_lots():
    t = CapitalGainsTracker(RULES)
    t.buy("A", 100, 100, date(2026, 1, 1))
    t.sell("A", 50, 400, date(2026, 2, 1))              # gain 15 000 (5 000 taxables)
    before = [lot.quantity for lot in t.lots["A"]]
    assert t.marginal_tax_of_sale("A", 50, 200, date(2026, 3, 1)) == pytest.approx(0.10 * 5_000)
    assert [lot.quantity for lot in t.lots["A"]] == before


def test_rebalance_cost_combines_tob_and_capital_gains(be):
    be.record_fill("ABI", 100, 100.0, date(2026, 1, 5))
    be.record_fill("ABI", -50, 400.0, date(2026, 2, 1))  # gain 15 000
    cost = be.rebalance_cost({"ABI": -10_000, "IWDA": 10_000}, {"ABI": 400.0, "IWDA": 100.0},
                             datetime(2026, 3, 1, tzinfo=timezone.utc))
    assert cost.transaction_tax == pytest.approx(35.0 + 12.0)
    # vente de 25 titres (base 100) à 400 : +7 500 de gain, entièrement taxable
    assert cost.capital_gains_tax == pytest.approx(750.0)
    assert be.transaction_taxes_paid == pytest.approx(35.0 + 70.0)
