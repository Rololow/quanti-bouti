import asyncio
import dataclasses
from datetime import datetime, timezone

import numpy as np
import pytest

from trading_engine.alerts.alerts import AlertConfig, AlertEngine
from trading_engine.config import PROJECT_ROOT, load_config
from trading_engine.decision.rebalance import (
    UDONOTHING,
    UREBALANCE,
    DecisionConfig,
    DecisionContext,
    DecisionEngine,
)
from trading_engine.engine import Engine
from trading_engine.models.regime import RegimeState
from trading_engine.risk.limits import RiskBreach
from trading_engine.signals.signal import Signal
from trading_engine.storage.decision_log import read_decisions
from trading_engine.tax.profile import Instrument, load_tax_profile
from trading_engine.tax.tax_model import TaxModel

T = datetime(2026, 3, 2, 15, 0, tzinfo=timezone.utc)
COV = np.array([[0.04, 0.0], [0.0, 0.01]])


def ctx(**kw):
    base = dict(
        timestamp=T, portfolio_value=1_000_000.0,
        current={"A": 0.0, "B": 0.5}, target={"A": 0.5, "B": 0.5},
        symbols=["A", "B"], cov=COV, prices={"A": 100.0, "B": 50.0},
        spreads={"A": 0.0, "B": 0.0},
    )
    base.update(kw)
    return DecisionContext(**base)


def engine(**cfg):
    params = dict(default_spread_bps=0.0, slippage_bps=0.0, holding_period="5d",
                  risk_aversion=5.0, k_sigma=1.0)
    params.update(cfg)
    return DecisionEngine(DecisionConfig(**params))


# ------------------------------------------------------------------ filtres

def test_halted_means_do_nothing():
    d = engine().decide(ctx(safety_state="HALTED"))
    assert d.action == UDONOTHING and "HALTED" in d.reasons[0]


def test_no_target_or_no_covariance():
    assert engine().decide(ctx(target=None)).action == UDONOTHING
    assert engine().decide(ctx(cov=None)).action == UDONOTHING


def test_no_trade_band_hysteresis():
    d = engine(no_trade_band=0.02).decide(ctx(current={"A": 0.49, "B": 0.5}))
    assert d.action == UDONOTHING
    assert "bande de non-trading" in d.symbols["A"].excluded


def test_frozen_and_bad_data_symbols_are_excluded():
    d = engine().decide(ctx(frozen=frozenset({"A"})))
    assert d.action == UDONOTHING and d.symbols["A"].excluded.startswith("gelé")
    d = engine().decide(ctx(data_scores={"A": 0.5, "B": 1.0}))
    assert d.symbols["A"].excluded.startswith("qualité")


def test_unrobust_target_means_do_nothing():
    d = engine(min_robustness=0.3).decide(ctx(robustness=0.1))
    assert d.action == UDONOTHING and "non robuste" in d.reasons[0]


# ------------------------------------------------------------------ économie

def test_cheap_rebalance_goes_all_the_way():
    d = engine().decide(ctx())
    assert d.action == UREBALANCE and d.fraction == 1.0
    assert d.symbols["A"].execution_weight == pytest.approx(0.5)
    assert d.symbols["A"].notional == pytest.approx(500_000)
    assert d.symbols["B"].notional == 0.0
    # TE² = 0.25 * 0.04 = 0.01 ; bénéfice = V γ/2 H TE²
    assert d.risk_benefit == pytest.approx(1e6 * 2.5 * (1 / 50.4) * 0.01)
    assert d.u_donothing == pytest.approx(-d.risk_benefit)


def test_costs_can_make_doing_nothing_optimal():
    d = engine(slippage_bps=50.0).decide(ctx())
    assert d.action == UDONOTHING
    assert d.reasons[0].startswith("le bénéfice ne couvre pas")
    assert all(s.execution_weight == s.current_weight for s in d.symbols.values())


def test_partial_rebalance_emerges_from_costs():
    # Bénéfice S(2f - f²), coût C·f : optimum f* = 1 - C / (2S) = 0.5 si C = S.
    S = 1e6 * 2.5 * (1 / 50.4) * 0.01
    slippage_bps = S / 500_000 * 1e4
    d = engine(slippage_bps=slippage_bps, fractions=(0.25, 0.5, 1.0)).decide(ctx())
    assert d.action == UREBALANCE and d.fraction == 0.5
    assert d.symbols["A"].execution_weight == pytest.approx(0.25)


def test_reliable_alpha_adds_benefit_and_urgency():
    sig = Signal("A", "30m", mean=0.001, std=0.002, n_obs=500, timestamp=T, source="x", reliability=0.5)
    with_alpha = engine().decide(ctx(signals={"A": sig}))
    unreliable = dataclasses.replace(sig, reliability=0.0)
    without = engine().decide(ctx(signals={"A": unreliable}))
    assert with_alpha.alpha_benefit == pytest.approx(500_000 * 0.001 * 0.5)
    assert without.alpha_benefit == 0.0
    assert with_alpha.urgency > 0 and without.urgency == 0
    # σ = erreur-type de μ (std / sqrt(n)), pas le bruit du rendement
    assert with_alpha.uncertainty == pytest.approx(500_000 * 0.002 / np.sqrt(500))


def test_taxes_are_part_of_the_cost():
    tax = TaxModel(load_tax_profile(PROJECT_ROOT / "config/taxes/BE.toml"),
                   {"A": Instrument("A", "stock", "BE"), "B": Instrument("B", "stock", "BE")})
    d = DecisionEngine(DecisionConfig(default_spread_bps=0.0, slippage_bps=0.0), tax).decide(ctx())
    # TOB actions 0,35 % : même la meilleure option (25 % du chemin, 125 000)
    # coûte plus que la réduction de l'écart de risque qu'elle apporte.
    assert d.action == UDONOTHING
    assert d.evaluated_fraction == 0.25
    assert d.costs.transaction_tax == pytest.approx(125_000 * 0.0035)
    assert d.costs.transaction_tax > d.risk_benefit
    # sans taxe, la même situation mène à un UREBALANCE complet
    assert engine().decide(ctx()).action == UREBALANCE


def test_decision_ids_are_unique_and_reasons_explain():
    e = engine()
    a, b = e.decide(ctx()), e.decide(ctx())
    assert a.decision_id != b.decision_id
    assert any("TE" in r for r in a.reasons) and any("coûts" in r for r in a.reasons)


# ------------------------------------------------------------------ alertes

def _regime(label, conf=0.95, degraded=False):
    other = "HIGH_VOL" if label == "LOW_VOL" else "LOW_VOL"
    return RegimeState("MT", {label: conf, other: 1 - conf}, T, 200, degraded)


def test_persistent_conditions_alert_once():
    ae = AlertEngine()
    breach = RiskBreach("DRIFT", "A", 0.1, 0.05)
    assert [a.kind for a in ae.evaluate(T, breaches=[breach])] == ["POSITION_DRIFT"]
    assert ae.evaluate(T, breaches=[breach]) == []
    ae.evaluate(T, breaches=[])
    assert len(ae.evaluate(T, breaches=[breach])) == 1


def test_volatility_and_correlation_spikes():
    ae = AlertEngine(AlertConfig(vol_fast="5m", vol_slow="1h", vol_spike_ratio=2.0))
    # vol 5m × sqrt(12) = 3.46 × 0.01 = 0.0346 > 2 × 0.01
    alerts = ae.evaluate(T, volatilities={"A": (0.01, 0.01), "B": (0.001, 0.01)})
    assert [(a.kind, a.symbol) for a in alerts] == [("VOLATILITY_SPIKE", "A")]
    low = np.array([[1, 0.1], [0.1, 1]])
    high = np.array([[1, 0.8], [0.8, 1]])
    ae.evaluate(T, correlation=low)
    assert any(a.kind == "CORRELATION_SPIKE" for a in ae.evaluate(T, correlation=high))


def test_regime_change_filters_horizon_and_confidence():
    ae = AlertEngine(AlertConfig(regime_horizons=("MT",), regime_min_confidence=0.8))
    ae.evaluate(T, regimes={"A": {"MT": _regime("LOW_VOL"), "HF": _regime("LOW_VOL")}})
    assert ae.evaluate(T, regimes={"A": {"MT": _regime("HIGH_VOL", conf=0.6)}}) == []   # ambigu
    alerts = ae.evaluate(T, regimes={"A": {"MT": _regime("HIGH_VOL")}})
    assert [a.kind for a in alerts] == ["REGIME_CHANGE"]
    hf_only = ae.evaluate(T, regimes={"A": {"HF": _regime("HIGH_VOL")}})
    assert hf_only == []


def test_signal_reversal_disagreement_and_safety():
    ae = AlertEngine()
    def sig(mean, dis=None):
        return Signal("A", "30m", mean=mean, std=0.001, n_obs=100, timestamp=T, source="e", disagreement=dis)
    ae.evaluate(T, signals={"A": sig(0.001)}, safety_state="NORMAL")
    kinds = {a.kind for a in ae.evaluate(T, signals={"A": sig(-0.001, dis=0.0008)}, safety_state="HALTED")}
    assert kinds == {"SIGNAL_REVERSAL", "SIGNAL_DISAGREEMENT", "SAFETY_STATE"}


# ------------------------------------------------------------------ bout en bout

def test_engine_decides_every_interval_and_logs(tmp_path):
    cfg = load_config()
    log = tmp_path / "decisions.jsonl"
    cfg = dataclasses.replace(
        cfg,
        engine=dataclasses.replace(cfg.engine, max_events=8000, report_every=0),
        allocation=dataclasses.replace(cfg.allocation, method="hrp"),
        storage=dataclasses.replace(cfg.storage, decision_log=str(log)),
    )
    e = Engine(cfg)
    events = []
    from trading_engine.data.events import EventType
    e.bus.subscribe(EventType.DECISION, events.append)
    asyncio.run(e.run())

    assert e.decisions and len(events) == len(e.decisions)
    records = read_decisions(log)
    assert len(records) == len(e.decisions)
    last = records[-1]
    assert last["decision_id"] == e.last_decision.decision_id
    assert last["action"] in (UREBALANCE, UDONOTHING) and last["reasons"]
    assert "total" in last["costs"]
