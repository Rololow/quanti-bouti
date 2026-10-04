import asyncio
import dataclasses
import math
from datetime import timedelta

import numpy as np
import pytest

from trading_engine.allocation.allocator import RiskAllocator
from trading_engine.config import load_config
from trading_engine.data.market_feed import SimulatedMarketFeed
from trading_engine.engine import Engine
from trading_engine.models.ensemble import ModelEnsemble
from trading_engine.models.regime import RegimeModel
from trading_engine.models.reliability import ContextReliability
from trading_engine.robustness.perturbation import PerturbedFeed
from trading_engine.robustness.stress import stress_test
from trading_engine.safety.safety_engine import ModelHealth, SafetyEngine, SafetyState
from trading_engine.signals.signal import Signal

COV = np.array([
    [0.040, 0.030, 0.000, 0.000],
    [0.030, 0.045, 0.000, 0.000],
    [0.000, 0.000, 0.010, 0.002],
    [0.000, 0.000, 0.002, 0.020],
])
SYMBOLS = ["SPY", "QQQ", "TLT", "GLD"]


# ------------------------------------------------------------------ MODEL_DEGRADED

def _regime_obs(rng, vol):
    r = rng.normal(0, vol)
    return np.array([r, math.log(vol) + rng.normal(0, 0.05)])


def test_regime_not_degraded_on_stationary_data(t0):
    rng = np.random.default_rng(0)
    model = RegimeModel("HF", n_states=2, min_samples=100, window=300, refit_every=50)
    for i in range(1500):
        vol = 0.01 if (i // 150) % 2 == 0 else 0.02     # régimes connus qui alternent
        model.update(_regime_obs(rng, vol), t0)
    assert model.degraded_count <= 1


def test_regime_degraded_when_data_leaves_learned_world(t0):
    rng = np.random.default_rng(1)
    # Ajustement unique sur 300 observations couvrant les deux régimes connus.
    model = RegimeModel("HF", n_states=2, min_samples=300, window=2000, refit_every=10_000)
    for i in range(600):
        model.update(_regime_obs(rng, 0.01 if (i // 100) % 2 == 0 else 0.02), t0)
    assert not model.degraded
    for _ in range(40):                                  # volatilité ×10 jamais vue
        state = model.update(_regime_obs(rng, 0.2), t0)
    assert state.degraded and model.fit_zscore < -model.degraded_z


def test_degraded_model_freezes_symbol(t0):
    se = SafetyEngine()
    st = se.evaluate(t0, model_health=ModelHealth(degraded=[("SPY", "regime HF z=-6")]))
    assert st.state is SafetyState.DEGRADED and st.frozen_symbols == {"SPY"}


# ------------------------------------------------------------------ fiabilité par contexte

def test_context_reliability_shrinks_to_global():
    rel = ContextReliability(lam=0.99, prior_strength=50)
    rng = np.random.default_rng(2)
    for _ in range(2000):                                 # bon modèle en LOW_VOL
        x = rng.normal()
        rel.update("LOW_VOL", x + rng.normal(0, 0.3), x, 0.3)
    for _ in range(5):                                    # très peu de données en CRISIS, mauvaises
        x = rng.normal()
        rel.update("CRISIS", -x, x, 0.3)
    assert rel.skill("LOW_VOL") > 0.8
    crisis = rel._monitors["CRISIS"].skill
    assert crisis < 0
    # peu d'observations : fortement ramené vers le global
    assert rel.skill("CRISIS") > crisis
    assert rel.skill("NEVER_SEEN") == rel.skill()
    assert 0.0 <= rel.reliability("CRISIS") <= 1.0


# ------------------------------------------------------------------ ensemble

def _feed_errors(ens, rng, n, rho, scales=(1.0, 1.0)):
    for _ in range(n):
        a = rng.normal()
        b = rho * a + math.sqrt(1 - rho**2) * rng.normal()
        ens.record({"m1": a * scales[0], "m2": b * scales[1]})


def test_correlated_errors_count_as_one_model():
    ens = ModelEnsemble(["m1", "m2"], lam=0.999, min_samples=30)
    _feed_errors(ens, np.random.default_rng(3), 3000, rho=0.95)
    assert ens.effective_models() == pytest.approx(1.0, abs=0.1)
    indep = ModelEnsemble(["m1", "m2"], lam=0.999, min_samples=30)
    _feed_errors(indep, np.random.default_rng(4), 3000, rho=0.0)
    assert indep.effective_models() == pytest.approx(2.0, abs=0.15)


def test_ensemble_prudent_before_enough_samples():
    ens = ModelEnsemble(["m1", "m2"], min_samples=30)
    assert ens.effective_models() == 1.0
    assert ens.weights() == pytest.approx([0.5, 0.5])


def test_ensemble_weights_favor_more_accurate_model():
    ens = ModelEnsemble(["m1", "m2"], lam=0.999, shrinkage=0.0, min_samples=30)
    _feed_errors(ens, np.random.default_rng(5), 3000, rho=0.0, scales=(1.0, 3.0))
    w = ens.weights()
    assert w[0] > 0.8


def test_disagreement_widens_uncertainty(t0):
    ens = ModelEnsemble(["m1", "m2"])
    def sig(mean):
        return Signal("SPY", "30m", mean=mean, std=0.001, n_obs=100, timestamp=t0,
                      source="x", reliability=0.2)
    agree = ens.combine({"m1": sig(0.002), "m2": sig(0.002)})
    disagree = ens.combine({"m1": sig(0.006), "m2": sig(-0.002)})
    assert agree.mean == pytest.approx(disagree.mean)
    assert disagree.std > agree.std
    assert disagree.disagreement == pytest.approx(0.004)
    assert agree.reliability == pytest.approx(0.2)
    assert ens.combine({"m1": None, "m2": None}) is None


# ------------------------------------------------------------------ allocation et fiabilité

def test_unreliable_signal_gets_no_budget(t0):
    def s(sym, rel):
        return Signal(sym, "30m", mean=0.002, std=0.01, n_obs=100, timestamp=t0,
                      source="x", reliability=rel)
    signals = {"SPY": s("SPY", 0.3), "QQQ": s("QQQ", 0.0), "TLT": s("TLT", None), "GLD": None}
    budgets = RiskAllocator("signal").signal_budgets(SYMBOLS, signals, skill=None)
    assert budgets[0] > 0
    assert budgets[1] == 0 and budgets[2] == 0 and budgets[3] == 0


# ------------------------------------------------------------------ stress tests

def test_risk_parity_is_robust():
    alloc = RiskAllocator("risk_parity", target_vol=0.1)
    report = stress_test(lambda c, s: alloc.allocate(SYMBOLS, c, s, None)[0], SYMBOLS, COV, {})
    assert report.score > 0.8
    assert {"corr_up", "corr_down", "vol_up:SPY"} <= set(report.instability)


def test_fragile_allocation_is_detected():
    def all_in_lowest_vol(cov, signals):
        i = int(np.argmin(np.diag(cov)))
        return {s: 1.0 if k == i else 0.0 for k, s in enumerate(SYMBOLS)}
    report = stress_test(all_in_lowest_vol, SYMBOLS, COV, {})
    assert report.score == pytest.approx(0.0)       # vol_up:TLT fait tout basculer
    assert report.worst == "vol_up:TLT"


def test_empty_allocation_is_trivially_stable():
    report = stress_test(lambda c, s: {}, SYMBOLS, COV, {})
    assert report.score == 1.0 and report.worst is None


# ------------------------------------------------------------------ bout en bout

def _config(**alloc):
    cfg = load_config()
    return dataclasses.replace(
        cfg, engine=dataclasses.replace(cfg.engine, max_events=8000, report_every=0),
        allocation=dataclasses.replace(cfg.allocation, **alloc),
        risk=dataclasses.replace(cfg.risk, min_observations=5),  # 8000 événements = 11 h
    )


def _sim_feed(cfg):
    return SimulatedMarketFeed(
        {s: cfg.portfolio.positions[s].avg_price if s in cfg.portfolio.positions else 100.0
         for s in cfg.feed.symbols},
        seed=cfg.feed.seed, tick_seconds=cfg.feed.tick_seconds, max_events=cfg.engine.max_events,
    )


@pytest.mark.parametrize("method", ["risk_parity", "hrp"])
def test_small_perturbations_do_not_change_the_target(method):
    """Un bruit de prix minuscule ne change ni la cible (au-delà d'une réponse
    petite et linéaire), ni les décisions de sécurité / d'intégrité.

    La réponse est mesurée pour deux amplitudes : un écart ×10 doit donner une
    réponse ≈ ×10 (pas de saut), et rester faible à 1e-6. Un bruit relatif sur
    les prix est amplifié dans les rendements (qui sont petits) : on ne peut
    pas exiger ||ΔS|| << ||δ|| en valeur absolue sur les prix.
    """
    cfg = _config(method=method)
    base = Engine(cfg, feed=_sim_feed(cfg))
    asyncio.run(base.run())

    diffs = {}
    for eps in (1e-7, 1e-6):
        noisy = Engine(cfg, feed=PerturbedFeed(_sim_feed(cfg), price_noise=eps, seed=7))
        asyncio.run(noisy.run())
        w0, w1 = base.last_allocation.weights, noisy.last_allocation.weights
        diffs[eps] = sum(abs(w0[s] - w1.get(s, 0.0)) for s in w0)
        assert noisy.safety.status.state == base.safety.status.state
        assert noisy.integrity.counts == base.integrity.counts

    assert diffs[1e-6] < 5e-3
    assert 5 < diffs[1e-6] / diffs[1e-7] < 20       # réponse linéaire, pas de discontinuité


def test_low_robustness_damps_the_move():
    cfg = _config(method="risk_parity")
    cfg = dataclasses.replace(cfg, robustness=dataclasses.replace(cfg.robustness, min_score=1.01))
    engine = Engine(cfg)
    asyncio.run(engine.run())
    alloc = engine.last_allocation
    assert alloc.robustness is not None and alloc.robustness < 1.01
    steps = alloc.attribution["SPY"]
    assert "robustness" in steps
    assert sum(steps.values()) == pytest.approx(alloc.weights["SPY"])
