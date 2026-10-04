import math
from datetime import timedelta

import numpy as np
import pytest

from trading_engine.models import hmm
from trading_engine.models.monitoring import ModelMonitor, PageHinkley
from trading_engine.models.online_factors import EWMAStandardizer, OnlineFactorModel
from trading_engine.models.online_regression import OnlineLinearRegression
from trading_engine.models.regime import RegimeModel
from trading_engine.signals.signal import Signal


# ------------------------------------------------------------------ signal

def test_signal_uncertainty(t0):
    sg = Signal("SPY", "30m", mean=0.002, std=0.001, n_obs=100, timestamp=t0, source="t")
    assert sg.t_stat == pytest.approx(2.0)
    assert sg.prob_positive == pytest.approx(0.97725, abs=1e-4)
    assert sg.net_of(0.0025) == pytest.approx(-0.0005)  # alpha net négatif
    with pytest.raises(ValueError):
        Signal("SPY", "30m", mean=0.0, std=0.0, n_obs=1, timestamp=t0, source="t")


# ------------------------------------------------------------------ regression

def test_regression_recovers_coefficients():
    rng = np.random.default_rng(0)
    model = OnlineLinearRegression(2, lam=1.0, ridge=1e-3, min_samples=20)
    for _ in range(2000):
        x = rng.normal(size=2)
        model.update(x, 0.5 + 2.0 * x[0] - 1.0 * x[1] + rng.normal(scale=0.1))
    assert model.coef == pytest.approx([0.5, 2.0, -1.0], abs=0.02)
    mean, std = model.predict(np.array([1.0, 1.0]))
    assert mean == pytest.approx(1.5, abs=0.05)
    assert std == pytest.approx(0.1, rel=0.3)


def test_regression_requires_min_samples():
    model = OnlineLinearRegression(1, min_samples=5)
    for i in range(4):
        model.update(np.array([float(i)]), 1.0)
        assert model.predict(np.array([1.0])) is None
    model.update(np.array([1.0]), 1.0)
    assert model.predict(np.array([1.0])) is not None


def test_forgetting_adapts_to_concept_drift():
    rng = np.random.default_rng(1)

    def run(lam):
        model = OnlineLinearRegression(1, lam=lam, ridge=1e-3, min_samples=1, fit_intercept=False)
        for beta in (1.0, -1.0):          # la relation s'inverse
            for _ in range(500):
                x = rng.normal(size=1)
                model.update(x, beta * x[0] + rng.normal(scale=0.1))
        return model.coef[0]

    assert run(0.98) == pytest.approx(-1.0, abs=0.05)
    assert run(1.0) > -0.5  # sans oubli, le modèle reste bloqué sur l'ancien régime


def test_uncertainty_bounded_without_excitation():
    model = OnlineLinearRegression(1, lam=0.9, ridge=1.0, min_samples=1)
    for _ in range(1000):
        model.update(np.array([0.0]), 0.0)  # la feature ne varie jamais
    assert np.trace(model.P) <= model.dim / model.ridge + 1e-9


# ------------------------------------------------------------------ monitoring

def test_monitor_skill_and_coverage():
    rng = np.random.default_rng(2)
    good, bad = ModelMonitor(), ModelMonitor()
    for _ in range(2000):
        signal = rng.normal(scale=1.0)
        realized = signal + rng.normal(scale=0.5)
        good.update(realized, signal, 0.5)
        bad.update(realized, -signal, 0.5)
    assert good.skill > 0.5
    assert bad.skill < 0
    assert good.coverage == pytest.approx(0.68, abs=0.1)


def test_page_hinkley_detects_shift():
    rng = np.random.default_rng(3)
    ph = PageHinkley(delta=0.5, threshold=30)
    assert not any(ph.update(v) for v in rng.chisquare(1, 500))
    assert any(ph.update(v) for v in rng.chisquare(1, 500) * 6)


# ------------------------------------------------------------------ factor model

def test_standardizer():
    st = EWMAStandardizer(1, lam=0.9)
    for v in np.random.default_rng(4).normal(5, 2, 2000):
        z = st.update(np.array([v]))
    assert st.mean[0] == pytest.approx(5, abs=0.5)
    assert abs(z[0]) <= st.clip


def test_factor_model_learns_without_look_ahead(t0):
    """r_{t+h} = 0.001 * signal_t + bruit : le modèle doit l'apprendre à partir
    des seuls résultats réalisés, avec `horizon_bars` barres de décalage."""
    rng = np.random.default_rng(5)
    h = 3
    model = OnlineFactorModel(["f"], timeframe="5m", horizon_bars=h, lam=1.0, ridge=1.0,
                              min_samples=30)
    price, feats = 100.0, []
    n_updates = []
    for i in range(1500):
        feats.append(rng.normal())
        # le rendement de la barre i dépend de la feature observée h barres avant
        drift = 0.001 * feats[i - h] / h if i >= h else 0.0
        price *= math.exp(drift + rng.normal(scale=0.0002))
        model.on_bar("A", price, {"f": feats[i]}, t0 + i * timedelta(minutes=5))
        n_updates.append(model.regression.n_updates)

    assert n_updates[h - 1] == 0 and n_updates[h] == 1  # premier résultat après h barres
    assert model.coefficients()["f"] > 0
    # Une seule des h barres dépend de f_t : skill théorique max ≈ 0.25.
    assert model.monitor.skill > 0.15
    sig = model.signal("A")
    assert sig.horizon == "15m" and sig.std > 0


def test_factor_model_skips_missing_features(t0):
    model = OnlineFactorModel(["f", "g"], timeframe="5m", horizon_bars=1, min_samples=1)
    assert model.on_bar("A", 100, {"f": 1.0}, t0) is None
    model.on_bar("A", 101, {"f": 1.0, "g": 2.0}, t0)
    assert model.regression.n_updates == 0  # l'échantillon incomplet n'est jamais appris


# ------------------------------------------------------------------ HMM

def _two_regime_data(rng, n=600):
    states, X = [], []
    s = 0
    for _ in range(n):
        if rng.random() < 0.02:
            s = 1 - s
        vol = 0.5 if s == 0 else 3.0
        X.append([rng.normal(0, vol), math.log(vol) + rng.normal(0, 0.1)])
        states.append(s)
    return np.array(X), np.array(states)


def test_hmm_recovers_regimes():
    X, states = _two_regime_data(np.random.default_rng(6))
    params, loglik = hmm.fit(X, 2)
    order = np.argsort(params.means[:, -1])
    params = params.permute(order)
    probs = np.array([hmm.filter_probs(X[: t + 1], params) for t in range(len(X))])
    accuracy = np.mean(probs.argmax(axis=1) == states)
    assert accuracy > 0.9
    assert np.allclose(params.transmat.sum(axis=1), 1.0)
    assert np.isfinite(loglik)


def test_filter_step_matches_batch_filter():
    X, _ = _two_regime_data(np.random.default_rng(7), n=200)
    params, _ = hmm.fit(X, 2)
    p = hmm.filter_probs(X[:100], params)
    for x in X[100:]:
        p = hmm.filter_step(p, x, params)
    assert p == pytest.approx(hmm.filter_probs(X, params), abs=1e-8)


def test_regime_model_lifecycle(t0):
    X, _ = _two_regime_data(np.random.default_rng(8), n=400)
    model = RegimeModel("HF", n_states=2, min_samples=100, window=200, refit_every=50)
    states = [model.update(x, t0) for x in X]
    assert all(s is None for s in states[:99])
    assert states[99] is not None
    assert model.fit_count == 1 + (400 - 100) // 50
    # états triés par volatilité : LOW_VOL a la plus petite vol moyenne
    assert model.params.means[0, -1] < model.params.means[1, -1]
    last = states[-1]
    assert set(last.probabilities) == {"LOW_VOL", "HIGH_VOL"}
    assert sum(last.probabilities.values()) == pytest.approx(1.0)
    assert model.switches > 0
