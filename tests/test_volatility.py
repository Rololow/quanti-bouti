import math

import pytest

from trading_engine.features.volatility import EWMAVolatility


def test_ewma_recursion():
    vol = EWMAVolatility(lam=0.9)
    assert vol.update(100.0) is None
    r1 = math.log(101 / 100)
    assert vol.update(101.0) == pytest.approx(abs(r1))
    r2 = math.log(99 / 101)
    expected = math.sqrt(0.9 * r1**2 + 0.1 * r2**2)
    assert vol.update(99.0) == pytest.approx(expected)
    assert vol.n_updates == 2


def test_constant_price_decays_to_zero():
    vol = EWMAVolatility(lam=0.5)
    vol.seed([100, 110] + [110] * 50)
    assert vol.volatility < 1e-6


def test_invalid_lambda():
    with pytest.raises(ValueError):
        EWMAVolatility(lam=1.0)
