import pytest

from trading_engine.portfolio.portfolio import Portfolio
from trading_engine.portfolio.positions import Position


def test_position_add_reduce_flip():
    p = Position("SPY", 10, 100.0)
    p = p.apply_fill(10, 110.0)
    assert (p.quantity, p.avg_price) == (20, 105.0)

    p = p.apply_fill(-5, 115.0)
    assert (p.quantity, p.avg_price) == (15, 105.0)
    assert p.realized_pnl == pytest.approx(50.0)

    p = p.apply_fill(-20, 100.0)  # retournement : clôture 15, short 5
    assert p.quantity == -5
    assert p.avg_price == 100.0
    assert p.realized_pnl == pytest.approx(50.0 - 75.0)


def test_close_position():
    p = Position("SPY", 10, 100.0).apply_fill(-10, 90.0)
    assert p.quantity == 0.0
    assert p.realized_pnl == pytest.approx(-100.0)


def test_snapshot_weights_and_drift():
    pf = Portfolio(
        cash=1000.0,
        positions={"SPY": Position("SPY", 10, 100.0)},
        target_weights={"SPY": 0.4, "TLT": 0.1},
    )
    pf.update_price("SPY", 100.0)
    state = pf.snapshot()

    assert state.total_value == 2000.0
    spy = state.position("SPY")
    assert spy.weight == pytest.approx(0.5)
    assert spy.drift == pytest.approx(0.1)
    tlt = state.position("TLT")
    assert tlt.quantity == 0 and tlt.drift == pytest.approx(-0.1)


def test_apply_fill_moves_cash():
    pf = Portfolio(cash=1000.0)
    pf.apply_fill("SPY", 5, 100.0)
    pf.update_price("SPY", 110.0)
    state = pf.snapshot()
    assert state.cash == 500.0
    assert state.total_value == pytest.approx(1050.0)
    assert state.unrealized_pnl == pytest.approx(50.0)
