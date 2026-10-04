"""Tests de chaos : ce qui tourne mal en live (flux broker dupliqué, perdu,
désordonné, rejets en rafale, compte injoignable, exception interne, état
corrompu). Le moteur doit toujours finir dans un état sûr et cohérent."""

import asyncio
import dataclasses
import random
from datetime import datetime, timezone

import pytest

from test_alpaca_paper import FakeClient, _cfg
from trading_engine.data.events import PortfolioEvent
from trading_engine.engine import Engine
from trading_engine.execution.alpaca_trading import AlpacaPaperBroker, parse_trade_update
from trading_engine.safety.invariants import InvariantMonitor, check_portfolio, check_tax_lots

UTC = timezone.utc


def run_chaos(client, transform=lambda msgs, rng: msgs, *, seed=0, cfg=None):
    """Moteur simulé + faux compte paper ; `transform` altère les trade_updates
    (doublons, pertes, désordre) avant qu'ils n'entrent dans le moteur."""
    rng = random.Random(seed)
    e = Engine(cfg or _cfg(mode="proposals"))
    e.remote_broker = AlpacaPaperBroker(client)
    e.account_synced = False                 # comme en mode alpaca_paper : sync requise
    orig = e._drain_injected

    async def drain():
        if client.sink:
            msgs, client.sink[:] = transform(list(client.sink), rng), []
            now = e.integrity.last_event_time
            for msg in msgs:
                ev = parse_trade_update(msg, clock=lambda: now)
                e._injected.append(dataclasses.replace(ev, timestamp=now))
        await orig()

    e._drain_injected = drain
    asyncio.run(e.run())
    return e


def engine_matches_broker(e, client):
    engine_pos = {s: round(p.quantity, 6) for s, p in e.portfolio.positions.items() if abs(p.quantity) > 1e-9}
    broker_pos = {s: round(q, 6) for s, (q, _) in client.positions_.items() if abs(q) > 1e-9}
    return engine_pos == broker_pos and e.portfolio.cash == pytest.approx(client.cash, abs=0.01)


def assert_safe(e):
    assert e.bus.error_count == 0
    assert e.invariants.violations == [], [str(v) for v in e.invariants.violations]


# ------------------------------------------------------------------ flux trade_updates

def test_duplicated_and_reordered_updates():
    def chaos(msgs, rng):
        msgs = msgs + [rng.choice(msgs) for _ in range(len(msgs))]       # doublons
        rng.shuffle(msgs)                                                 # désordre
        return msgs

    client = FakeClient(cash=50_000.0, positions={"SPY": [20, 600.0]}, sink=[])
    e = run_chaos(client, chaos)
    assert client.submitted and e.ignored_order_updates > 0
    assert engine_matches_broker(e, client)
    assert not [a for a in e.alerts if a.kind == "RECONCILIATION"]
    assert_safe(e)


def test_lost_partial_fills_are_caught_up_by_the_cumulative_quantity():
    client = FakeClient(cash=50_000.0, sink=[])
    e = run_chaos(client, lambda msgs, rng: [m for m in msgs if m["data"]["event"] != "partial_fill"])
    assert client.submitted and e.fills
    assert engine_matches_broker(e, client)
    assert_safe(e)


def test_stream_outage_is_repaired_by_reconciliation():
    """Aucune mise à jour pour la moitié des ordres (flux coupé) : les ordres
    sont oubliés après expiration et le rapprochement adopte l'état du compte."""
    def chaos(msgs, rng):
        lost = {m["data"]["order"]["client_order_id"] for m in msgs if rng.random() < 0.5}
        return [m for m in msgs if m["data"]["order"]["client_order_id"] not in lost]

    client = FakeClient(cash=50_000.0, sink=[])
    e = run_chaos(client, chaos, seed=3)
    assert e.remote_broker.forgotten > 0
    assert [a for a in e.alerts if a.kind == "RECONCILIATION"]
    assert engine_matches_broker(e, client)
    assert_safe(e)


# ------------------------------------------------------------------ broker et compte

def test_broker_rejects_everything():
    client = FakeClient(cash=50_000.0, sink=[], reject={"SPY", "QQQ", "TLT", "GLD"})
    e = run_chaos(client)
    assert e.remote_errors >= 3 and not client.submitted
    assert e.safety.state.value == "HALTED"
    assert any("HARD_CONTROL" in r or "BROKER" in r for r in e.safety.status.reasons)
    assert e.remote_broker.working == {}
    assert e.bus.error_count == 0


def test_no_order_while_the_account_was_never_read():
    """Compte injoignable dès le démarrage : l'état réel est inconnu, le moteur
    n'envoie aucun ordre (il calculerait sur les positions de la config)."""
    class Down(FakeClient):
        def account(self):
            raise OSError("connection reset")

    client = Down(cash=50_000.0, sink=[])
    cfg = _cfg(mode="proposals")
    eager = dataclasses.replace(cfg, decision=dataclasses.replace(cfg.decision, risk_aversion=5_000.0))
    e = run_chaos(client, cfg=eager)
    assert not client.submitted and not e.account_synced and e.bus.error_count == 0
    assert [a for a in e.alerts if a.kind == "BROKER_UNSYNCED" and a.severity == "critical"]
    assert any(p.orders for p in e.plans)                     # il aurait voulu trader


def test_trading_starts_once_the_account_answers():
    class Late(FakeClient):
        calls = 0

        def account(self):
            Late.calls += 1
            if Late.calls <= 3:
                raise OSError("timeout")
            return super().account()

    client = Late(cash=50_000.0, positions={"SPY": [20, 600.0]}, sink=[])
    e = run_chaos(client)
    assert e.account_synced and client.submitted
    assert engine_matches_broker(e, client)
    assert_safe(e)


def test_account_blocked_mid_session_halts_and_cancels():
    class Blocking(FakeClient):
        calls = 0

        def account(self):
            Blocking.calls += 1
            acc = super().account()
            return dataclasses.replace(acc, trading_blocked=Blocking.calls > 3)

    client = Blocking(cash=50_000.0, sink=[])
    e = run_chaos(client)
    assert e.safety.state.value == "HALTED"
    assert any("BROKER_ACCOUNT" in r for r in e.safety.status.reasons)
    assert e.remote_broker.working == {}


def test_corrupt_broker_position_halts():
    e = Engine(_cfg(mode="proposals"))
    t = datetime(2026, 1, 6, 15, tzinfo=UTC)
    ev = PortfolioEvent(timestamp=t, received_at=t, symbol=None, source="alpaca_trading",
                        payload={"kind": "sync", "cash": 1000.0, "equity": 1000.0, "status": "ACTIVE",
                                 "trading_blocked": False, "positions": {"SPY": [5, float("nan"), float("nan")]}})
    asyncio.run(e._ingest(ev))
    assert e.safety.state.value == "HALTED" and "INVARIANT" in e.safety.status.reasons[0]
    assert [a for a in e.alerts if a.kind == "INVARIANT" and a.severity == "critical"]


# ------------------------------------------------------------------ moniteur d'invariants

def test_handler_exception_halts_the_engine():
    e = Engine(_cfg(mode="paper"))
    calls = {"n": 0}
    original = e._on_bar

    def flaky(bar):
        calls["n"] += 1
        if calls["n"] == 500:
            raise RuntimeError("boom")
        return original(bar)

    e.bus._handlers["bar"][e.bus._handlers["bar"].index(original)] = flaky
    asyncio.run(e.run())
    assert e.bus.error_count == 1
    assert e.safety.state.value == "HALTED" and "HANDLER" in e.safety.status.reasons[0]
    assert e.broker.working == {}                             # ordres annulés
    assert len([a for a in e.alerts if a.kind == "INVARIANT"]) == 1


@pytest.mark.parametrize("corrupt, name", [
    (lambda e: setattr(e.portfolio, "cash", -0.5 * e.portfolio.total_value()), "CASH"),
    (lambda e: e.portfolio.apply_fill("SPY", -1000, 600.0), "SHORT"),
    (lambda e: e.portfolio.set_target_weights({"SPY": 0.9}), "TARGET"),
    (lambda e: e.tax.gains.lots.clear(), "TAX_LOTS"),
    (lambda e: e.portfolio.update_price("SPY", float("inf"), datetime(2026, 1, 6, tzinfo=UTC)), "FINITE"),
])
def test_each_invariant_detects_its_corruption(corrupt, name):
    e = Engine(_cfg(mode="paper"))
    asyncio.run(e._bootstrap(datetime(2026, 1, 5, 14, 30, tzinfo=UTC)))
    assert e.invariants.check(e) == []
    corrupt(e)
    asyncio.run(e._check_invariants(datetime(2026, 1, 6, 15, tzinfo=UTC)))
    assert name in {v.name for v in e.invariants.violations}
    assert e.safety.state.value == "HALTED"


def test_alert_mode_does_not_halt():
    cfg = _cfg(mode="paper")
    e = Engine(dataclasses.replace(cfg, safety=dataclasses.replace(cfg.safety, invariants="alert")))
    e.portfolio.cash = -1e9
    asyncio.run(e._check_invariants(datetime(2026, 1, 6, tzinfo=UTC)))
    asyncio.run(e._check_invariants(datetime(2026, 1, 6, 1, tzinfo=UTC)))
    assert e.safety.state.value != "HALTED"
    assert len([a for a in e.alerts if a.kind == "INVARIANT"]) == 1       # pas de répétition


def test_invariant_helpers():
    assert check_portfolio(100.0, {"A": (1.0, 50.0)}, max_gross=1.0, long_only=True) == []
    assert [v.name for v in check_portfolio(-60.0, {"A": (3.0, 50.0)}, max_gross=1.0, long_only=True)] == \
        ["CASH", "GROSS"]
    assert [v.name for v in check_portfolio(0.0, {}, max_gross=1.0, long_only=True)] == ["FINITE"]
    assert check_tax_lots({"A": 2.0, "B": -1.0}, {"A": 2.0}) == []
    assert check_tax_lots({"A": 2.0}, {"A": 1.0})[0].name == "TAX_LOTS"
    with pytest.raises(ValueError):
        InvariantMonitor("sometimes")
