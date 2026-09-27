import asyncio
import dataclasses
import json
import math
from datetime import datetime, timedelta, timezone
from enum import Enum

import numpy as np
import pytest

from trading_engine.api.server import DashboardServer, dashboard_html, export_static
from trading_engine.api.state import build_state, dumps_state, to_jsonable
from trading_engine.config import load_config
from trading_engine.data.events import NewsEvent
from trading_engine.engine import Engine

T0 = datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc)


@dataclasses.dataclass
class _D:
    x: float
    when: datetime


class _E(str, Enum):
    A = "A"


def test_to_jsonable():
    out = to_jsonable({"nan": math.nan, "inf": np.float64(np.inf), "arr": np.array([1.0, 2.0]),
                       "d": _D(1.5, T0), "td": timedelta(minutes=5), "set": {"b", "a"}, "enum": _E.A})
    assert out == {"nan": None, "inf": None, "arr": [1.0, 2.0], "d": {"x": 1.5, "when": T0.isoformat()},
                   "td": 300.0, "set": ["a", "b"], "enum": "A"}
    json.dumps(out, allow_nan=False)


@pytest.fixture(scope="module")
def engine():
    cfg = load_config()
    cfg = dataclasses.replace(
        cfg,
        engine=dataclasses.replace(cfg.engine, max_events=12000, report_every=0),
        allocation=dataclasses.replace(cfg.allocation, method="hrp"),
        risk=dataclasses.replace(cfg.risk, min_observations=5),
        decision=dataclasses.replace(cfg.decision, risk_aversion=50.0, holding_period="20d"),
        tax=dataclasses.replace(cfg.tax, profile=None),
    )
    e = Engine(cfg)
    asyncio.run(e.run())
    return e


def test_state_is_complete_and_strict_json(engine):
    state = build_state(engine)
    for key in ("meta", "safety", "portfolio", "positions", "signals", "regimes", "risk", "allocation",
                "decision", "decisions", "execution", "news", "alerts", "history", "qualitative"):
        assert key in state
    json.dumps(state, allow_nan=False)                       # aucun NaN / inf
    assert state["history"] and state["decisions"]
    assert {p["symbol"] for p in state["positions"]} == set(engine.config.feed.symbols)
    assert state["execution"]["fills"]                        # paper trading actif
    assert len(dumps_state(engine)) > 1000


async def _http(port, method="GET", path="/"):
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(f"{method} {path} HTTP/1.1\r\nHost: localhost\r\n\r\n".encode())
    await writer.drain()
    raw = await reader.read()
    writer.close()
    head, _, body = raw.partition(b"\r\n\r\n")
    return int(head.split()[1]), head.decode(), body


def test_server_routes_are_read_only(engine):
    async def scenario():
        server = DashboardServer(engine, port=0)
        await server.start()
        try:
            assert server.host == "127.0.0.1"
            status, head, body = await _http(server.port, path="/")
            assert status == 200 and "text/html" in head and b"Control Tower" in body
            status, head, body = await _http(server.port, path="/api/state")
            assert status == 200 and json.loads(body)["safety"]["state"] in ("NORMAL", "DEGRADED", "HALTED")
            assert (await _http(server.port, path="/api/history"))[0] == 200
            assert (await _http(server.port, path="/api/decisions"))[0] == 200
            assert (await _http(server.port, path="/healthz"))[2] == b"ok"
            assert (await _http(server.port, path="/nope"))[0] == 404
            assert (await _http(server.port, method="POST", path="/api/state"))[0] == 405
        finally:
            await server.stop()
    asyncio.run(scenario())


def test_dashboard_never_injects_html():
    html = dashboard_html()
    assert "innerHTML" not in html and "insertAdjacentHTML" not in html
    assert "textContent" in html or "createTextNode" in html


def test_static_export_escapes_hostile_headlines(engine, tmp_path):
    hostile = '</script><img src=x onerror="alert(1)">'
    engine.news.on_news(NewsEvent(timestamp=engine.features.now, received_at=engine.features.now,
                                  symbol="SPY", source="t", headline=hostile, news_id="evil"))
    path = tmp_path / "dash.html"
    export_static(engine, path)
    html = path.read_text(encoding="utf-8")
    assert "</script><img" not in html                       # la balise ne peut pas se fermer
    embedded = html.split('<script id="embedded-state" type="application/json">')[1].split("</script>")[0]
    state = json.loads(embedded)
    assert any(n["headline"] == hostile for n in state["news"])   # le texte reste intact (affiché via textContent)
