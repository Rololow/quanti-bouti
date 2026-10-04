import asyncio
import dataclasses
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from trading_engine.ai.news_analyzer import (
    AnalysisFailed,
    AnalysisRequest,
    ClaudeNewsAnalyzer,
    SimulatedNewsAnalyzer,
    build_user_prompt,
)
from trading_engine.ai.news_intelligence import AIConfig, NewsIntelligence
from trading_engine.ai.schemas import NEWS_EXTRACTION_SCHEMA, SYSTEM_PROMPT
from trading_engine.ai.validation import validate_analysis
from trading_engine.alerts.alerts import AlertEngine
from trading_engine.config import load_config
from trading_engine.data.events import NewsEvent
from trading_engine.engine import Engine

T0 = datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc)


def raw(**overrides):
    base = {
        "relevant": True, "event_type": "earnings", "canonical_event": "acme beats earnings raises outlook",
        "duplicate_of": None, "sentiment": 0.6, "relevance": 0.9, "importance": 0.8,
        "revenue_impact": 0.3, "earnings_impact": 0.5, "risk_impact": 0.0, "horizon_days": 20,
        "is_rumor": False, "source_type": "major_wire", "key_facts": ["EPS 2.10 vs 1.95 expected"],
        "reported_eps": 2.10, "eps_estimate": 1.95, "guidance_eps": 8.8, "period": "2026Q1",
        "confidence": 0.9,
    }
    base.update(overrides)
    return base


def news(headline="ACME beats earnings, raises outlook", minutes=0, news_id="n1", provider="Reuters", sym="ACME"):
    ts = T0 + timedelta(minutes=minutes)
    return NewsEvent(timestamp=ts, received_at=ts, symbol=sym, source="t", headline=headline,
                     news_id=news_id, provider=provider)


# ------------------------------------------------------------------ schéma et prompt

def test_schema_is_strict_and_prompt_forbids_decisions():
    assert NEWS_EXTRACTION_SCHEMA["additionalProperties"] is False
    assert set(NEWS_EXTRACTION_SCHEMA["required"]) == set(NEWS_EXTRACTION_SCHEMA["properties"])
    assert "Never recommend buying, selling" in SYSTEM_PROMPT
    assert "untrusted data" in SYSTEM_PROMPT
    prompt = build_user_prompt(AnalysisRequest(news(), known_events=[("E1", "acme cuts guidance")]))
    assert "<article>" in prompt and "E1: acme cuts guidance" in prompt


# ------------------------------------------------------------------ validation

def v(r, **kw):
    params = dict(symbol="ACME", news_id="n1", provider="Reuters", known_event_ids={"E1"})
    params.update(kw)
    return validate_analysis(r, **params)


def test_valid_analysis():
    res = v(raw())
    assert res.accepted and res.analysis.source_quality == 0.9
    assert -1 <= res.analysis.impact <= 1


@pytest.mark.parametrize("bad,issue", [
    (raw(sentiment=1.7), "RANGE"),
    (raw(confidence=-0.1), "RANGE"),
    (raw(horizon_days=0), "RANGE"),
    (raw(event_type="buy_signal"), "SCHEMA"),
    ({k: v for k, v in raw().items() if k != "sentiment"}, "SCHEMA"),
    ("not json", "SCHEMA"),
    (raw(sentiment=float("nan")), "SCHEMA"),
])
def test_invalid_outputs_are_rejected_not_corrected(bad, issue):
    res = v(bad)
    assert not res.accepted and res.issues[0].startswith(issue)


def test_consistency_source_and_numbers():
    res = v(raw(relevant=False, importance=0.9, duplicate_of="E99"))
    assert res.analysis.importance == 0.3 and res.analysis.duplicate_of is None
    rumor = v(raw(is_rumor=True, confidence=0.95))
    assert rumor.analysis.confidence == 0.5
    implausible = v(raw(reported_eps=40.0), last_eps=2.0)
    assert implausible.analysis.reported_eps is None
    assert any(i.startswith("NUMBERS") for i in implausible.issues)
    blog = v(raw(source_type="blog_social"), provider="random-blog")
    assert blog.analysis.source_quality == 0.3
    assert not v(raw(confidence=0.3)).accepted


# ------------------------------------------------------------------ client Claude (factice)

class FakeMessages:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        stop, payload = reply
        text = payload if isinstance(payload, str) else json.dumps(payload)
        return SimpleNamespace(stop_reason=stop, model=kwargs["model"],
                               content=[SimpleNamespace(type="text", text=text)],
                               usage=SimpleNamespace(input_tokens=500, output_tokens=120))


def fake_client(*replies):
    messages = FakeMessages(replies)
    return SimpleNamespace(beta=SimpleNamespace(messages=messages)), messages


def test_claude_request_uses_structured_output_effort_and_fallbacks():
    client, messages = fake_client(("end_turn", raw()))
    analyzer = ClaudeNewsAnalyzer(client)
    out = asyncio.run(analyzer.analyze(AnalysisRequest(news()), "low"))
    call = messages.calls[0]
    assert call["model"] == "claude-opus-5"
    assert call["output_config"]["effort"] == "low"
    assert call["output_config"]["format"] == {"type": "json_schema", "schema": NEWS_EXTRACTION_SCHEMA}
    assert call["betas"] == ["server-side-fallback-2026-07-01"] and call["fallbacks"] == "default"
    assert call["system"] == SYSTEM_PROMPT
    assert out.raw["event_type"] == "earnings" and out.usage["output_tokens"] == 120


@pytest.mark.parametrize("reply", [("refusal", raw()), ("max_tokens", "{\"relevant\": tr"), ("end_turn", "not json")])
def test_claude_failures_raise(reply):
    client, _ = fake_client(reply)
    with pytest.raises(AnalysisFailed):
        asyncio.run(ClaudeNewsAnalyzer(client).analyze(AnalysisRequest(news()), "low"))


# ------------------------------------------------------------------ orchestration

def run_intel(intel, *items):
    async def go():
        for item in items:
            intel.submit(item)
        await intel.drain()
        return intel.release(None)
    return asyncio.run(go())


def test_cascade_escalates_only_when_doubtful():
    client, messages = fake_client(("end_turn", raw(confidence=0.55)), ("end_turn", raw(confidence=0.9)))
    now = T0 + timedelta(seconds=3)
    intel = NewsIntelligence(ClaudeNewsAnalyzer(client), clock=lambda: now)
    [ev] = run_intel(intel, news())
    assert [c["output_config"]["effort"] for c in messages.calls] == ["low", "high"]
    assert ev.payload["escalated"] and ev.payload["validated"]["confidence"] == 0.9
    assert ev.received_at == now                          # disponible quand la réponse arrive

    confident, messages2 = fake_client(("end_turn", raw()))
    run_intel(NewsIntelligence(ClaudeNewsAnalyzer(confident), clock=lambda: now), news())
    assert len(messages2.calls) == 1


def test_analyzer_errors_degrade_gracefully():
    client, _ = fake_client(RuntimeError("network down"), ("refusal", raw()))
    intel = NewsIntelligence(ClaudeNewsAnalyzer(client), clock=lambda: T0)
    [ev] = run_intel(intel, news())
    assert ev.payload["validated"] is None and len(ev.payload["issues"]) == 2
    assert intel.stats["failures"] == 1 and intel.on_analysis(ev) == (None, [])


def test_same_article_not_reanalysed_and_budget_enforced():
    client, messages = fake_client(*[("end_turn", raw())] * 5)
    intel = NewsIntelligence(ClaudeNewsAnalyzer(client), AIConfig(max_calls_per_hour=2), clock=lambda: T0)
    run_intel(intel, news(news_id="a"), news(news_id="a"), news(news_id="b"), news(news_id="c"))
    assert intel.stats["skipped_duplicate"] == 1 and intel.stats["skipped_budget"] == 1
    assert len(messages.calls) == 2


def test_one_event_many_articles_confirms_without_amplifying():
    intel = NewsIntelligence(SimulatedNewsAnalyzer())
    evs = run_intel(intel, news("ACME beats expectations and raises outlook", 0, "a", "Reuters"),
                    news("ACME tops estimates, lifts forecast", 5, "b", "Benzinga"),
                    news("ACME faces regulatory probe", 6, "c", "Bloomberg"))
    events = [intel.on_analysis(e)[0] for e in evs]
    assert events[0] is events[1] and events[2] is not events[0]
    first = events[0]
    assert first.confirmation == pytest.approx(1 - 0.1 * 0.3)       # Reuters 0.9 + Benzinga 0.7
    assert len(intel.events("ACME")) == 2                             # 2 événements, pas 3
    score = intel.snapshot("ACME", T0 + timedelta(minutes=10))["ai_news_score"]
    single = first.impact * first.confidence * first.confirmation
    assert score < single + 1e-9                                      # la reprise n'a pas doublé le signal


def test_analysis_and_extracted_facts_respect_availability():
    client, _ = fake_client(("end_turn", raw()))
    later = T0 + timedelta(seconds=30)
    intel = NewsIntelligence(ClaudeNewsAnalyzer(client), clock=lambda: later)
    [ev] = run_intel(intel, news())
    assert intel.release(T0) == []                                    # rien avant la réponse
    _, facts = intel.on_analysis(ev)
    assert {f.name for f in facts} == {"eps", "guidance_eps"}
    assert all(f.available_at == later for f in facts)                # pas à la publication
    assert intel.snapshot("ACME", T0 - timedelta(minutes=1)) == {}


def test_no_fundamentals_from_rumors_or_weak_sources():
    for r, provider in ((raw(is_rumor=True), "Reuters"), (raw(source_type="blog_social"), "someblog")):
        client, _ = fake_client(("end_turn", r))
        intel = NewsIntelligence(ClaudeNewsAnalyzer(client), AIConfig(min_confidence=0.4), clock=lambda: T0)
        [ev] = run_intel(intel, news(provider=provider))
        assert intel.on_analysis(ev)[1] == []


def test_important_news_alert_once_per_event():
    intel = NewsIntelligence(SimulatedNewsAnalyzer())
    for e in run_intel(intel, news("ACME beats expectations and raises outlook", 0, "a"),
                       news("ACME tops estimates, lifts forecast", 5, "b", "Benzinga")):
        intel.on_analysis(e)
    ae = AlertEngine()
    now = T0 + timedelta(minutes=10)
    alerts = ae.evaluate(now, important_news=intel.important_events(now))
    assert [a.kind for a in alerts] == ["IMPORTANT_NEWS"] and "2 source(s)" in alerts[0].message
    assert intel.important_events(now) == []


# ------------------------------------------------------------------ moteur

def test_engine_ai_is_replayed_from_journal_without_model(tmp_path):
    cfg = load_config()
    log = tmp_path / "events.jsonl"
    cfg = dataclasses.replace(cfg, engine=dataclasses.replace(cfg.engine, max_events=15000, report_every=0))
    live = Engine(dataclasses.replace(cfg, storage=dataclasses.replace(cfg.storage, event_log=str(log))))
    asyncio.run(live.run())
    assert live.ai.stats["accepted"] > 0 and live.ai.analyzer is not None

    replay = Engine(dataclasses.replace(cfg, feed=dataclasses.replace(cfg.feed, provider="replay",
                                                                      replay_path=str(log))))
    assert replay.ai.analyzer is None                                  # aucun appel au modèle en replay
    asyncio.run(replay.run())
    for sym in live.features.symbols():
        assert replay.features.snapshot(sym) == live.features.snapshot(sym)
        assert [e.event_id for e in replay.ai.events(sym)] == [e.event_id for e in live.ai.events(sym)]
    assert any(a.kind == "IMPORTANT_NEWS" for a in live.alerts)
    assert list(replay.alerts) == list(live.alerts)


def test_claude_must_be_enabled_explicitly():
    cfg = load_config()
    alpaca = dataclasses.replace(cfg, feed=dataclasses.replace(cfg.feed, provider="replay", replay_path="x"))
    assert Engine.__dict__["_build_analyzer"](SimpleNamespace(config=alpaca)) is None
