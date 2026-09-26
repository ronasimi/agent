"""Requested scenarios A–E through the production engine with scripted Qwen XML.

Provider payloads are fixtures, not live Gmail/weather/host measurements. The
catalog, routing, argument validation, persistence, grounding and compaction are
real. Run with pytest -q tests/test_audit_scenarios.py -rA.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from test_autonomous_loop import FakeClient
from test_runtime_audit import isolated_db  # noqa: F401 - shared pytest fixture


def call(name, **arguments):
    params = "".join(f"<parameter={key}>{json.dumps(value) if not isinstance(value, str) else value}</parameter>" for key, value in arguments.items())
    return f"<tool_call><function={name}>{params}</function></tool_call>"


@pytest.fixture
def harness(monkeypatch, isolated_db):
    from al_agent import turn_engine as te, state
    from tools import user_profile
    from tools.conversation_context import conversation_context
    from al_agent.events import frontend_event_context

    user_profile.complete_onboarding_profile(name="Audit User", timezone="UTC", location="London, Ontario")
    monkeypatch.setattr(state, "MODEL_TRACE_ENABLED", False)
    monkeypatch.setitem(state.AGENT_CFG, "tool_protocol", "qwen_xml")
    now = datetime.now(timezone.utc).isoformat()
    results = {
        "gmail_inbox_counts": {"ok": True, "messages_total": 27, "messages_unread": 3},
        "temperature_sensors": {"k10temp": [{"label": "Tctl", "current": 54.5, "critical": 100.0}]},
        "current_time": {"utc": now, "local": now, "timezone": "UTC"},
        "calculate": {"expression": "6*7", "result": 42},
        "host_snapshot": {"cpu": {"percent": 12.0}, "memory": {"percent": 40}, "hostname": "fixture-host"},
        "geocode_location": [{"name": "London", "country": "Canada", "admin1": "Ontario", "latitude": 42.98, "longitude": -81.25}],
        "weather_forecast": {"provider": "Open-Meteo", "latitude": 42.98, "longitude": -81.25, "retrieved_at": now,
                             "current": {"time": now, "temperature_2m": 19.0, "weather_code": 0}},
        "browse_url": "URL: https://example.com/docs\nOfficial protocol documentation: the example protocol uses framed messages.",
    }
    executed = []

    def execute(name, args, **kwargs):
        executed.append(name)
        if name == "get_user_profile":
            return user_profile.get_user_profile(**args)
        if name in {"read_observation", "search_observations"}:
            return getattr(isolated_db, name)(**args)
        assert name in results, f"Unexpected tool: {name}"
        return results[name]

    monkeypatch.setattr(te, "execute_registered_tool", execute)

    def turn(query, replies=()):
        events = []
        client = FakeClient(replies)
        with conversation_context("scenarios"), frontend_event_context(events.append):
            te.handle_user_turn([], query, False, refresh_history=True, runtime_overrides={"OLLAMA": client})
        assert events[-1]["type"] == "turn_end"
        assert events[-1]["success"], client.requests[-1]["messages"][-1]["content"] if client.requests else events
        return events, client

    return turn, executed, isolated_db


def test_scenario_a_profile(harness, record_property):
    turn, executed, memory = harness
    for query, expected in [("What is my name?", "Audit User"), ("Read my profile.", "Audit User"), ("What city do I live in?", "London")]:
        events, client = turn(query)
        assert not client.requests
        assert expected in next(e["content"] for e in events if e["type"] == "assistant_final")
    with memory._connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM tool_observations WHERE conversation_id='scenarios' AND tool_name='get_user_profile'").fetchone()[0] == 3
    record_property("result", "3 profile queries; 0 model calls; 3 verified profile observations")


def gmail_turn(turn, query):
    return turn(query, [call("load_tools", names=["gmail_inbox_counts"]), call("gmail_inbox_counts"), "There are 27 inbox messages and 3 unread."])


def test_scenario_b_gmail_followup_chain(harness, record_property):
    turn, executed, _ = harness
    queries = ["How many emails are in my inbox?", "Double check your Gmail access.", "It is included in the harness.", "How many are unread?"]
    for query in queries:
        events, client = gmail_turn(turn, query)
        route = next(e for e in events if e["type"] == "routing_complete")
        assert any(name.startswith("gmail_") for name in route["selected"]), route
        if query.startswith(("It is", "How many are")):
            assert route["contextual"]
    assert executed == ["gmail_inbox_counts"] * 4
    record_property("result", "4 Gmail turns; each has actual Gmail fixture evidence; inherited topic survives the chain")


def test_scenario_c_historical_exact_rehydration(harness, record_property):
    turn, executed, memory = harness
    turn("What is the current CPU temperature?", [call("load_tools", names=["temperature_sensors"]), call("temperature_sensors"), "CPU Tctl was 54.5 C."])
    with memory._connect() as conn:
        ref = conn.execute("SELECT id FROM tool_observations WHERE conversation_id='scenarios' AND tool_name='temperature_sensors'").fetchone()[0]
    for i in range(7):
        turn(f"Calculate 6*7, request {i}", [call("load_tools", names=["calculate"]), call("calculate", expression="6*7"), "42"])
    query = "What did we determine about my CPU temperature earlier?"
    events, client = turn(query, [call("load_tools", names=["search_observations", "read_observation"]),
                                  call("search_observations", query="temperature_sensors"), call("read_observation", observation_id=ref), "The earlier Tctl reading was 54.5 C."])
    turn("What was the exact critical threshold earlier?", [call("load_tools", names=["read_observation"]), call("read_observation", observation_id=ref), "Its critical threshold was 100.0 C."])
    assert executed.count("temperature_sensors") == 1
    assert executed.count("read_observation") == 2
    assert "critical" in json.dumps(client.requests[-1])
    record_property("result", "CPU evidence recovered after 7 unrelated tool turns; 2 exact rehydrations; no fresh sensor read")


def test_scenario_d_explicit_topic_switch(harness, record_property):
    turn, _, _ = harness
    gmail_turn(turn, "How many emails are in my inbox?")
    events, _ = turn("What is the current CPU temperature?", [call("load_tools", names=["temperature_sensors"]), call("temperature_sensors"), "CPU Tctl is 54.5 C."])
    route = next(e for e in events if e["type"] == "routing_complete")
    assert not route["contextual"] and not any(name.startswith("gmail_") for name in route["selected"])
    record_property("result", "CPU route selected without Gmail affinity")


def test_scenario_e_multidomain_prompt_growth(harness, record_property):
    turn, executed, _ = harness
    names = ["current_time", "geocode_location", "weather_forecast", "calculate", "host_snapshot", "browse_url", "get_user_profile"]
    query = ("1. Check the current time in UTC.\n2. Check the weather in London Ontario now.\n"
             "3. Calculate 6*7.\n4. Check host snapshot.\n5. Verify the official protocol documentation at https://example.com/docs.\n6. Read my profile.")
    replies = [call("load_tools", names=names), call("current_time", timezone_name="UTC"),
               call("geocode_location", query="London Ontario"), call("weather_forecast", latitude=42.98, longitude=-81.25),
               call("calculate", expression="6*7"), call("host_snapshot"), call("browse_url", url="https://example.com/docs"),
               call("get_user_profile"), "Time, weather, 42, host state, documentation and profile retrieved."]
    events, client = turn(query, replies)
    sizes = [e["prompt_telemetry"]["estimated_input_tokens"] for e in events if e["type"] == "model_start"]
    assert len(sizes) == 9 and max(sizes) <= 13824
    assert set(names) <= set(executed)
    assert "54.5" not in json.dumps(client.requests[-1])  # no unrelated host measurement inherited
    record_property("prompt_estimated_tokens_by_call", json.dumps(sizes))
