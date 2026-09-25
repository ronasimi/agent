from __future__ import annotations

from al_agent.deterministic_router import DeterministicToolRouter
from al_agent.routing_decision import RoutingFeedbackStore
from al_agent.tool_session import ToolSession


def schema(name: str, description: str, required=None):
    required = required or []
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {key: {"type": "string"} for key in required},
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def make_router(tmp_path, **kwargs):
    return DeterministicToolRouter(
        feedback=RoutingFeedbackStore(str(tmp_path / "routing.db")), **kwargs
    )


def test_exact_intent_activates_single_schema_without_model_client(tmp_path):
    router = make_router(tmp_path)
    schemas = [
        schema("current_time", "Return the current clock/date for an IANA timezone."),
        schema("time_difference", "Return the difference between two datetimes.", ["a", "b"]),
        schema("weather_forecast", "Fetch current weather and daily forecast."),
    ]
    decision = router.decide("current_time", schemas)
    assert decision.selected == ("current_time",)
    assert decision.tier == "direct"
    assert decision.confidence >= 0.85


def test_unrelated_conversation_does_not_activate_arbitrary_tools(tmp_path):
    router = make_router(tmp_path)
    schemas = [
        schema("current_time", "Return the current clock/date for an IANA timezone."),
        schema("weather_forecast", "Fetch current weather and daily forecast."),
    ]
    decision = router.decide("Good afternoon, how are you?", schemas)
    assert decision.selected == ()
    assert decision.candidates == ()


def test_multirequirement_prompt_returns_bounded_candidate_set(tmp_path):
    router = make_router(tmp_path, candidate_count=6)
    schemas = [
        schema("current_time", "Return the current clock/date for an IANA timezone."),
        schema("calculate", "Evaluate a small arithmetic expression deterministically."),
        schema("weather_forecast", "Fetch structured current weather conditions and forecast."),
        schema("memory_info", "Return current system memory and swap counters."),
        schema("temperature_sensors", "Return bounded CPU temperature sensor readings."),
        schema("git_status", "Return repository status."),
        schema("gmail_search_messages", "Search email messages."),
    ]
    decision = router.decide(
        "Report the current local time, calculate 137 times 29, retrieve the weather, "
        "then read host memory usage and CPU temperature.",
        schemas,
    )
    assert 2 <= len(decision.selected) <= 6
    assert "current_time" in decision.selected
    assert "weather_forecast" in decision.selected
    assert "memory_info" in decision.selected
    assert "temperature_sensors" in decision.selected
    assert "git_status" not in decision.selected


def test_tool_search_activates_relevant_candidate_set_without_schema_duplication(tmp_path):
    router = make_router(tmp_path)
    schemas = [
        schema("memory_info", "Return current system memory and swap counters."),
        schema("search_memory", "Search durable explicit memories."),
        schema("temperature_sensors", "Return bounded CPU temperature readings."),
    ]
    session = ToolSession(schemas, lambda *a: None, router=router)
    result = session.invoke("tool_search", {"query": "host memory usage", "limit": 6})
    assert "schemas" not in result
    assert all("function" not in row for row in result["candidates"])
    assert "memory_info" in result["activated"]
    assert [s["function"]["name"] for s in session.schemas][:2] == [
        "tool_search", "load_tools"
    ]


def test_feedback_reorders_only_related_candidates(tmp_path):
    router = make_router(tmp_path, auto_activate_threshold=1.0)
    schemas = [
        schema("memory_info", "Return current system memory counters."),
        schema("search_memory", "Search saved memory facts."),
        schema("git_status", "Return repository status."),
    ]
    before = router.decide("memory", schemas)
    key = before.context_key
    for _ in range(8):
        router.record("search_memory", key, 1.0, event_type="task_success")
    after = make_router(tmp_path, auto_activate_threshold=1.0).decide("memory", schemas)
    assert [r.name for r in after.candidates].index("search_memory") <= [r.name for r in before.candidates].index("search_memory")
    assert "git_status" not in [r.name for r in after.candidates]


def test_real_catalog_covers_original_multistep_timeout_prompt(tmp_path):
    from tools.catalog import catalog_snapshot

    schemas, _functions, metadata = catalog_snapshot()
    router = make_router(
        tmp_path,
        candidate_count=8,
        auto_activate_threshold=0.80,
        auto_activate_margin=0.20,
        min_candidate_score=0.18,
    )
    prompt = """Run this tool-routing test one step at a time.
1. Report the current local time in America/Toronto and UTC.
2. Calculate (137 × 29) − 418.
3. Retrieve the current weather for London, Ontario, Canada. Include the observation time and source.
4. Read the host’s memory usage and CPU temperature. State clearly if either is unavailable.
5. Retrieve the local time again using the same timezone.
"""
    decision = router.decide(prompt, schemas, metadata)
    selected = set(decision.selected)
    assert {"current_time", "calculate", "weather_forecast"} <= selected
    assert "temperature_sensors" in selected
    assert {"memory_info", "host_snapshot"} & selected
    assert len(decision.selected) <= 8

    session = ToolSession(
        schemas, lambda *_args: None, metadata,
        max_active=16, max_schema_chars=20000,
        router=router, initial_active=list(decision.selected),
    )
    assert {"current_time", "calculate", "weather_forecast", "temperature_sensors"} <= set(session.active)
