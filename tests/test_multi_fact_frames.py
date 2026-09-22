import json

from tools.grounding import FactGroundingLedger, make_observation, requested_fact_types, validate_fact_grounding
from tools.task_requirements import (
    TaskRequirementLedger,
    build_news_query,
    derive_fact_frames,
    derive_task_frame,
    select_primary_fact_frame,
)


def test_headlines_and_weather_have_independent_scopes_without_query_bleed():
    request = "What are the current headlines and weather?"
    frames = derive_fact_frames(request, default_location="London, Ontario, Canada")
    assert set(frames) == {"news", "weather"}
    assert frames["news"]["source_text"].lower().endswith("current headlines")
    assert frames["news"].get("entity", "") == ""
    assert frames["weather"]["entity"] == "London, Ontario, Canada"
    assert build_news_query(request, frames["news"], "London, Ontario, Canada") == "latest news"

    primary = select_primary_fact_frame(
        frames,
        derive_task_frame(request, default_location="London, Ontario, Canada"),
    )
    # Compatibility remains single-valued for legacy code, but no longer controls
    # the scope of the secondary news requirement.
    assert primary["intent"] == "weather"


def test_shared_prefix_time_modifier_is_inherited_by_both_fact_frames():
    frames = derive_fact_frames("Give me today's headlines and weather in London")
    assert frames["news"]["time_scope"] == "today's"
    assert frames["weather"]["time_scope"] == "today's"
    assert frames["news"].get("entity", "") == ""
    assert frames["weather"]["entity"] == "London"


def test_local_time_modifiers_override_shared_inheritance_independently():
    frames = derive_fact_frames("headlines today and weather tomorrow", default_location="London, Ontario, Canada")
    assert frames["news"]["time_scope"] == "today"
    assert frames["weather"]["time_scope"] == "tomorrow"


def test_topic_and_location_modifiers_do_not_cross_fact_boundaries():
    frames = derive_fact_frames("headlines about AI and weather in London")
    assert frames["news"].get("entity", "") == ""
    assert build_news_query("headlines about AI and weather in London", frames["news"]) == "ai latest news"
    assert frames["weather"]["entity"] == "London"


def test_as_well_as_compound_request_and_single_domain_entity_conjunction():
    compound = derive_fact_frames(
        "weather as well as local news",
        default_location="London, Ontario, Canada",
    )
    assert set(compound) == {"weather", "news"}
    assert compound["weather"]["entity"] == "London, Ontario, Canada"
    assert compound["news"]["entity"] == "London, Ontario, Canada"

    single = derive_fact_frames("weather in London and Windsor")
    assert set(single) == {"weather"}
    assert single["weather"]["source_text"] == "weather in London and Windsor"


def test_fact_frame_completeness_invariant_keeps_sparse_secondary_requirements():
    frames = derive_fact_frames(
        "weather now",
        required_fact_types={"weather", "web_fact"},
        default_location="London, Ontario, Canada",
    )
    assert set(frames) == {"weather", "web_fact"}
    assert frames["web_fact"]["intent"] == "web_fact"


def test_grounding_ledger_preserves_success_when_another_fact_is_missing():
    ledger = FactGroundingLedger.from_fact_types({"weather", "news"})
    ledger.apply_report({
        "required_fact_types": ["weather", "news"],
        "missing_fact_types": ["news"],
        "evidence": {"weather": ["weather_forecast"]},
        "reason": "requested fact type is not present in qualifying observations",
    })
    assert ledger.requirements["weather"].satisfied is True
    assert ledger.requirements["news"].satisfied is False

    # A later news failure must not reopen already-grounded weather.
    ledger.mark_error("news", "provider timeout")
    assert ledger.requirements["weather"].satisfied is True
    assert ledger.missing_fact_types() == {"news"}


def test_compound_grounding_clears_each_fact_independently():
    request = "What are the current headlines and weather?"
    frames = derive_fact_frames(request, default_location="London, Ontario, Canada")
    assert requested_fact_types(request, fact_frames=frames) == {"news", "weather"}

    weather = make_observation(
        "weather_api",
        '{"temperature_2m":9.6,"wind_speed_10m":19.2,"precipitation_probability":0}',
        turn_id=7,
        fact_frames=frames,
    )
    report = validate_fact_grounding(request, [weather], current_turn_id=7, fact_frames=frames)
    assert report["grounded"] is False
    assert report["missing_fact_types"] == ["news"]
    statuses = {row["fact_type"]: row["status"] for row in report["fact_requirements"]}
    assert statuses == {"news": "pending", "weather": "satisfied"}

    news = make_observation(
        "news_search",
        json.dumps([{
            "title": "Headline",
            "url": "https://example.com/story",
            "source": "Example",
            "date": "2026-09-22",
        }]),
        arguments={"query": "latest news", "location": "", "timelimit": "d"},
        turn_id=7,
        fact_frames=frames,
    )
    report = validate_fact_grounding(request, [weather, news], current_turn_id=7, fact_frames=frames)
    assert report["grounded"] is True
    assert report["missing_fact_types"] == []


def test_compound_requirements_include_both_weather_and_news():
    ledger = TaskRequirementLedger.from_request("What are the current headlines and weather?")
    assert {"weather_forecast", "news_search"} <= set(ledger.required_tools())


def test_fact_tool_pruning_respects_all_active_fact_frames():
    from al_agent.turn_support import _prune_mismatched_fact_tools

    schemas = [
        {"type": "function", "function": {"name": "weather_forecast", "parameters": {}}},
        {"type": "function", "function": {"name": "news_search", "parameters": {}}},
        {"type": "function", "function": {"name": "current_time", "parameters": {}}},
    ]
    frames = derive_fact_frames(
        "What are the current headlines and weather?",
        default_location="London, Ontario, Canada",
    )
    primary = select_primary_fact_frame(frames, {"intent": "weather"})
    changed = _prune_mismatched_fact_tools(
        schemas,
        primary,
        "What are the current headlines and weather?",
        fact_frames=frames,
    )
    names = {schema["function"]["name"] for schema in schemas}
    assert changed is True
    assert names == {"weather_forecast", "news_search"}


def test_compound_turn_pregrounds_weather_and_news_before_model(monkeypatch):
    from al_agent import turn_engine as te

    calls = []

    def fake_weather_recovery(request, memory_context="", *, frame=None):
        calls.append(("weather_recovery", request, dict(frame or {})))
        return {
            "ok": True,
            "stages": [
                {"id": "place", "tool": "geocode_location", "ok": True, "args": {"query": "London, Ontario, Canada"}},
                {"id": "forecast", "tool": "weather_forecast", "ok": True, "args": {"latitude": 42.99, "longitude": -81.24, "forecast_days": 1}},
                {"id": "result", "tool": "compose_object", "ok": True},
            ],
            "result": {
                "location": "London, Ontario, Canada",
                "forecast": {
                    "provider": "Open-Meteo",
                    "current": {"temperature_2m": 9.6, "wind_speed_10m": 19.2},
                    "daily": {"temperature_2m_max": [16.7], "temperature_2m_min": [7.1]},
                },
            },
            "grounding_recovery": {
                "fact_type": "weather",
                "query": "weather London Ontario current",
                "location": "London, Ontario, Canada",
            },
        }

    def fake_execute(name, args):
        calls.append((name, dict(args)))
        if name == "news_search":
            return json.dumps([{
                "title": "Headline",
                "url": "https://example.com/story",
                "source": "Example",
                "date": "2026-09-22",
            }])
        raise AssertionError((name, args))

    class Model:
        def chat(self, **kwargs):
            calls.append(("model", [schema.get("function", {}).get("name") for schema in kwargs.get("tools", [])]))
            yield {"message": {"content": "Weather and headline summary."}, "done": True}

    monkeypatch.setattr(te, "execute_weather_grounding_recovery", fake_weather_recovery)
    monkeypatch.setattr(te, "_execute_registered_tool", fake_execute)
    monkeypatch.setattr(te, "WORKING_STATE_ENABLED", False)
    monkeypatch.setattr(te, "RECIPES_ENABLED", False)
    monkeypatch.setattr(te, "LOOP_VALIDATOR_ENABLED", False)
    monkeypatch.setattr(te, "get_conversation_summary", lambda: "")
    monkeypatch.setattr(te, "build_memory_context", lambda *_: "")
    monkeypatch.setattr(te, "get_relevant_user_prompt_context", lambda *_: "")
    monkeypatch.setattr(te, "get_user_location", lambda: "London, Ontario, Canada")
    monkeypatch.setattr(te, "_bounded_tool_result_with_ref", lambda _name, text: (text, ""))
    monkeypatch.setattr(te, "evict_report_model_for_interactive", lambda: None)
    monkeypatch.setattr(te, "_prune_compacted_history", lambda _messages: None)
    monkeypatch.setattr(te, "log_perf_stats", lambda *a, **k: None)

    messages = [{"role": "system", "content": "system"}]
    te.handle_user_turn(
        messages,
        "What are the current headlines and weather?",
        False,
        runtime_overrides={
            "OLLAMA": Model(),
            "record_monitor_state": lambda *a, **k: None,
            "append_and_save": lambda rows, item: rows.append(item),
            "acquire_inference_lock": lambda: None,
            "release_inference_lock": lambda _lock: None,
            "queue_compaction_if_needed": lambda *a, **k: None,
        },
    )

    weather_call = next(item for item in calls if item[0] == "weather_recovery")
    assert weather_call[1].lower().startswith("weather")
    assert weather_call[2]["entity"] == "London, Ontario, Canada"

    news_call = next(item for item in calls if item[0] == "news_search")
    assert news_call[1]["query"] == "latest news"
    assert news_call[1]["location"] == ""

    model_call = next(item for item in calls if item[0] == "model")
    assert "weather_forecast" not in model_call[1]
    assert "news_search" not in model_call[1]
    assert messages[-1]["role"] == "assistant"
