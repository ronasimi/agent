from __future__ import annotations

import json


def test_dense_context_estimation_has_byte_floor_and_hard_budget():
    from tools.context import estimate_messages_tokens, estimate_tokens, build_active_messages

    dense = "x" * 100_000
    assert estimate_tokens(dense) >= 25_000
    messages = build_active_messages(
        system_prompt="system",
        summary="",
        history=[{"role": "user", "content": dense}],
        max_ctx_tokens=2048,
        reserve_tokens=256,
    )
    assert estimate_messages_tokens(messages) <= 2048 - 256


def test_tool_transaction_is_not_left_orphaned_by_context_fitting():
    from tools.context import fit_tool_loop_messages

    prefix = [{"role": "system", "content": "s"}]
    tail = [
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "x", "arguments": {}}}]},
        {"role": "tool", "name": "x", "tool_call_id": "call-1", "content": "y" * 50_000},
        {"role": "assistant", "content": "done"},
    ]
    fitted = fit_tool_loop_messages(prefix, tail, max_ctx_tokens=512, reserve_tokens=128)
    seen_calls = {
        str(call.get("id"))
        for msg in fitted if msg.get("role") == "assistant"
        for call in (msg.get("tool_calls") or []) if isinstance(call, dict)
    }
    for msg in fitted:
        if msg.get("role") == "tool" and msg.get("tool_call_id"):
            assert str(msg["tool_call_id"]) in seen_calls


def test_task_frame_resolves_weather_followups_and_new_task_boundary():
    from tools.task_requirements import derive_task_frame, is_followup_request

    first = derive_task_frame("What's the weather in London, Ontario today?")
    tomorrow = derive_task_frame("What about tomorrow?", first)
    toronto = derive_task_frame("And Toronto?", tomorrow)
    assert first["intent"] == tomorrow["intent"] == toronto["intent"] == "weather"
    assert "london" in tomorrow["entity"].lower()
    assert tomorrow["time_scope"] == "tomorrow"
    assert "toronto" in toronto["entity"].lower()
    assert not is_followup_request("Now write a Python script")


def test_non_host_temperature_and_non_network_neighbors_do_not_trigger_requirements():
    from tools.task_requirements import derive_requirements

    assert not derive_requirements("What temperature should I bake bread at?")
    assert not derive_requirements("What is the temperature of the Sun?")
    assert not derive_requirements("Tell me about my neighbors")


def test_requirement_scope_rejects_success_for_wrong_dns_target():
    from tools.task_requirements import TaskRequirementLedger

    ledger = TaskRequirementLedger.from_request("Resolve example.com with DNS")
    assert "dns_diagnose" in ledger.required_tools()
    ledger.record_tool(
        "dns_diagnose", status="ok", arguments={"host": "example.org"},
        result_text='{"host":"example.org","addresses":["203.0.113.1"]}',
    )
    assert ledger.status_for_tool("dns_diagnose") == "failed"
    ledger.record_tool(
        "dns_diagnose", status="ok", arguments={"host": "example.com"},
        result_text='{"host":"example.com","addresses":["203.0.113.2"]}',
    )
    assert ledger.status_for_tool("dns_diagnose") == "satisfied"


def test_weather_grounding_requires_linked_discovery_and_correct_scope():
    from tools.grounding import make_observation, validate_fact_grounding
    from tools.task_requirements import derive_task_frame

    request = "What's the weather in London, Ontario tomorrow?"
    frame = derive_task_frame(request)
    search = make_observation(
        "web_search",
        json.dumps([{"title": "London weather forecast", "url": "https://weather.example/london", "snippet": "Tomorrow forecast 18 C rain"}]),
        arguments={"query": "London Ontario weather tomorrow"}, turn_id=7,
    )
    wrong_browse = make_observation(
        "browse_url", "URL: https://weather.example/toronto\nToronto tomorrow weather forecast 20 C",
        arguments={"url": "https://weather.example/toronto"}, turn_id=7,
    )
    report = validate_fact_grounding(request, [search, wrong_browse], current_turn_id=7, task_frame=frame)
    assert report["grounded"] is False
    right_browse = make_observation(
        "browse_url", "URL: https://weather.example/london\nLondon Ontario tomorrow weather forecast 18 C with rain",
        arguments={"url": "https://weather.example/london"}, turn_id=7,
    )
    report = validate_fact_grounding(request, [search, right_browse], current_turn_id=7, task_frame=frame)
    assert report["grounded"] is True


def test_lazy_catalog_does_not_import_heavy_report_or_browser_modules():
    import subprocess, sys

    code = "import sys, tools.catalog; print(int('weasyprint' in sys.modules), int('playwright' in sys.modules))"
    result = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    assert result.stdout.strip() == "0 0"


def test_generic_web_grounding_requires_linked_search_and_browse_for_modern_observations():
    from tools.grounding import make_observation, validate_fact_grounding

    request = "Look up current Acme protocol documentation on the web"
    search = make_observation(
        "web_search",
        json.dumps([{"title": "Acme protocol docs", "url": "https://docs.example/acme"}]),
        arguments={"query": "Acme protocol documentation"}, turn_id=9,
    )
    wrong = make_observation(
        "browse_url", "URL: https://docs.example/other\nOther protocol documentation",
        arguments={"url": "https://docs.example/other"}, turn_id=9,
    )
    assert validate_fact_grounding(request, [search, wrong], current_turn_id=9)["grounded"] is False
    right = make_observation(
        "browse_url", "URL: https://docs.example/acme\nAcme protocol documentation",
        arguments={"url": "https://docs.example/acme"}, turn_id=9,
    )
    assert validate_fact_grounding(request, [search, right], current_turn_id=9)["grounded"] is True


def test_conversation_scoping_isolates_history_observations_and_working_state(tmp_path, monkeypatch):
    from tools import memory, working_state
    from tools.conversation_context import conversation_context

    db = str(tmp_path / "scoped.db")
    monkeypatch.setattr(memory, "DB_PATH", db)
    monkeypatch.setattr(working_state, "DB_PATH", db)
    memory.init_db()
    a = memory.create_conversation("A")["id"]
    b = memory.create_conversation("B")["id"]

    with conversation_context(a):
        memory._save_message_to_db({"role": "user", "content": "from A"})
        obs_a = memory.store_tool_observation("current_time", "A observation")
        store_a = working_state.WorkingStateStore()
        store_a.begin_turn(turn_id=1, objective="task A", rolling_summary="", recalled_context="", recent_messages=[], policy_note="", tool_schemas=[], task_frame={"intent": "current_time"})
    with conversation_context(b):
        memory._save_message_to_db({"role": "user", "content": "from B"})
        store_b = working_state.WorkingStateStore()
        store_b.begin_turn(turn_id=2, objective="task B", rolling_summary="", recalled_context="", recent_messages=[], policy_note="", tool_schemas=[], task_frame={"intent": "weather", "entity": "Toronto"})
        assert "from B" in memory._load_chat_history_from_db()[0]["content"]
        assert "from A" not in str(memory._load_chat_history_from_db())
        assert "was not found" in memory.read_observation(obs_a)
        assert store_b.load()["objective"] == "task B"
    with conversation_context(a):
        assert memory._load_chat_history_from_db()[0]["content"] == "from A"
        assert json.loads(memory.read_observation(obs_a))["content"] == "A observation"
        assert working_state.WorkingStateStore().load()["objective"] == "task A"


def test_pipeline_uses_central_timeout_executor(monkeypatch):
    import tools
    from tools import executor, pipeline

    calls = []
    monkeypatch.setitem(tools.AVAILABLE_TOOLS_MAP, "unit_executor_probe", lambda value="": value)
    monkeypatch.setitem(tools.TOOL_METADATA, "unit_executor_probe", {"readonly": True})

    def fake_executor(name, args):
        calls.append((name, dict(args)))
        return json.dumps({"value": args.get("value")})

    monkeypatch.setattr(executor, "execute_registered_tool", fake_executor)
    result = pipeline.execute_pipeline([
        {"id": "probe", "tool": "unit_executor_probe", "args": {"value": "ok"}},
    ])
    assert result["ok"] is True
    assert calls == [("unit_executor_probe", {"value": "ok"})]


def test_onboarding_reset_replaces_questionnaire_owned_context(tmp_path, monkeypatch):
    from tools import user_profile

    db = str(tmp_path / "profile.db")
    profile_dir = tmp_path / "profile"
    profile_path = profile_dir / "user_picture.png"
    monkeypatch.setattr(user_profile, "DB_PATH", db)
    monkeypatch.setattr(user_profile, "PROFILE_DIR", profile_dir)
    monkeypatch.setattr(user_profile, "PROFILE_IMAGE_PATH", profile_path)
    user_profile.init_user_profile_db()

    first = user_profile.complete_onboarding_profile(
        name="First", role="Admin", timezone="America/Toronto", location="London, Ontario",
        interests=["Linux"], response_style="concise", research_depth="deep", reset=True,
    )
    assert first["completed"] is True
    assert first["identity"]["name"] == "First"
    assert "First" in user_profile.get_user_prompt_context()
    profile_dir.mkdir(parents=True, exist_ok=True)
    profile_path.write_bytes(b"legacy-picture")

    second = user_profile.complete_onboarding_profile(
        name="Second", role="Developer", timezone="UTC", location="Toronto, Ontario",
        interests=["Agents"], response_style="detailed", research_depth="balanced", reset=True,
    )
    assert second["identity"]["name"] == "Second"
    assert second["identity"]["role"] == "Developer"
    assert second["profile_image"] == ""
    assert not profile_path.exists()
    assert "First" not in user_profile.get_user_prompt_context()


def test_legacy_orphan_tool_row_is_dropped_from_model_context():
    from tools.context import fit_tool_loop_messages

    fitted = fit_tool_loop_messages(
        [{"role": "system", "content": "s"}],
        [{"role": "tool", "name": "old_tool", "content": "orphan"}, {"role": "user", "content": "hello"}],
        max_ctx_tokens=512,
        reserve_tokens=128,
    )
    assert [m["role"] for m in fitted] == ["system", "user"]


def test_weather_scope_does_not_accept_same_region_wrong_city():
    from tools.grounding import make_observation, validate_fact_grounding
    from tools.task_requirements import derive_task_frame

    request = "Weather in London Ontario tomorrow"
    frame = derive_task_frame(request)
    search = make_observation(
        "web_search",
        json.dumps([{"title": "Toronto Ontario weather", "url": "https://weather.example/toronto", "snippet": "Tomorrow forecast"}]),
        arguments={"query": "Ontario weather tomorrow"}, turn_id=11,
    )
    browse = make_observation(
        "browse_url", "URL: https://weather.example/toronto\nToronto Ontario tomorrow forecast temperature 20 C wind 10 km/h",
        arguments={"url": "https://weather.example/toronto"}, turn_id=11,
    )
    assert validate_fact_grounding(request, [search, browse], current_turn_id=11, task_frame=frame)["grounded"] is False


def test_weather_location_hint_survives_mixed_profile_and_memory_context():
    from tools.grounding import build_weather_query

    context = '''
### User Context
**Name**: Test User
**Timezone**: America/Toronto
**Location**: London, Ontario, Canada

[
  {"topic":"user_location","fact":"London, Ontario, Canada"}
]
'''
    query = build_weather_query("What is the weather for the next week?", context)
    assert "London, Ontario, Canada" in query


def test_weather_temporal_qualifiers_are_not_location_entities():
    from tools.task_requirements import derive_task_frame

    for request in (
        "What is the weather right now?",
        "What is the weather currently?",
        "What is the weather this evening?",
        "What is the weather at the moment?",
    ):
        frame = derive_task_frame(request)
        assert frame["intent"] == "weather"
        assert not frame.get("entity"), (request, frame)


def test_weather_explicit_location_strips_trailing_right_now():
    from tools.task_requirements import derive_task_frame

    frame = derive_task_frame("What is the weather in London Ontario right now?")
    assert frame["intent"] == "weather"
    assert frame.get("entity") == "London Ontario"
    assert frame.get("time_scope") == "now"


def test_weather_right_now_falls_back_to_saved_profile_location(monkeypatch):
    from tools.grounding import build_weather_query, _forecast_days_for_request
    import tools.user_profile

    monkeypatch.setattr(tools.user_profile, "get_user_location", lambda: "London, Ontario, Canada")
    query = build_weather_query("What is the weather right now?", "")
    assert "London, Ontario, Canada" in query
    assert "Rightangle" not in query
    assert _forecast_days_for_request("What is the weather right now?") == 1
