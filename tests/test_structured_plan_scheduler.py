from __future__ import annotations

import json
import tempfile
from pathlib import Path


def _tool_names(schemas):
    return {str(item.get("function", {}).get("name") or "") for item in schemas}


def test_compound_prompt_crosses_structured_plan_threshold():
    from al_agent.fast_tasks import should_compile_structured_plan

    prompt = "Check the weather in London, read my latest unread Gmail, scan the LAN, and summarize report.txt"
    assert should_compile_structured_plan(prompt, min_chars=900, min_commands=3)
    assert not should_compile_structured_plan("Check the weather in London", min_chars=900, min_commands=3)


def test_fast_compiler_uses_strict_json_array_and_no_tool_schemas():
    from al_agent.fast_tasks import STRUCTURED_PLAN_SCHEMA, compile_structured_plan

    class FakeClient:
        def __init__(self):
            self.calls = []

        def chat(self, **kwargs):
            self.calls.append(kwargs)
            return {
                "message": {
                    "content": json.dumps([
                        "Check weather in London, Ontario",
                        "Read the latest unread Gmail message",
                        "Scan the local 192.168.1.0/24 subnet",
                    ])
                }
            }

    client = FakeClient()
    objective = "Check weather in London, then read my latest unread Gmail, then scan 192.168.1.0/24"
    plan = compile_structured_plan(
        client,
        model="agent-main:2b",
        objective=objective,
        options={"num_ctx": 4096},
        min_chars=10,
        min_commands=2,
    )

    assert plan == [
        "Check weather in London, Ontario",
        "Read the latest unread Gmail message",
        "Scan the local 192.168.1.0/24 subnet",
    ]
    call = client.calls[0]
    assert call["model"] == "agent-main:2b"
    assert call["format"] == STRUCTURED_PLAN_SCHEMA
    assert call["stream"] is False
    assert call["options"]["temperature"] == 0.0
    assert "tools" not in call or call["tools"] == []
    compiler_text = "\n".join(msg["content"] for msg in call["messages"])
    assert "Return ONLY the JSON array" in compiler_text
    assert "Do not turn examples" in compiler_text


def test_fast_compiler_allows_one_item_instead_of_inventing_tasks():
    from al_agent.fast_tasks import compile_structured_plan

    class FakeClient:
        def chat(self, **kwargs):
            return {"message": {"content": '["Review the supplied architecture document"]'}}

    objective = "Review this architecture carefully. " + ("context " * 180)
    plan = compile_structured_plan(
        FakeClient(), model="agent-main:2b", objective=objective, min_chars=100, min_commands=3,
    )
    assert plan == ["Review the supplied architecture document"]


def test_fast_compiler_filters_global_constraints_from_executable_steps():
    from al_agent.fast_tasks import compile_structured_plan

    class FakeClient:
        def chat(self, **kwargs):
            return {"message": {"content": json.dumps([
                "Do not install or modify packages.",
                "Check the current local time",
                "Read the repository README",
            ])}}

    plan = compile_structured_plan(
        FakeClient(),
        model="agent-main:2b",
        objective="Do not install packages. Then check the time, then read the README. " + ("context " * 40),
        min_chars=10,
        min_commands=2,
    )
    assert plan == ["Check the current local time", "Read the repository README"]


def test_explicit_numbered_suite_is_extracted_without_compiler_or_safety_steps():
    from al_agent.fast_tasks import compile_structured_plan

    class CompilerMustNotRun:
        def chat(self, **kwargs):  # pragma: no cover - failure message is clearer
            raise AssertionError("explicit numbered requirements should bypass the model compiler")

    prompt = """# Stress Test

## Safety Rules
1. Prefer read-only operations.
2. Do not install packages.
3. Never modify network configuration.

Treat every numbered requirement below as independent.

## 1. Runtime Identity
Determine hostname and current time using runtime tools.

## 2. Host Snapshot
Collect a read-only host snapshot and report memory.

## 3. Repository Status
Inspect Git status. Do not commit anything.

## 4. Deterministic Termination
Verify retries are bounded and the test terminates cleanly.

# FINAL REPORT
Summarize all completed requirements.
"""
    plan = compile_structured_plan(
        CompilerMustNotRun(),
        model="agent-main:2b",
        objective=prompt,
        min_chars=10,
        min_commands=2,
        max_steps=96,
    )

    assert len(plan) == 4
    assert plan[0].startswith("1. Runtime Identity:")
    assert plan[-1].startswith("4. Deterministic Termination:")
    assert all("Prefer read-only operations" not in step for step in plan)
    assert all("Do not install packages" not in step for step in plan)
    assert all("FINAL REPORT" not in step for step in plan)


def test_catalog_active_task_blocks_future_intent_schema_leakage():
    from tools.catalog import select_tool_schemas

    full = (
        "Check the weather in London, then read my latest unread Gmail, "
        "then scan the local 192.168.1.0/24 subnet, then read report.txt"
    )
    schemas = select_tool_schemas(
        full,
        max_tools=3,
        context_text="",
        active_task="Check the weather in London, Ontario",
    )
    names = _tool_names(schemas)
    assert len(names) <= 3
    assert "gmail_search_messages" not in names
    assert "scan_subnet" not in names
    assert "read_file" not in names
    assert "weather_forecast" in names or "geocode_location" in names


def test_working_state_scheduler_hides_future_steps_and_advances_pointer(monkeypatch):
    from tools import working_state

    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setattr(working_state, "DB_PATH", str(Path(td) / "state.db"))
        store = working_state.WorkingStateStore(limits={"max_render_chars": 8000})
        objective = "FIRST_WEATHER then SECRET_FUTURE_GMAIL then SECRET_FUTURE_LAN"
        steps = ["FIRST_WEATHER", "SECRET_FUTURE_GMAIL", "SECRET_FUTURE_LAN"]
        store.begin_turn(
            turn_id=1,
            objective=objective,
            rolling_summary="",
            recalled_context="",
            recent_messages=[],
            policy_note="",
            tool_schemas=[],
            execution_plan=steps,
            task_frame={"intent": "weather", "source_text": "FIRST_WEATHER"},
            fact_frames={"weather": {"intent": "weather", "source_text": "FIRST_WEATHER"}},
        )

        durable = store.load()
        assert durable["scheduler"]["overall_objective"] == objective
        assert durable["scheduler"]["steps"][2]["task"] == "SECRET_FUTURE_LAN"

        projected = store.render()
        assert "FIRST_WEATHER" in projected
        assert "SECRET_FUTURE_GMAIL" not in projected
        assert "SECRET_FUTURE_LAN" not in projected
        assert store.active_requirement() == "FIRST_WEATHER"

        store.mark_active_requirement("PASS", result="weather ok")
        store.update_active_context(
            task_frame={"intent": "gmail", "source_text": "SECRET_FUTURE_GMAIL"},
            fact_frames={},
            fact_requirements=[],
        )
        projected = store.render()
        parsed = json.loads(projected)
        assert store.active_requirement() == "SECRET_FUTURE_GMAIL"
        assert "SECRET_FUTURE_GMAIL" in projected
        assert "SECRET_FUTURE_LAN" not in projected
        assert parsed["task_frame"]["source_text"] == "SECRET_FUTURE_GMAIL"
        assert parsed["scheduler"]["active_index"] == 1

        store.mark_active_requirement("FAIL", reason="mail unavailable", result="provider blocked")
        store.mark_active_requirement("PASS", result="lan scan complete")
        assert store.scheduler_complete()
        results = json.loads(store.render_scheduler_results())
        assert [row["status"] for row in results] == ["PASS", "FAIL", "PASS"]
        assert results[1]["reason"] == "mail unavailable"


def test_working_state_preserves_full_explicit_safety_section(monkeypatch):
    from tools import working_state

    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setattr(working_state, "DB_PATH", str(Path(td) / "state.db"))
        store = working_state.WorkingStateStore()
        safety = "\n".join(f"{i}. Safety rule {i}: do not perform action {i}." for i in range(1, 21))
        objective = f"# Audit\n\n## Safety Rules\n{safety}\n\n## 1. Runtime Identity\nCheck hostname."
        state = store.begin_turn(
            turn_id=1,
            objective=objective,
            rolling_summary="",
            recalled_context="",
            recent_messages=[],
            policy_note="",
            tool_schemas=[],
        )
        assert len(state["constraints"]) == 20
        assert state["constraints"][0].startswith("Safety rule 1")
        assert state["constraints"][-1].startswith("Safety rule 20")


def test_structured_plan_constraints_exclude_future_step_local_instructions(monkeypatch):
    from tools import working_state

    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setattr(working_state, "DB_PATH", str(Path(td) / "state.db"))
        store = working_state.WorkingStateStore()
        objective = """# Audit

## Safety Rules
1. Prefer read-only operations.
2. Do not install packages.

## 1. Runtime Identity
Check hostname and time.

## 2. Tool Selection
Identify a primitive without executing it yet.

## 3. Host Snapshot
Collect a read-only host snapshot. Do not infer missing hardware details.
"""
        state = store.begin_turn(
            turn_id=1,
            objective=objective,
            rolling_summary="",
            recalled_context="",
            recent_messages=[],
            policy_note="",
            tool_schemas=[],
            execution_plan=["1. Runtime Identity", "2. Tool Selection", "3. Host Snapshot"],
        )
        constraints = "\n".join(state["constraints"])
        assert "Prefer read-only operations" in constraints
        assert "Do not install packages" in constraints
        assert "without executing it yet" not in constraints
        assert "Do not infer missing hardware details" not in constraints


def test_scheduler_render_does_not_duplicate_durable_observations(monkeypatch):
    from tools import working_state

    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setattr(working_state, "DB_PATH", str(Path(td) / "state.db"))
        store = working_state.WorkingStateStore(limits={"max_render_chars": 8000})
        store.begin_turn(
            turn_id=1,
            objective="Run two checks",
            rolling_summary="",
            recalled_context="",
            recent_messages=[],
            policy_note="",
            tool_schemas=[],
            execution_plan=["FIRST", "SECOND"],
        )
        store.record_tool_result(
            tool_name="hostname", arguments={}, status="ok", reason="ok",
            result_text='{"host_hostname":"muninn"}', observation_id="obs123",
        )
        projected = json.loads(store.render())
        assert projected["verified_observations"] == []
        assert "hostname" in store.render_evidence()


def test_scheduler_final_synthesis_keeps_all_large_plan_rows(monkeypatch):
    from tools import working_state

    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setattr(working_state, "DB_PATH", str(Path(td) / "state.db"))
        store = working_state.WorkingStateStore(limits={"scheduler_steps": 96})
        steps = [f"Requirement {i}: verify subsystem {i}" for i in range(1, 78)]
        store.begin_turn(
            turn_id=1,
            objective="Run all 77 requirements",
            rolling_summary="",
            recalled_context="",
            recent_messages=[],
            policy_note="",
            tool_schemas=[],
            execution_plan=steps,
        )
        for i in range(1, 78):
            store.mark_active_requirement("PASS", result=f"verified result for subsystem {i}")

        rows = json.loads(store.render_scheduler_results(24000))
        assert len(rows) == 77
        assert rows[0]["id"] == "step-001"
        assert rows[-1]["id"] == "step-077"
        assert all(row["status"] == "PASS" for row in rows)


def test_scheduler_schema_budget_prioritizes_active_required_tool():
    # Importing turn_engine normally requires the optional Ollama package. Test
    # the helper source contract statically in environments without that package.
    source = Path("al_agent/turn_engine.py").read_text()
    assert "def _bounded_active_schemas(" in source
    assert "active_task=active_request if plan_enabled else None" in source
    assert "pending_hint = (active_requirement_ledger if plan_enabled else requirement_ledger).pending_hint()" in source
    assert 'model_user_msg["content"] = active_request' in source


def test_turn_engine_never_sends_future_steps_to_main_model(monkeypatch, tmp_path):
    from al_agent import turn_engine as te
    from tools import working_state

    class CompilerClient:
        def __init__(self):
            self.calls = []

        def chat(self, **kwargs):
            self.calls.append(kwargs)
            return {"message": {"content": json.dumps([
                "Handle ACTIVE_ALPHA",
                "Handle FUTURE_BETA",
                "Handle FUTURE_GAMMA",
            ])}}

    class MainClient:
        def __init__(self):
            self.calls = []

        def chat(self, **kwargs):
            self.calls.append(kwargs)
            index = len(self.calls)
            text = {1: "alpha complete", 2: "beta complete", 3: "gamma complete"}.get(index, "combined complete")
            return iter([
                {"message": {"content": text}, "done": False},
                {"message": {"content": ""}, "done": True, "prompt_eval_count": 8, "eval_count": 3},
            ])

    db = str(tmp_path / "scheduler.db")
    monkeypatch.setattr(working_state, "DB_PATH", db)
    store = working_state.WorkingStateStore(limits={"max_render_chars": 8000})
    compiler = CompilerClient()
    main = MainClient()

    monkeypatch.setattr(te, "WORKING_STATE", store)
    monkeypatch.setattr(te, "WORKING_STATE_ENABLED", True)
    monkeypatch.setattr(te, "STRUCTURED_PLAN_ENABLED", True)
    monkeypatch.setattr(te, "STRUCTURED_PLAN_MIN_CHARS", 10_000)
    monkeypatch.setattr(te, "STRUCTURED_PLAN_MIN_COMMANDS", 2)
    monkeypatch.setattr(te, "STRUCTURED_PLAN_MAX_TOOLS", 3)
    monkeypatch.setattr(te, "GROUNDING_ENABLED", False)
    monkeypatch.setattr(te, "RECIPES_ENABLED", False)
    monkeypatch.setattr(te, "LOOP_VALIDATOR_ENABLED", False)
    monkeypatch.setattr(te, "MODEL_TRACE_ENABLED", False)
    monkeypatch.setattr(te, "get_conversation_summary", lambda: "")
    monkeypatch.setattr(te, "build_memory_context", lambda *_: "")
    monkeypatch.setattr(te, "get_relevant_user_prompt_context", lambda *_: "")
    monkeypatch.setattr(te, "get_user_location", lambda: "")
    monkeypatch.setattr(te, "build_historical_recall_context", lambda *a, **k: "")
    monkeypatch.setattr(te, "render_relevant_skill_index", lambda *a, **k: "")
    monkeypatch.setattr(te, "render_failure_lessons", lambda *a, **k: "")
    monkeypatch.setattr(te, "render_relevant_reflections", lambda *a, **k: "")
    monkeypatch.setattr(te, "evict_report_model_for_interactive", lambda: None)
    monkeypatch.setattr(te, "_prune_compacted_history", lambda _messages: None)
    monkeypatch.setattr(te, "log_perf_stats", lambda *a, **k: None)

    messages = [{"role": "system", "content": "system"}]
    te.handle_user_turn(
        messages,
        "Check ALPHA, then check BETA, then check GAMMA",
        False,
        runtime_overrides={
            "OLLAMA": main,
            "LOOP_VALIDATOR_CLIENT": compiler,
            "record_monitor_state": lambda *a, **k: None,
            "append_and_save": lambda rows, item: rows.append(item),
            "acquire_turn_lock": lambda: object(),
            "release_turn_lock": lambda _lock: None,
            "acquire_inference_lock": lambda: object(),
            "release_inference_lock": lambda _lock: None,
            "queue_compaction_if_needed": lambda *a, **k: None,
        },
    )

    assert len(compiler.calls) == 1
    assert len(main.calls) == 4
    first = json.dumps(main.calls[0]["messages"], ensure_ascii=False)
    second = json.dumps(main.calls[1]["messages"], ensure_ascii=False)
    third = json.dumps(main.calls[2]["messages"], ensure_ascii=False)
    final = json.dumps(main.calls[3]["messages"], ensure_ascii=False)

    assert "ACTIVE_ALPHA" in first
    assert "FUTURE_BETA" not in first
    assert "FUTURE_GAMMA" not in first
    assert "FUTURE_BETA" in second
    assert "FUTURE_GAMMA" not in second
    assert "FUTURE_GAMMA" in third
    assert "alpha complete" not in second
    assert "beta complete" not in third
    assert second.count("[Harness scheduler] Active requirement ") == 1
    assert third.count("[Harness scheduler] Active requirement ") == 1
    assert "Active requirement 2/3: Handle FUTURE_BETA" not in third
    # Full completed scheduler results become visible only to the final synthesis.
    assert "ACTIVE_ALPHA" in final and "FUTURE_BETA" in final and "FUTURE_GAMMA" in final
    assert messages[-1]["content"] == "combined complete"


def test_explicit_numbered_plan_does_not_attach_next_phase_heading():
    from al_agent.fast_tasks import compile_structured_plan

    class CompilerMustNotRun:
        def chat(self, **kwargs):
            raise AssertionError("explicit numbered requirements should bypass the compiler")

    prompt = """# Audit
Treat every numbered requirement below as independent.

# PHASE 1 — DISCOVERY
## 1. Tool Selection
Identify the appropriate primitive without executing it.

# PHASE 2 — HOST
## 2. Host Snapshot
Collect the host snapshot.

# FINAL REPORT
Summarize the results.
"""
    plan = compile_structured_plan(
        CompilerMustNotRun(), model="agent-main:2b", objective=prompt,
        min_chars=10, min_commands=2, max_steps=96,
    )
    assert len(plan) == 2
    assert plan[0] == "1. Tool Selection: Identify the appropriate primitive without executing it."
    assert "PHASE 2" not in plan[0]


def test_selection_only_capability_digest_uses_metadata_not_executable_calls():
    from al_agent.turn_engine import _selection_only_capability_digest

    request = (
        "3. Tool Selection: identify the most appropriate primitive without executing it yet: "
        "* current time * CPU information * memory usage * network routes * DNS lookup "
        "* current weather * public webpage retrieval * repository status * reading a local file "
        "* parsing JSON * running a calculation * historical conversation recall"
    )
    digest = _selection_only_capability_digest(request)
    for name in (
        "current_time", "cpu_info", "memory_info", "route_list", "dns_query", "weather_forecast",
        "browse_url", "repo_status", "read_file", "json_query", "calculate", "search_conversation_history",
    ):
        assert name in digest
    assert "DO NOT execute these tools" in digest


def test_working_state_cannot_complete_with_pending_scheduler_steps(monkeypatch):
    from tools import working_state

    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setattr(working_state, "DB_PATH", str(Path(td) / "state.db"))
        store = working_state.WorkingStateStore()
        store.begin_turn(
            turn_id=1,
            objective="Run two steps",
            rolling_summary="",
            recalled_context="",
            recent_messages=[],
            policy_note="",
            tool_schemas=[],
            execution_plan=["FIRST", "SECOND"],
        )

        store.complete_turn(blocked=False)
        assert store.load()["status"] == "active"

        store.mark_active_requirement("PASS", result="first done")
        store.complete_turn(blocked=False)
        assert store.load()["status"] == "active"

        store.mark_active_requirement("PASS", result="second done")
        store.complete_turn(blocked=False)
        assert store.load()["status"] == "complete"


def test_structured_plan_pregrounding_does_not_end_turn_after_current_time(monkeypatch, tmp_path):
    """Regression for a plan ending after its first deterministic grounding call.

    Step 1 needs both current_time and hostname.  current_time is pre-grounded by
    the harness before the first main-model request; that must not trigger the
    whole-turn deterministic fast path while hostname and later scheduler steps
    remain unresolved.
    """
    from al_agent import turn_engine as te
    from tools import working_state

    calls = []

    def fake_execute(name, args):
        calls.append((name, dict(args or {})))
        if name == "current_time":
            return json.dumps({
                "utc": "2026-09-24T14:03:35+00:00",
                "local": "2026-09-24T10:03:35-04:00",
                "date": "2026-09-24",
                "time": "10:03:35",
                "day_of_week": "Thursday",
                "timezone": "America/Toronto",
                "timezone_abbreviation": "EDT",
                "utc_offset": "-0400",
                "unix_timestamp": 1790258615,
                "host_timezone": "Canada/Eastern",
                "system_local": "2026-09-24T10:03:35-04:00",
            })
        if name == "hostname":
            return json.dumps({"host_hostname": "muninn", "runtime_hostname": "muninn", "same_hostname": True})
        raise AssertionError((name, args))

    class MainClient:
        def __init__(self):
            self.calls = []

        def chat(self, **kwargs):
            self.calls.append(kwargs)
            index = len(self.calls)
            if index == 1:
                names = _tool_names(kwargs.get("tools") or [])
                assert "hostname" in names
                return iter([{"done": True, "message": {"content": "", "tool_calls": [
                    {"id": "h1", "function": {"name": "hostname", "arguments": {}}}
                ]}}])
            if index == 2:
                return iter([{"done": True, "message": {"content": "runtime identity verified", "tool_calls": []}}])
            if index == 3:
                assert kwargs.get("tools") == []
                return iter([{"done": True, "message": {"content": "hostname is the appropriate primitive", "tool_calls": []}}])
            return iter([{"done": True, "message": {"content": "all scheduled checks complete", "tool_calls": []}}])

    db = str(tmp_path / "onecall-regression.db")
    monkeypatch.setattr(working_state, "DB_PATH", db)
    store = working_state.WorkingStateStore(limits={"max_render_chars": 8000})
    model = MainClient()

    monkeypatch.setattr(te, "WORKING_STATE", store)
    monkeypatch.setattr(te, "WORKING_STATE_ENABLED", True)
    monkeypatch.setattr(te, "STRUCTURED_PLAN_ENABLED", True)
    monkeypatch.setattr(te, "STRUCTURED_PLAN_MIN_CHARS", 10)
    monkeypatch.setattr(te, "STRUCTURED_PLAN_MIN_COMMANDS", 2)
    monkeypatch.setattr(te, "STRUCTURED_PLAN_MAX_TOOLS", 3)
    monkeypatch.setattr(te, "GROUNDING_ENABLED", True)
    monkeypatch.setattr(te, "RECIPES_ENABLED", False)
    monkeypatch.setattr(te, "LOOP_VALIDATOR_ENABLED", False)
    monkeypatch.setattr(te, "MODEL_TRACE_ENABLED", False)
    monkeypatch.setattr(te, "get_conversation_summary", lambda: "")
    monkeypatch.setattr(te, "build_memory_context", lambda *_: "")
    monkeypatch.setattr(te, "get_relevant_user_prompt_context", lambda *_: "")
    monkeypatch.setattr(te, "get_user_location", lambda: "")
    monkeypatch.setattr(te, "build_historical_recall_context", lambda *a, **k: "")
    monkeypatch.setattr(te, "render_relevant_skill_index", lambda *a, **k: "")
    monkeypatch.setattr(te, "render_failure_lessons", lambda *a, **k: "")
    monkeypatch.setattr(te, "render_relevant_reflections", lambda *a, **k: "")
    monkeypatch.setattr(te, "evict_report_model_for_interactive", lambda: None)
    monkeypatch.setattr(te, "_prune_compacted_history", lambda _messages: None)
    monkeypatch.setattr(te, "log_perf_stats", lambda *a, **k: None)
    monkeypatch.setattr(te, "_execute_registered_tool", fake_execute)

    prompt = """# Scheduler regression
Treat every numbered requirement below as independent.

## 1. Runtime Identity
Determine using runtime/system tools: hostname and current local time. Verify each value from tool output.

## 2. Tool Selection
Identify the appropriate primitive for hostname without executing it.

# FINAL REPORT
Summarize all completed requirements.
"""
    messages = [{"role": "system", "content": "system"}]
    te.handle_user_turn(
        messages,
        prompt,
        False,
        runtime_overrides={
            "OLLAMA": model,
            "LOOP_VALIDATOR_CLIENT": model,
            "record_monitor_state": lambda *a, **k: None,
            "append_and_save": lambda rows, item: rows.append(item),
            "acquire_turn_lock": lambda: object(),
            "release_turn_lock": lambda _lock: None,
            "acquire_inference_lock": lambda: object(),
            "release_inference_lock": lambda _lock: None,
            "queue_compaction_if_needed": lambda *a, **k: None,
        },
    )

    assert [name for name, _ in calls].count("current_time") >= 1
    assert [name for name, _ in calls].count("hostname") == 1
    assert len(model.calls) >= 4
    assert store.scheduler_complete()
    assert store.load()["status"] == "complete"
    assert messages[-1]["content"] == "all scheduled checks complete"


def test_structured_plan_model_timeout_fails_one_step_and_continues(monkeypatch, tmp_path):
    from al_agent import turn_engine as te
    from tools import working_state

    class MainClient:
        def __init__(self):
            self.calls = []

        def chat(self, **kwargs):
            self.calls.append(kwargs)
            index = len(self.calls)
            if index == 1:
                return iter([{"done": True, "message": {"content": "first selection complete", "tool_calls": []}}])
            if index in {2, 3}:
                raise TimeoutError("timed out")
            if index == 4:
                return iter([{"done": True, "message": {"content": "third selection complete", "tool_calls": []}}])
            return iter([{"done": True, "message": {"content": "final synthesis complete", "tool_calls": []}}])

    db = str(tmp_path / "step-timeout.db")
    monkeypatch.setattr(working_state, "DB_PATH", db)
    store = working_state.WorkingStateStore(limits={"max_render_chars": 8000})
    model = MainClient()

    monkeypatch.setattr(te, "WORKING_STATE", store)
    monkeypatch.setattr(te, "WORKING_STATE_ENABLED", True)
    monkeypatch.setattr(te, "STRUCTURED_PLAN_ENABLED", True)
    monkeypatch.setattr(te, "STRUCTURED_PLAN_MIN_CHARS", 10)
    monkeypatch.setattr(te, "STRUCTURED_PLAN_MIN_COMMANDS", 2)
    monkeypatch.setattr(te, "STRUCTURED_PLAN_MAX_TOOLS", 3)
    monkeypatch.setattr(te, "GROUNDING_ENABLED", False)
    monkeypatch.setattr(te, "RECIPES_ENABLED", False)
    monkeypatch.setattr(te, "LOOP_VALIDATOR_ENABLED", False)
    monkeypatch.setattr(te, "MODEL_TRACE_ENABLED", False)
    monkeypatch.setattr(te, "MODEL_PREFLIGHT_RETRIES", 0)
    monkeypatch.setattr(te, "MODEL_NO_PROGRESS_MAX_RETRIES", 2)
    monkeypatch.setattr(te, "get_conversation_summary", lambda: "")
    monkeypatch.setattr(te, "build_memory_context", lambda *_: "")
    monkeypatch.setattr(te, "get_relevant_user_prompt_context", lambda *_: "")
    monkeypatch.setattr(te, "get_user_location", lambda: "")
    monkeypatch.setattr(te, "build_historical_recall_context", lambda *a, **k: "")
    monkeypatch.setattr(te, "render_relevant_skill_index", lambda *a, **k: "")
    monkeypatch.setattr(te, "render_failure_lessons", lambda *a, **k: "")
    monkeypatch.setattr(te, "render_relevant_reflections", lambda *a, **k: "")
    monkeypatch.setattr(te, "evict_report_model_for_interactive", lambda: None)
    monkeypatch.setattr(te, "_prune_compacted_history", lambda _messages: None)
    monkeypatch.setattr(te, "log_perf_stats", lambda *a, **k: None)

    prompt = """# Timeout isolation regression
Treat every numbered requirement below as independent.

## 1. Tool Selection
Identify the appropriate primitive for current time without executing it yet.

## 2. Tool Selection
Identify the appropriate primitive for CPU information without executing it yet.

## 3. Tool Selection
Identify the appropriate primitive for memory usage without executing it yet.

# FINAL REPORT
Summarize every requirement.
"""
    messages = [{"role": "system", "content": "system"}]
    te.handle_user_turn(
        messages,
        prompt,
        False,
        runtime_overrides={
            "OLLAMA": model,
            "LOOP_VALIDATOR_CLIENT": model,
            "record_monitor_state": lambda *a, **k: None,
            "append_and_save": lambda rows, item: rows.append(item),
            "acquire_turn_lock": lambda: object(),
            "release_turn_lock": lambda _lock: None,
            "acquire_inference_lock": lambda: object(),
            "release_inference_lock": lambda _lock: None,
            "queue_compaction_if_needed": lambda *a, **k: None,
        },
    )

    scheduler = store.scheduler_snapshot()
    assert [step["status"] for step in scheduler["steps"]] == ["PASS", "FAIL", "PASS"]
    assert store.load()["status"] == "complete"
    assert len(model.calls) == 5
    assert messages[-1]["content"] == "final synthesis complete"


def test_scheduler_step_boundary_drops_prior_tool_protocol(monkeypatch, tmp_path):
    from al_agent import turn_engine as te
    from tools import working_state

    executed = []

    def fake_execute(name, args):
        executed.append((name, dict(args or {})))
        if name == "hostname":
            return json.dumps({"host_hostname": "muninn", "runtime_hostname": "muninn", "same_hostname": True})
        raise AssertionError((name, args))

    class MainClient:
        def __init__(self):
            self.calls = []

        def chat(self, **kwargs):
            self.calls.append(kwargs)
            index = len(self.calls)
            if index == 1:
                return iter([{"done": True, "message": {"content": "", "tool_calls": [
                    {"id": "first-host", "function": {"name": "hostname", "arguments": {}}}
                ]}}])
            if index == 2:
                return iter([{"done": True, "message": {"content": "first hostname verified", "tool_calls": []}}])
            if index == 3:
                wire = json.dumps(kwargs.get("messages") or [], ensure_ascii=False)
                assert "first-host" not in wire
                assert "first hostname verified" not in wire
                return iter([{"done": True, "message": {"content": "", "tool_calls": [
                    {"id": "second-host", "function": {"name": "hostname", "arguments": {}}}
                ]}}])
            if index == 4:
                return iter([{"done": True, "message": {"content": "second hostname verified", "tool_calls": []}}])
            return iter([{"done": True, "message": {"content": "final", "tool_calls": []}}])

    monkeypatch.setattr(working_state, "DB_PATH", str(tmp_path / "tail-reset.db"))
    store = working_state.WorkingStateStore(limits={"max_render_chars": 8000})
    model = MainClient()

    monkeypatch.setattr(te, "WORKING_STATE", store)
    monkeypatch.setattr(te, "WORKING_STATE_ENABLED", True)
    monkeypatch.setattr(te, "STRUCTURED_PLAN_ENABLED", True)
    monkeypatch.setattr(te, "STRUCTURED_PLAN_MIN_CHARS", 10)
    monkeypatch.setattr(te, "STRUCTURED_PLAN_MIN_COMMANDS", 2)
    monkeypatch.setattr(te, "STRUCTURED_PLAN_MAX_TOOLS", 3)
    monkeypatch.setattr(te, "GROUNDING_ENABLED", False)
    monkeypatch.setattr(te, "RECIPES_ENABLED", False)
    monkeypatch.setattr(te, "LOOP_VALIDATOR_ENABLED", False)
    monkeypatch.setattr(te, "MODEL_TRACE_ENABLED", False)
    monkeypatch.setattr(te, "get_conversation_summary", lambda: "")
    monkeypatch.setattr(te, "build_memory_context", lambda *_: "")
    monkeypatch.setattr(te, "get_relevant_user_prompt_context", lambda *_: "")
    monkeypatch.setattr(te, "get_user_location", lambda: "")
    monkeypatch.setattr(te, "build_historical_recall_context", lambda *a, **k: "")
    monkeypatch.setattr(te, "render_relevant_skill_index", lambda *a, **k: "")
    monkeypatch.setattr(te, "render_failure_lessons", lambda *a, **k: "")
    monkeypatch.setattr(te, "render_relevant_reflections", lambda *a, **k: "")
    monkeypatch.setattr(te, "evict_report_model_for_interactive", lambda: None)
    monkeypatch.setattr(te, "_prune_compacted_history", lambda _messages: None)
    monkeypatch.setattr(te, "log_perf_stats", lambda *a, **k: None)
    monkeypatch.setattr(te, "_execute_registered_tool", fake_execute)

    prompt = """# Tail isolation
Treat every numbered requirement below as independent.
This regression intentionally contains two independent scheduler requirements so the structured-plan path is exercised.
The purpose is to verify that completed tool protocol from one atomic step is not replayed into the next atomic step.

## 1. Hostname
Check hostname using the hostname primitive.

## 2. Hostname Cross-check
Cross-check hostname using the hostname primitive.

# FINAL REPORT
Summarize both checks.
"""
    messages = [{"role": "system", "content": "system"}]
    te.handle_user_turn(
        messages,
        prompt,
        False,
        runtime_overrides={
            "OLLAMA": model,
            "LOOP_VALIDATOR_CLIENT": model,
            "record_monitor_state": lambda *a, **k: None,
            "append_and_save": lambda rows, item: rows.append(item),
            "acquire_turn_lock": lambda: object(),
            "release_turn_lock": lambda _lock: None,
            "acquire_inference_lock": lambda: object(),
            "release_inference_lock": lambda _lock: None,
            "queue_compaction_if_needed": lambda *a, **k: None,
        },
    )

    assert [name for name, _ in executed] == ["hostname", "hostname"]
    assert store.scheduler_complete()
    assert messages[-1]["content"] == "final"


def test_operational_scheduler_result_persists_evidence_not_model_claims(monkeypatch, tmp_path):
    from al_agent import turn_engine as te
    from tools import working_state

    def fake_execute(name, args):
        assert name == "hostname"
        return json.dumps({"host_hostname": "muninn", "runtime_hostname": "muninn", "same_hostname": True})

    class MainClient:
        def __init__(self):
            self.calls = []

        def chat(self, **kwargs):
            self.calls.append(kwargs)
            index = len(self.calls)
            if index == 1:
                return iter([{"done": True, "message": {"content": "", "tool_calls": [
                    {"id": "host-1", "function": {"name": "hostname", "arguments": {}}}
                ]}}])
            if index == 2:
                return iter([{"done": True, "message": {
                    "content": "Hostname muninn. Kernel 5.15.0-FAKE. All values verified.", "tool_calls": []
                }}])
            if index == 3:
                return iter([{"done": True, "message": {"content": "hostname maps to hostname", "tool_calls": []}}])
            return iter([{"done": True, "message": {"content": "final synthesis", "tool_calls": []}}])

    monkeypatch.setattr(working_state, "DB_PATH", str(tmp_path / "verified-result.db"))
    store = working_state.WorkingStateStore(limits={"max_render_chars": 8000})
    model = MainClient()

    monkeypatch.setattr(te, "WORKING_STATE", store)
    monkeypatch.setattr(te, "WORKING_STATE_ENABLED", True)
    monkeypatch.setattr(te, "STRUCTURED_PLAN_ENABLED", True)
    monkeypatch.setattr(te, "STRUCTURED_PLAN_MIN_CHARS", 10)
    monkeypatch.setattr(te, "STRUCTURED_PLAN_MIN_COMMANDS", 2)
    monkeypatch.setattr(te, "STRUCTURED_PLAN_MAX_TOOLS", 3)
    monkeypatch.setattr(te, "GROUNDING_ENABLED", False)
    monkeypatch.setattr(te, "RECIPES_ENABLED", False)
    monkeypatch.setattr(te, "LOOP_VALIDATOR_ENABLED", False)
    monkeypatch.setattr(te, "MODEL_TRACE_ENABLED", False)
    monkeypatch.setattr(te, "get_conversation_summary", lambda: "")
    monkeypatch.setattr(te, "build_memory_context", lambda *_: "")
    monkeypatch.setattr(te, "get_relevant_user_prompt_context", lambda *_: "")
    monkeypatch.setattr(te, "get_user_location", lambda: "")
    monkeypatch.setattr(te, "build_historical_recall_context", lambda *a, **k: "")
    monkeypatch.setattr(te, "render_relevant_skill_index", lambda *a, **k: "")
    monkeypatch.setattr(te, "render_failure_lessons", lambda *a, **k: "")
    monkeypatch.setattr(te, "render_relevant_reflections", lambda *a, **k: "")
    monkeypatch.setattr(te, "evict_report_model_for_interactive", lambda: None)
    monkeypatch.setattr(te, "_prune_compacted_history", lambda _messages: None)
    monkeypatch.setattr(te, "log_perf_stats", lambda *a, **k: None)
    monkeypatch.setattr(te, "_execute_registered_tool", fake_execute)

    prompt = """# Evidence-backed scheduler result
Treat every numbered requirement below as independent.
This plan verifies that model prose cannot become the durable factual result for an operational step.

## 1. Hostname
Check hostname using the hostname primitive and verify it from tool output.

## 2. Tool Selection
Identify the hostname primitive without executing it yet.

# FINAL REPORT
Summarize both requirements.
"""
    messages = [{"role": "system", "content": "system"}]
    te.handle_user_turn(
        messages,
        prompt,
        False,
        runtime_overrides={
            "OLLAMA": model,
            "LOOP_VALIDATOR_CLIENT": model,
            "record_monitor_state": lambda *a, **k: None,
            "append_and_save": lambda rows, item: rows.append(item),
            "acquire_turn_lock": lambda: object(),
            "release_turn_lock": lambda _lock: None,
            "acquire_inference_lock": lambda: object(),
            "release_inference_lock": lambda _lock: None,
            "queue_compaction_if_needed": lambda *a, **k: None,
        },
    )

    scheduler = store.scheduler_snapshot()
    assert scheduler["steps"][0]["status"] == "PASS"
    result = scheduler["steps"][0]["result"]
    assert "Verified scheduler evidence:" in result
    assert "muninn" in result
    assert "5.15.0-FAKE" not in result
    assert scheduler["steps"][1]["result"] == "hostname maps to hostname"
    assert store.scheduler_complete()


def test_many_selection_steps_keep_provider_prompt_bounded(monkeypatch, tmp_path):
    from al_agent import turn_engine as te
    from tools import working_state

    class MainClient:
        def __init__(self):
            self.calls = []

        def chat(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) <= 20:
                return iter([{"done": True, "message": {"content": f"selection {len(self.calls)} complete", "tool_calls": []}}])
            return iter([{"done": True, "message": {"content": "final bounded synthesis", "tool_calls": []}}])

    monkeypatch.setattr(working_state, "DB_PATH", str(tmp_path / "bounded-prompts.db"))
    store = working_state.WorkingStateStore(limits={"max_render_chars": 8000})
    model = MainClient()
    monkeypatch.setattr(te, "WORKING_STATE", store)
    monkeypatch.setattr(te, "WORKING_STATE_ENABLED", True)
    monkeypatch.setattr(te, "STRUCTURED_PLAN_ENABLED", True)
    monkeypatch.setattr(te, "STRUCTURED_PLAN_MIN_CHARS", 10)
    monkeypatch.setattr(te, "STRUCTURED_PLAN_MIN_COMMANDS", 2)
    monkeypatch.setattr(te, "GROUNDING_ENABLED", False)
    monkeypatch.setattr(te, "RECIPES_ENABLED", False)
    monkeypatch.setattr(te, "LOOP_VALIDATOR_ENABLED", False)
    monkeypatch.setattr(te, "MODEL_TRACE_ENABLED", False)
    monkeypatch.setattr(te, "get_conversation_summary", lambda: "")
    monkeypatch.setattr(te, "build_memory_context", lambda *_: "")
    monkeypatch.setattr(te, "get_relevant_user_prompt_context", lambda *_: "")
    monkeypatch.setattr(te, "get_user_location", lambda: "")
    monkeypatch.setattr(te, "build_historical_recall_context", lambda *a, **k: "")
    monkeypatch.setattr(te, "render_relevant_skill_index", lambda *a, **k: "")
    monkeypatch.setattr(te, "render_failure_lessons", lambda *a, **k: "")
    monkeypatch.setattr(te, "render_relevant_reflections", lambda *a, **k: "")
    monkeypatch.setattr(te, "evict_report_model_for_interactive", lambda: None)
    monkeypatch.setattr(te, "_prune_compacted_history", lambda _messages: None)
    monkeypatch.setattr(te, "log_perf_stats", lambda *a, **k: None)

    steps = "\n\n".join(
        f"## {i}. Tool Selection\nIdentify the current-time primitive without executing it yet."
        for i in range(1, 21)
    )
    prompt = (
        "# Bounded scheduler prompts\nTreat every numbered requirement below as independent.\n\n"
        + steps
        + "\n\n# FINAL REPORT\nSummarize all requirements.\n"
    )
    messages = [{"role": "system", "content": "system"}]
    te.handle_user_turn(
        messages, prompt, False,
        runtime_overrides={
            "OLLAMA": model,
            "LOOP_VALIDATOR_CLIENT": model,
            "record_monitor_state": lambda *a, **k: None,
            "append_and_save": lambda rows, item: rows.append(item),
            "acquire_turn_lock": lambda: object(),
            "release_turn_lock": lambda _lock: None,
            "acquire_inference_lock": lambda: object(),
            "release_inference_lock": lambda _lock: None,
            "queue_compaction_if_needed": lambda *a, **k: None,
        },
    )

    assert len(model.calls) == 21
    payload_sizes = [len(json.dumps(call.get("messages") or [], ensure_ascii=False)) for call in model.calls[:20]]
    # Step number/completed_count changes by a few bytes, but old step protocol
    # must not accumulate linearly across the plan.
    assert max(payload_sizes) - min(payload_sizes) < 1200
    assert payload_sizes[-1] < payload_sizes[0] + 1200
    assert store.scheduler_complete()


def test_zero_tool_protocol_failures_only_fail_active_scheduler_step(monkeypatch, tmp_path):
    from al_agent import turn_engine as te
    from tools import working_state

    class MainClient:
        def __init__(self):
            self.calls = []

        def chat(self, **kwargs):
            self.calls.append(kwargs)
            index = len(self.calls)
            if index == 1:
                return iter([{"done": True, "message": {"content": "first complete", "tool_calls": []}}])
            if index in {2, 3}:
                return iter([{"done": True, "message": {"content": "", "tool_calls": [
                    {"id": f"bogus-{index}", "function": {"name": "invented_tool", "arguments": {}}}
                ]}}])
            if index == 4:
                return iter([{"done": True, "message": {"content": "third complete", "tool_calls": []}}])
            return iter([{"done": True, "message": {"content": "final", "tool_calls": []}}])

    monkeypatch.setattr(working_state, "DB_PATH", str(tmp_path / "zero-tool-step.db"))
    store = working_state.WorkingStateStore(limits={"max_render_chars": 8000})
    model = MainClient()
    monkeypatch.setattr(te, "WORKING_STATE", store)
    monkeypatch.setattr(te, "WORKING_STATE_ENABLED", True)
    monkeypatch.setattr(te, "STRUCTURED_PLAN_ENABLED", True)
    monkeypatch.setattr(te, "STRUCTURED_PLAN_MIN_CHARS", 10)
    monkeypatch.setattr(te, "STRUCTURED_PLAN_MIN_COMMANDS", 2)
    monkeypatch.setattr(te, "GROUNDING_ENABLED", False)
    monkeypatch.setattr(te, "RECIPES_ENABLED", False)
    monkeypatch.setattr(te, "LOOP_VALIDATOR_ENABLED", False)
    monkeypatch.setattr(te, "MODEL_TRACE_ENABLED", False)
    monkeypatch.setattr(te, "MODEL_NO_PROGRESS_MAX_RETRIES", 2)
    monkeypatch.setattr(te, "get_conversation_summary", lambda: "")
    monkeypatch.setattr(te, "build_memory_context", lambda *_: "")
    monkeypatch.setattr(te, "get_relevant_user_prompt_context", lambda *_: "")
    monkeypatch.setattr(te, "get_user_location", lambda: "")
    monkeypatch.setattr(te, "build_historical_recall_context", lambda *a, **k: "")
    monkeypatch.setattr(te, "render_relevant_skill_index", lambda *a, **k: "")
    monkeypatch.setattr(te, "render_failure_lessons", lambda *a, **k: "")
    monkeypatch.setattr(te, "render_relevant_reflections", lambda *a, **k: "")
    monkeypatch.setattr(te, "evict_report_model_for_interactive", lambda: None)
    monkeypatch.setattr(te, "_prune_compacted_history", lambda _messages: None)
    monkeypatch.setattr(te, "log_perf_stats", lambda *a, **k: None)

    prompt = """# Zero-tool failure isolation
Treat every numbered requirement below as independent.

## 1. Tool Selection
Identify the time primitive without executing it yet.

## 2. Tool Selection
Identify the CPU primitive without executing it yet.

## 3. Tool Selection
Identify the memory primitive without executing it yet.

# FINAL REPORT
Summarize all requirements.
"""
    messages = [{"role": "system", "content": "system"}]
    te.handle_user_turn(
        messages, prompt, False,
        runtime_overrides={
            "OLLAMA": model,
            "LOOP_VALIDATOR_CLIENT": model,
            "record_monitor_state": lambda *a, **k: None,
            "append_and_save": lambda rows, item: rows.append(item),
            "acquire_turn_lock": lambda: object(),
            "release_turn_lock": lambda _lock: None,
            "acquire_inference_lock": lambda: object(),
            "release_inference_lock": lambda _lock: None,
            "queue_compaction_if_needed": lambda *a, **k: None,
        },
    )

    scheduler = store.scheduler_snapshot()
    assert [step["status"] for step in scheduler["steps"]] == ["PASS", "FAIL", "PASS"]
    assert store.load()["status"] == "complete"
    assert messages[-1]["content"] == "final"


def test_explicit_numbered_suite_does_not_request_inference_slot():
    from al_agent.fast_tasks import compile_structured_plan

    class CompilerMustNotRun:
        def chat(self, **kwargs):  # pragma: no cover
            raise AssertionError("explicit numbered requirements should bypass the model compiler")

    prompt = """# Stress Test

Complete the entire suite. Treat every numbered requirement below as independent.

## 1. Runtime Identity
Determine hostname and current time using runtime tools.

## 2. Tool Registry
Inspect registered tool health.

## 3. Storage
Inspect mounted filesystems.
"""
    lock_requests = []

    plan = compile_structured_plan(
        CompilerMustNotRun(),
        model="agent-main:2b",
        objective=prompt,
        min_chars=10,
        min_commands=2,
        max_steps=96,
        before_model_call=lambda: lock_requests.append("model-lock"),
    )

    assert len(plan) == 3
    assert lock_requests == []


def test_fast_plan_compiler_requests_inference_slot_only_when_model_is_needed():
    from al_agent.fast_tasks import compile_structured_plan

    order = []

    class FakeClient:
        def chat(self, **kwargs):
            order.append("chat")
            return {"message": {"content": '["Check weather", "Read README"]'}}

    plan = compile_structured_plan(
        FakeClient(),
        model="agent-main:2b",
        objective="Check weather, then read README. " + ("context " * 30),
        min_chars=10,
        min_commands=2,
        before_model_call=lambda: order.append("lock"),
    )

    assert plan == ["Check weather", "Read README"]
    assert order == ["lock", "chat"]
