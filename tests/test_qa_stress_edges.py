from __future__ import annotations

import json
import os
import psutil


def test_shell_success_reports_exit_code_and_replaces_invalid_utf8(monkeypatch, tmp_path):
    import tools.system as system

    monkeypatch.setattr(system, "WORKSPACE_DIR", str(tmp_path))
    result = system.execute_shell(
        "python -c \"import os; os.write(1,b'\\xff\\n'); os.write(2,b'\\xfe\\n')\""
    )
    assert "EXIT_CODE: 0" in result
    assert "STDOUT:" in result and "STDERR:" in result
    assert "Execution error" not in result
    assert "�" in result


def test_shell_timeout_kills_descendant_process_group(monkeypatch, tmp_path):
    import tools.system as system

    monkeypatch.setattr(system, "WORKSPACE_DIR", str(tmp_path))
    marker = tmp_path / "orphan.txt"
    pid_file = tmp_path / "descendant.pid"
    result = system.execute_shell(
        "sh -c 'echo $$ > descendant.pid; sleep 5; echo orphan > orphan.txt' & wait",
        timeout=1,
    )
    assert result.startswith("Error: Command timed out")
    assert pid_file.exists(), "descendant did not start before the timeout"
    descendant_pid = int(pid_file.read_text().strip())
    try:
        descendant = psutil.Process(descendant_pid)
    except psutil.NoSuchProcess:
        descendant = None
    if descendant is not None:
        _, alive = psutil.wait_procs([descendant], timeout=3.0)
        assert not alive, "descendant process survived the process-group kill"
    assert not marker.exists(), "orphaned descendant executed after timeout"


def test_structured_parsers_strip_leading_utf8_bom():
    from tools.primitive_modules.structured import csv_query, json_query

    csv_text = '\ufeffname,notes\nAlice,"hello, world"\n'
    rows = json.loads(csv_query(column="name", equals="Alice", text=csv_text))
    assert rows == [{"name": "Alice", "notes": "hello, world"}]
    assert json.loads(json_query(data='\ufeff{"a": 1}', path_expr="$.a")) == 1


def test_csv_summary_marks_bounded_source_as_partial():
    from tools.primitive_modules.structured import csv_summary

    text = "a,b\n" + "".join(f"{i},{i * 2}\n" for i in range(30000))
    payload = json.loads(csv_summary(text=text, sample_rows=0))
    assert payload["source_truncated"] is True
    assert "partial" in payload["warning"].lower()
    # The uncertain partial final record must not be counted as a real row.
    assert payload["missing"]["b"] == 0


def test_jsonl_summary_tolerates_bad_records_and_reports_line_numbers():
    from tools.primitive_modules.structured import jsonl_summary

    payload = json.loads(jsonl_summary(text='{"id":1}\nnot-json\n{"id":2}\n'))
    assert payload["valid_records"] == 2
    assert payload["malformed_count"] == 1
    assert payload["malformed"][0]["line"] == 2
    assert payload["source_truncated"] is False


def test_jsonl_prompt_selects_native_tolerant_parser():
    from tools import load_tools, select_tool_schemas

    load_tools()
    selected = {
        item["function"]["name"]
        for item in select_tool_schemas(
            "Parse logs/events.jsonl, keep valid JSON records, and report malformed line numbers.",
            max_tools=12,
        )
    }
    assert "jsonl_summary" in selected


def test_memory_search_ignores_conversational_stopword_matches(monkeypatch, tmp_path):
    import tools.memory as memory

    monkeypatch.setattr(memory, "DB_PATH", str(tmp_path / "memory.db"))
    memory.init_db()
    memory.remember("favorite_color", "my favorite color is green")
    memory.remember("vacation", "you asked about Paris earlier")
    assert memory.search_memory(
        "What did I tell you my deployment target was earlier?"
    ) == "No related memories found."
    memory.remember("deployment_target", "The deployment target is staging-eu-west")
    rows = json.loads(memory.search_memory("What was my deployment target?"))
    assert rows[0]["topic"] == "deployment_target"


def test_capability_question_does_not_inherit_previous_task_frame():
    from tools.task_requirements import derive_task_frame, is_task_continuation

    previous = {"intent": "weather", "entity": "London, Ontario, Canada", "time_scope": "today"}
    assert is_task_continuation("What else can you do?", previous) is False
    assert derive_task_frame("What else can you do?", {}) == {}


def test_explicit_no_tools_constraint_blocks_all_tool_paths():
    from tools import AVAILABLE_TOOLS_MAP, TOOL_METADATA
    from tools.turn_policy import derive_turn_tool_policy

    policy = derive_turn_tool_policy(
        "Without using any tools, tell me the exact current time.",
        set(AVAILABLE_TOOLS_MAP),
        TOOL_METADATA,
    )
    assert policy.all_tools_blocked is True
    assert not policy.allowed("current_time", TOOL_METADATA["current_time"])
    assert len(policy.blocked) == len(AVAILABLE_TOOLS_MAP)


def test_slash_parser_accepts_general_whitespace_separators():
    from al_agent.slash_commands import parse_slash_command

    spec, args = parse_slash_command("/job\tdeadbeef")
    assert spec is not None and spec.name == "/job" and args == "deadbeef"
    spec, args = parse_slash_command("/research\tLondon transit reliability")
    assert spec is not None and spec.name == "/research"
    assert args == "London transit reliability"


def test_current_time_accepts_explicit_iana_timezone():
    from tools.primitives import current_time

    payload = json.loads(current_time("Asia/Tokyo"))
    assert payload["timezone"] == "Asia/Tokyo"
    assert payload["utc_offset"] in {"+0900"}


def test_time_request_uses_deterministic_pregrounding_and_skips_main_model(monkeypatch):
    from al_agent import runtime as agent
    from al_agent import turn_engine

    calls = []

    def fake_execute(name, args):
        calls.append((name, dict(args)))
        if name == "geocode_location":
            return json.dumps([{"name": "Tokyo", "timezone": "Asia/Tokyo", "latitude": 35.6, "longitude": 139.7}])
        if name == "current_time":
            assert args.get("timezone_name") == "Asia/Tokyo"
            return json.dumps({
                "utc": "2026-09-20T13:00:00+00:00",
                "local": "2026-09-20T22:00:00+09:00",
                "date": "2026-09-20",
                "time": "22:00:00",
                "day_of_week": "Sunday",
                "timezone": "Asia/Tokyo",
                "timezone_abbreviation": "JST",
                "utc_offset": "+0900",
            })
        raise AssertionError(name)

    class NoModel:
        def chat(self, **kwargs):
            raise AssertionError("main model should not be called for a simple grounded time request")

    monkeypatch.setattr(agent, "OLLAMA", NoModel())
    monkeypatch.setattr(agent, "_acquire_inference_lock", lambda: None)
    monkeypatch.setattr(agent, "_release_inference_lock", lambda lock: None)
    monkeypatch.setattr(agent, "record_monitor_state", lambda *a, **k: None)
    monkeypatch.setattr(agent, "_queue_compaction_if_needed", lambda *a, **k: None)
    monkeypatch.setattr(agent, "append_and_save", lambda messages, msg: messages.append(msg))
    monkeypatch.setattr(turn_engine, "_execute_registered_tool", fake_execute)
    monkeypatch.setattr(turn_engine, "WORKING_STATE_ENABLED", False)
    monkeypatch.setattr(turn_engine, "RECIPES_ENABLED", False)
    monkeypatch.setattr(turn_engine, "get_conversation_summary", lambda: "")
    monkeypatch.setattr(turn_engine, "build_memory_context", lambda *_: "")
    monkeypatch.setattr(turn_engine, "get_relevant_user_prompt_context", lambda *_: "")

    events = []
    messages = [{"role": "system", "content": "system"}]
    with agent.frontend_event_context(events.append):
        agent.handle_user_turn(messages, "What time is it in Tokyo?", False)

    assert [name for name, _ in calls] == ["geocode_location", "current_time"]
    assert messages[-1]["role"] == "assistant"
    assert "10:00:00 PM JST" in messages[-1]["content"]
    assert "Tokyo" in messages[-1]["content"]
    assert any(event.get("type") == "assistant_final" and event.get("deterministic") for event in events)
