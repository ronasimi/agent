from __future__ import annotations

import json
from pathlib import Path


def test_qwen_schema_adapter_orders_required_and_drops_defaults():
    from tools.tool_registry import adapt_schema_for_qwen

    schema = {
        "type": "function",
        "function": {
            "name": "demo",
            "description": "  A   demo tool.  ",
            "parameters": {
                "type": "object",
                "properties": {
                    "optional": {"type": "string", "default": "x", "description": " optional   value "},
                    "required_arg": {"type": "integer", "description": " required value "},
                },
                "required": ["required_arg"],
            },
        },
    }
    adapted = adapt_schema_for_qwen(schema)
    props = adapted["function"]["parameters"]["properties"]
    assert list(props) == ["required_arg", "optional"]
    assert "default" not in props["optional"]
    assert adapted["function"]["description"] == "A demo tool."
    assert adapted["function"]["parameters"]["additionalProperties"] is False


def test_tool_search_finds_catalog_capabilities():
    from tools.tool_discovery import tool_search

    rows = json.loads(tool_search("calendar events", limit=5))
    assert any(row["name"] == "google_calendar_list_events" for row in rows)
    assert all("description" in row and "required" in row for row in rows)


def test_failure_lessons_strengthen_after_recovery(tmp_path, monkeypatch):
    from tools import failure_lessons as fl

    monkeypatch.setattr(fl, "DB_PATH", str(tmp_path / "knowledge.db"))
    lesson_id = fl.record_failure("read_file", {"filename": "/app/workspace/app/tools.py"}, "tool_error", "Error: no such file")
    assert lesson_id
    fl.record_recovery(lesson_id, {"filename": "/app/workspace/app/tools.py"}, {"filename": "tools.py"}, "read_file")
    rows = fl.relevant_failure_lessons("read tools.py", {"read_file"})
    assert rows
    assert "Recovery" in rows[0]["fix"]


def test_reflection_notes_are_trigger_scoped(tmp_path, monkeypatch):
    from tools import reflection

    monkeypatch.setattr(reflection, "DB_PATH", str(tmp_path / "knowledge.db"))
    assert reflection.store_reflection_notes("c1", [{"trigger_terms": ["workspace path"], "note": "Use workspace-relative paths after the harness has already rooted the request."}]) == 1
    assert "workspace-relative" in reflection.render_relevant_reflections("c1", "fix this workspace path", limit=2)
    assert reflection.render_relevant_reflections("c1", "tell me a joke", limit=2) == ""


def test_persistent_goal_round_trip(tmp_path, monkeypatch):
    from tools import goals

    monkeypatch.setattr(goals, "DB_PATH", str(tmp_path / "knowledge.db"))
    monkeypatch.setattr(goals, "get_active_conversation_id", lambda: "conv-a")
    result = json.loads(goals.set_goal("Ship the harness", "All regression tests pass"))
    assert result["goal"] == "Ship the harness"
    assert json.loads(goals.get_goal())["definition_of_done"] == "All regression tests pass"
    assert json.loads(goals.clear_goal())["cleared"] is True
    assert json.loads(goals.get_goal()) == {}


def test_targeted_extraction_prefers_relevant_source_passages():
    from tools.extraction import targeted_extract

    text = ("Navigation and unrelated boilerplate. " * 50) + "\n\nBrent crude is 98.40 dollars per barrel. WTI is 94.10 dollars per barrel.\n\n" + ("Sports unrelated text. " * 50)
    out = targeted_extract(text, "Brent and WTI prices", max_chars=600)
    assert "Brent crude" in out
    assert "WTI" in out
    assert len(out) <= 600


def test_sensitive_file_policy_requires_explicit_user_intent():
    from tools.security import arguments_reference_sensitive_path, redact_secrets, user_explicitly_requested_sensitive_access

    args = {"filename": ".env"}
    assert arguments_reference_sensitive_path(args)
    assert not user_explicitly_requested_sensitive_access("inspect the project", args)
    assert user_explicitly_requested_sensitive_access("read the .env file", args)
    assert "supersecret" not in redact_secrets("API_KEY=supersecret")


def test_workspace_write_is_atomic(tmp_path, monkeypatch):
    from tools import workspace

    monkeypatch.setattr(workspace, "WORKSPACE_DIR", str(tmp_path.resolve()))
    assert "Successfully wrote" in workspace.write_file("nested/a.txt", "first")
    assert "Successfully wrote" in workspace.write_file("nested/a.txt", "second")
    assert (tmp_path / "nested" / "a.txt").read_text() == "second"
    assert not list((tmp_path / "nested").glob(".agent-write-*"))


def test_model_trace_writer_appends_jsonl(tmp_path):
    from al_agent.model_traces import record_model_trace

    path = tmp_path / "traces.jsonl"
    record_model_trace(
        path=str(path), enabled=True, max_bytes=1024 * 1024,
        conversation_id="c", turn_id=1, call_index=1, model="agent-main:2b",
        role="main", purpose="tool_selection", thinking_enabled=False,
        messages=[{"role": "user", "content": "hi"}], tools=[], options={"num_ctx": 16384},
        completion={"content": "hello"}, metrics={"eval_count": 1},
    )
    row = json.loads(path.read_text().strip())
    assert row["request"]["messages"][0]["content"] == "hi"
    assert row["completion"]["content"] == "hello"
