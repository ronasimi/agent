from al_agent.runtime import (
    _refresh_requirement_tool_schemas,
    _suppress_completed_requirement_calls,
)
from tools import TOOL_METADATA, get_tool_schema
from tools.loop_validator import tool_call_signature
from tools.task_requirements import TaskRequirementLedger
from tools.turn_policy import derive_turn_tool_policy


def _call(name: str, arguments=None):
    return {
        "id": f"call-{name}",
        "type": "function",
        "function": {"name": name, "arguments": arguments or {}},
    }


def test_satisfied_requirement_schema_is_pruned_while_pending_tool_remains():
    ledger = TaskRequirementLedger.from_request(
        "Inspect host CPU and memory and CPU memory IO pressure"
    )
    policy = derive_turn_tool_policy("Inspect host CPU and memory and pressure", set(TOOL_METADATA), TOOL_METADATA)
    schemas = [get_tool_schema("host_snapshot"), get_tool_schema("pressure_snapshot")]
    ledger.record_tool("host_snapshot", status="ok", fingerprint="host-one")

    changed = _refresh_requirement_tool_schemas(schemas, ledger, policy)
    names = [schema["function"]["name"] for schema in schemas]
    assert changed is True
    assert "host_snapshot" not in names
    assert "pressure_snapshot" in names


def test_successful_completed_readonly_repeat_is_suppressed_when_work_remains():
    ledger = TaskRequirementLedger.from_request(
        "Inspect host CPU and memory and CPU memory IO pressure"
    )
    ledger.record_tool("host_snapshot", status="ok", fingerprint="host-one")
    repeated = _call("host_snapshot")
    calls, notes = _suppress_completed_requirement_calls(
        [repeated], ledger, {tool_call_signature(repeated)}, "Inspect host CPU and memory and pressure"
    )
    assert calls == []
    assert any("redundant completed requirement" in note for note in notes)


def test_successful_completed_readonly_repeat_is_suppressed_after_all_work_finishes():
    ledger = TaskRequirementLedger.from_request("What time is it?")
    ledger.record_tool("current_time", status="ok", fingerprint="time-one")
    repeated = _call("current_time")
    calls, notes = _suppress_completed_requirement_calls(
        [repeated], ledger, {tool_call_signature(repeated)}, "What time is it?"
    )
    assert calls == []
    assert any("redundant completed requirement" in note for note in notes)


def test_fallback_finalizer_preserves_latest_media(monkeypatch):
    from al_agent import runtime as agent

    captured = {}

    class FakeClient:
        def chat(self, **kwargs):
            captured["messages"] = kwargs["messages"]
            return {"message": {"content": "final"}}

    monkeypatch.setattr(agent, "OLLAMA", FakeClient())
    monkeypatch.setattr(agent, "append_and_save", lambda messages, message: messages.append(message))
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "inspect screenshot"},
    ]
    tail = [
        {"role": "user", "content": "actual media", "images": ["base64-image"]},
    ]
    agent._finalize_after_limit(messages, tail)
    assert any(message.get("images") == ["base64-image"] for message in captured["messages"])
