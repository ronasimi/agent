import json
import threading
import time
from pathlib import Path

import pytest

from al_agent.turn_support import _parse_tool_calls, _recover_textual_readonly_tool_call
from tools.loop_validator import classify_tool_outcome
from tools.subprocess_utils import run_argv
from tools.tool_registry import _normalize_schema_arguments


def test_subprocess_capture_is_bounded_and_binary_safe():
    result = run_argv(
        ["python", "-c", "import os; os.write(1,b'\\xff'+b'x'*200000); os.write(2,b'e'*200000)"],
        timeout=5,
        max_output_bytes=8192,
    )
    assert result.returncode == 0
    assert result.stdout_truncated is True
    assert result.stderr_truncated is True
    assert "�" in result.stdout
    assert len(result.stdout) < 9000
    assert len(result.stderr) < 9000


def test_subprocess_timeout_kills_descendant_tree(tmp_path):
    marker = tmp_path / "child-survived"
    script = (
        "import subprocess,time,sys; "
        f"subprocess.Popen([sys.executable,'-c',\"import time,pathlib; time.sleep(1.2); pathlib.Path(r'{marker}').write_text('bad')\"]); "
        "time.sleep(20)"
    )
    result = run_argv(["python", "-c", script], timeout=0.3)
    assert result.timed_out is True
    time.sleep(1.5)
    assert not marker.exists()


def test_schema_validation_recurses_and_enforces_bounds():
    schema = {
        "function": {
            "parameters": {
                "type": "object",
                "properties": {
                    "timeout": {"type": "integer", "minimum": 1, "maximum": 10},
                    "options": {
                        "type": "object",
                        "properties": {"port": {"type": "integer", "minimum": 1, "maximum": 65535}},
                        "required": ["port"],
                        "additionalProperties": False,
                    },
                },
                "required": ["options"],
                "additionalProperties": False,
            }
        }
    }
    assert _normalize_schema_arguments(schema, {"timeout": "3", "options": {"port": "443"}}) == {
        "timeout": 3, "options": {"port": 443}
    }
    with pytest.raises(TypeError, match="<= 65535"):
        _normalize_schema_arguments(schema, {"options": {"port": 70000}})
    with pytest.raises(TypeError, match="unknown field"):
        _normalize_schema_arguments(schema, {"options": {"port": 443, "oops": 1}})
    with pytest.raises(TypeError, match="missing required field"):
        _normalize_schema_arguments(schema, {"options": {}})


def test_native_tool_args_accept_fenced_and_double_encoded_json():
    calls, errors = _parse_tool_calls([
        {"function": {"name": "current_time", "arguments": '''```json
{}
```'''}},
        {"function": {"name": "current_time", "arguments": json.dumps(json.dumps({}))}},
    ], {"current_time"})
    assert len(calls) == 2
    assert not errors


def test_textual_recovery_accepts_common_readonly_envelopes_but_not_mutations():
    text = '''Tool call:\n```json\n{"name":"current_time","arguments":{}}\n```'''
    calls, name = _recover_textual_readonly_tool_call(text, {"current_time"})
    assert name == "current_time"
    assert calls[0]["function"]["arguments"] == {}

    mutation = '''<tool_call>{"function":{"name":"execute_shell","arguments":{"command":"echo nope"}}}</tool_call>'''
    calls, name = _recover_textual_readonly_tool_call(mutation, {"execute_shell"})
    assert calls == [] and name == ""


def test_nonzero_execution_with_stdout_is_not_classified_as_success():
    outcome = classify_tool_outcome(
        "Partial: command exited with status 1 but produced usable stdout.\nEXIT_CODE: 1\nSTDOUT:\nprobe\nSTDERR:\npermission denied",
        tool_name="execute_shell",
    )
    assert outcome["success"] is False
    assert outcome["reason"] == "nonzero_exit"


def test_executor_blocks_duplicate_copy_of_still_running_timed_out_tool(monkeypatch):
    import tools.catalog as catalog
    import tools.executor as executor

    name = "qa_slow_timeout_tool"
    def slow():
        time.sleep(2.0)
        return "done"

    monkeypatch.setitem(catalog.AVAILABLE_TOOLS_MAP, name, slow)
    monkeypatch.setitem(catalog.TOOL_METADATA, name, {"timeout": 1, "readonly": True})
    errors = []

    def first_call():
        try:
            executor.execute_registered_tool(name, {})
        except Exception as exc:
            errors.append(str(exc))

    thread = threading.Thread(target=first_call)
    thread.start()
    thread.join(timeout=1.5)
    assert errors and "exceeded" in errors[0]
    with pytest.raises(TimeoutError, match="previous timed-out invocation"):
        executor.execute_registered_tool(name, {})


def test_malformed_missing_required_call_is_rejected_without_execution():
    calls, errors = _parse_tool_calls([
        {"function": {"name": "execute_shell", "arguments": {"timeout": 2}}}
    ], {"execute_shell"})
    assert calls == []
    assert errors and "Missing required argument" in errors[0]


def test_readonly_native_batch_executes_concurrently(monkeypatch):
    import agent
    import al_agent.turn_engine as turn_engine
    from tools import get_tool_schema

    monkeypatch.setattr(agent, "_acquire_inference_lock", lambda: None)
    monkeypatch.setattr(agent, "_release_inference_lock", lambda lock: None)
    monkeypatch.setattr(agent, "record_monitor_state", lambda *a, **k: None)
    monkeypatch.setattr(agent, "_queue_compaction_if_needed", lambda *a, **k: None)
    monkeypatch.setattr(agent, "append_and_save", lambda messages, msg: messages.append(msg))
    monkeypatch.setattr(turn_engine, "WORKING_STATE_ENABLED", False)
    monkeypatch.setattr(turn_engine, "RECIPES_ENABLED", False)
    monkeypatch.setattr(turn_engine, "get_conversation_summary", lambda: "")
    monkeypatch.setattr(turn_engine, "build_memory_context", lambda *_: "")
    monkeypatch.setattr(turn_engine, "get_relevant_user_prompt_context", lambda *_: "")
    monkeypatch.setattr(turn_engine.TaskRequirementLedger, "from_request", classmethod(lambda cls, text: cls([])))
    schemas = [get_tool_schema("hostname"), get_tool_schema("environment_summary")]
    monkeypatch.setattr(turn_engine, "select_tool_schemas", lambda *a, **k: schemas)

    def fake_execute(name, args):
        time.sleep(0.30)
        return json.dumps({"tool": name, "ok": True})
    monkeypatch.setattr(turn_engine, "_execute_registered_tool", fake_execute)

    class FakeClient:
        def __init__(self): self.calls = 0
        def chat(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return iter([{"done": True, "message": {"content": "", "tool_calls": [
                    {"id": "a", "function": {"name": "hostname", "arguments": {}}},
                    {"id": "b", "function": {"name": "environment_summary", "arguments": {}}},
                ]}}])
            return iter([{"done": True, "message": {"content": "done", "tool_calls": []}}])

    monkeypatch.setattr(agent, "OLLAMA", FakeClient())
    monkeypatch.setattr(turn_engine, "OLLAMA", agent.OLLAMA)
    started = time.monotonic()
    messages = [{"role": "system", "content": "system"}]
    agent.handle_user_turn(messages, "Run hostname and environment summary together", False)
    elapsed = time.monotonic() - started
    assert elapsed < 0.55, elapsed
    assert messages[-1]["content"] == "done"
