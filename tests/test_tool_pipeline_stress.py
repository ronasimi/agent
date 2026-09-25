import json
import threading
import time
from pathlib import Path

import pytest

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






def test_nonzero_execution_with_stdout_is_not_classified_as_success():
    outcome = classify_tool_outcome(
        "Partial: command exited with status 1 but produced usable stdout.\nEXIT_CODE: 1\nSTDOUT:\nprobe\nSTDERR:\npermission denied",
        tool_name="execute_shell",
    )
    assert outcome["success"] is False
    assert outcome["reason"] == "nonzero_exit"


def test_timeout_bounded_registered_tools_use_killable_isolated_worker(monkeypatch):
    import tools.catalog as catalog
    import tools.executor as executor

    name = "qa_timeout_tool"
    monkeypatch.setitem(catalog.AVAILABLE_TOOLS_MAP, name, lambda: "in-process")
    monkeypatch.setitem(catalog.TOOL_METADATA, name, {"timeout": 7, "readonly": True})
    calls = []
    monkeypatch.setattr(
        executor,
        "_execute_isolated_tool",
        lambda tool_name, args, seconds: calls.append((tool_name, args, seconds)) or "isolated",
    )

    assert executor.execute_registered_tool(name, {}) == "isolated"
    assert calls == [(name, {}, 7)]
    assert not hasattr(executor, "_TIMED_OUT_THREADS")
