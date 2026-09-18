import json

from tools.host_diagnostics import process_snapshot
from tools.repo_diagnostics import tool_health
from tools.system import execute_shell


def test_process_snapshot_returns_bounded_json():
    payload = json.loads(process_snapshot(limit=2))
    assert len(payload["processes"]) <= 2
    assert payload["sort_by"] == "cpu"


def test_tool_health_reports_registry():
    payload = json.loads(tool_health())
    assert payload["registered"] >= 70
    names = {item["name"] for item in payload["tools"]}
    assert "tool_health" in names
    assert "dns_diagnose" in names


def test_shell_nonzero_with_stdout_is_partial():
    result = execute_shell("printf useful; exit 1")
    assert result.startswith("Partial:")
    assert "useful" in result


def test_shell_nonzero_without_stdout_is_error():
    result = execute_shell("exit 3")
    assert result.startswith("Error:")
