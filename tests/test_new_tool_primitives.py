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


def test_text_readers_reject_binary_images(tmp_path, monkeypatch):
    from tools import workspace
    from tools.primitive_modules import filesystem

    monkeypatch.setattr(workspace, "WORKSPACE_DIR", str(tmp_path.resolve()))
    image = tmp_path / "pixel.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"\x00" * 32)

    legacy = workspace.read_file(str(image))
    primitive = filesystem.read_text(str(image))
    assert legacy.startswith("Error: read_file only supports text files")
    assert primitive.startswith("Error: read_text only supports text files")
    assert "image/png" in legacy
    assert "image/png" in primitive
