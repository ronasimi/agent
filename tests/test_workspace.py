import os
from pathlib import Path


def test_absolute_workspace_path_not_duplicated(tmp_path, monkeypatch):
    import tools.workspace as workspace

    monkeypatch.setattr(workspace, "WORKSPACE_DIR", os.path.realpath(tmp_path))
    target = Path(tmp_path) / "nested" / "example.txt"
    target.parent.mkdir(parents=True)
    target.write_text("ok", encoding="utf-8")

    assert workspace._get_safe_path(str(target)) == os.path.realpath(target)
    assert workspace.read_file(str(target)) == "ok"


def test_relative_workspace_path_resolves_inside_root(tmp_path, monkeypatch):
    import tools.workspace as workspace

    monkeypatch.setattr(workspace, "WORKSPACE_DIR", os.path.realpath(tmp_path))
    target = Path(tmp_path) / "example.txt"
    target.write_text("ok", encoding="utf-8")

    assert workspace.read_file("example.txt") == "ok"


def test_absolute_outside_path_is_blocked(tmp_path, monkeypatch):
    import tools.workspace as workspace

    root = tmp_path / "workspace"
    outside = tmp_path / "outside.txt"
    root.mkdir()
    outside.write_text("secret", encoding="utf-8")
    monkeypatch.setattr(workspace, "WORKSPACE_DIR", os.path.realpath(root))

    result = workspace.read_file(str(outside))
    assert "Path outside workspace blocked" in result
