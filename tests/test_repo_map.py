from pathlib import Path

import pytest

from tools import repo_map


def test_repo_map_is_bounded_and_includes_symbols(monkeypatch, tmp_path: Path):
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "sample.py").write_text(
        "def useful(value: int) -> int:\n    return value + 1\n", encoding="utf-8"
    )
    (tmp_path / "memory").mkdir()
    (tmp_path / "memory" / "secret.py").write_text("TOKEN = 'no'\n", encoding="utf-8")
    monkeypatch.setattr(repo_map, "source_root", lambda: tmp_path)
    mapped = repo_map.build_repo_map(tmp_path)
    assert [item["path"] for item in mapped["files"]] == ["tools/sample.py"]
    assert mapped["files"][0]["symbols"][0]["name"] == "useful"


def test_source_path_rejects_traversal(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(repo_map, "source_root", lambda: tmp_path)
    with pytest.raises(ValueError):
        repo_map.resolve_source_path("../outside.py")
