import hashlib
import json
import os
from pathlib import Path

import pytest

from tools import runtime
from tools import self_optimization as optimization


def test_patch_policy_allows_small_allowlisted_diff():
    patch = """diff --git a/tools/example.py b/tools/example.py
index 1111111..2222222 100644
--- a/tools/example.py
+++ b/tools/example.py
@@ -1 +1 @@
-old = True
+old = False
"""
    assert optimization._patch_paths(patch) == {"tools/example.py"}


def test_patch_policy_rejects_runtime_data():
    patch = """diff --git a/memory/knowledge.db b/memory/knowledge.db
--- a/memory/knowledge.db
+++ b/memory/knowledge.db
@@ -1 +1 @@
-old
+new
"""
    with pytest.raises(ValueError):
        optimization._patch_paths(patch)


def test_candidate_uses_isolated_git_worktree(monkeypatch, tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "agent.py").write_text("VALUE = 1\n", encoding="utf-8")
    candidates = tmp_path / "optimizer"
    monkeypatch.setattr(optimization, "WORKSPACE_ROOT", candidates)
    monkeypatch.setattr(optimization, "source_root", lambda: source)
    candidate_dir, worktree = optimization._prepare_worktree("candidate-1")
    assert candidate_dir == candidates / "candidates" / "candidate-1"
    assert (worktree / ".git").exists()
    assert (worktree / "agent.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert source.joinpath("agent.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_generated_code_execution_fails_closed_without_sandbox(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(optimization.shutil, "which", lambda _name: None)
    result = optimization._run_sandboxed(["python", "-V"], tmp_path, 10)
    assert not result["passed"]
    assert "refusing" in result["output"]


def test_networkless_validator_executes_fixed_gate(monkeypatch, tmp_path: Path):
    if os.geteuid() != 0:
        pytest.skip("validator privilege-drop smoke test requires container root")
    from scripts import optimization_validator as validator

    candidates = tmp_path / "candidates"
    repo = candidates / "abc-123" / "repo"
    repo.mkdir(parents=True)
    (repo / "agent.py").write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setattr(validator, "CANDIDATES", candidates)
    try:
        result = validator.validate_request(
            {"request_id": "request123", "candidate_id": "abc-123", "phase": "baseline"},
            {
                "test_commands": [["python", "-c", "print('validator-ok')"]],
                "benchmark_commands": [],
                "command_timeout_seconds": 10,
                "test_memory_limit_mb": 512,
            },
        )
    except OSError as exc:
        pytest.skip(f"test sandbox does not permit UID remapping: {exc}")
    assert result["passed"]
    assert "validator-ok" in result["commands"][0]["output"]


def test_approval_requires_matching_digest(monkeypatch, tmp_path: Path):
    db = str(tmp_path / "agent.db")
    monkeypatch.setattr(runtime, "DB_PATH", db)
    monkeypatch.setattr(optimization, "WORKSPACE_ROOT", tmp_path / "optimizations")
    runtime.init_runtime_db()
    job_id = runtime.create_job("self_optimization", "demo", {})
    candidate_id = runtime.create_optimization_candidate(job_id, "demo")
    patch_path = tmp_path / "candidate.patch"
    patch_path.write_text("demo patch\n", encoding="utf-8")
    digest = hashlib.sha256(patch_path.read_bytes()).hexdigest()
    runtime.update_optimization_candidate(
        candidate_id,
        status="awaiting_approval",
        patch_path=str(patch_path),
        patch_sha256=digest,
    )
    assert "mismatch" in optimization.approve_self_optimization(candidate_id, "0" * 64).lower()
    result = json.loads(optimization.approve_self_optimization(candidate_id, digest))
    assert result["status"] == "approved"
    manifest = json.loads(Path(result["approved_patch"]).with_suffix(".json").read_text(encoding="utf-8"))
    assert manifest["status"] == "approved"
