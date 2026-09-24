from __future__ import annotations

import asyncio
from pathlib import Path


def test_bug_report_writes_timestamped_redacted_report_to_repository_root(tmp_path, monkeypatch):
    from webui import diagnostics

    monkeypatch.setattr(diagnostics, "ensure_conversation", lambda cid: cid or "default")
    monkeypatch.setattr(
        diagnostics,
        "_load_chat_history_from_db",
        lambda **kwargs: [{"role": "user", "content": "hello", "_created_at": "2026-09-24 00:40:00"}],
    )
    monkeypatch.setattr(diagnostics, "get_conversation_summary", lambda cid: "short rolling summary")
    monkeypatch.setattr(diagnostics, "get_compacted_through_id", lambda cid: 12)

    class FakeState:
        def __init__(self, *args, **kwargs):
            pass
        def load(self):
            return {
                "status": "blocked",
                "objective": "hello?",
                "failed_approaches": ["no usable model response"],
                "updated_at": "2026-09-24T00:40:00+00:00",
            }

    monkeypatch.setattr(diagnostics, "WorkingStateStore", FakeState)
    monkeypatch.setattr(diagnostics, "_runtime_snapshot", lambda cid: {"model": "agent-main:4b"})
    monkeypatch.setattr(diagnostics.agent_runtime, "build_system_prompt", lambda: "SYSTEM TEST PROMPT")
    monkeypatch.setattr(
        diagnostics,
        "_recent_model_traces",
        lambda cid, **kwargs: [{
            "at": "2026-09-24T00:40:01Z",
            "request": {"messages": [{"role": "system", "content": "WIRE SYSTEM"}]},
            "metrics": {"done_reason": "length"},
            "refresh_token": "secret-value",
        }],
    )
    monkeypatch.setattr(diagnostics, "list_jobs", lambda limit=30: [])
    monkeypatch.setattr(diagnostics, "list_monitor_events", lambda limit=40: [{"summary": "model no progress"}])
    monkeypatch.setattr(diagnostics, "_db_diagnostics", lambda: {"journal_mode": "wal"})
    monkeypatch.setattr(diagnostics, "_run_git", lambda *args, **kwargs: "clean")
    monkeypatch.setattr(diagnostics, "_repository_inventory", lambda root: {"root_files": ["README.md"]})

    result = diagnostics.generate_bug_report(tmp_path, "conversation-1")
    path = Path(result["path"])
    assert path.parent == tmp_path
    assert path.name.startswith("al-agent-bug-report-") and path.suffix == ".md"
    text = path.read_text(encoding="utf-8")
    assert "# Al Agent Bug Report" in text
    assert "## Triage summary" in text
    assert "SYSTEM TEST PROMPT" in text
    assert "WIRE SYSTEM" in text
    assert "2026-09-24 00:40:00" in text
    assert "model no progress" in text
    assert "no usable model response" in text
    assert "secret-value" not in text
    assert "[REDACTED]" in text


def test_bug_report_endpoint_runs_generation_off_event_loop(tmp_path, monkeypatch):
    from webui import server

    monkeypatch.setattr(server, "SOURCE_ROOT", tmp_path)
    monkeypatch.setattr(server, "ensure_conversation", lambda cid: "diag-thread")
    called = {}

    def fake_generate(source_root, cid):
        called["source_root"] = Path(source_root)
        called["cid"] = cid
        target = Path(source_root) / "al-agent-bug-report-20260924-004000Z.md"
        target.write_text("diagnostic", encoding="utf-8")
        return {"ok": True, "filename": target.name, "path": str(target)}

    monkeypatch.setattr(server, "generate_bug_report", fake_generate)
    result = asyncio.run(server.bug_report_generate("requested"))
    assert result["ok"] is True
    assert called["source_root"] == tmp_path.resolve()
    assert called["cid"] == "diag-thread"


def test_working_state_is_replaced_by_bug_report_action_in_webui():
    root = Path(__file__).resolve().parents[1]
    html = (root / "webui" / "static" / "index.html").read_text(encoding="utf-8")
    js = (root / "webui" / "static" / "app.js").read_text(encoding="utf-8")
    assert 'id="statePanel"' not in html
    assert "Working State" not in html
    assert "loadState" not in js
    assert 'id="bugReportGenerate"' in html
    assert "Generate Bug Report" in html
    assert "/api/bug-report/generate" in js


def test_repository_keeps_docs_and_diagnostics_out_of_root_and_general_scripts():
    root = Path(__file__).resolve().parents[1]
    root_markdown = sorted(path.name for path in root.glob("*.md"))
    assert root_markdown == ["README.md"]
    assert (root / "docs" / "README.md").is_file()
    assert (root / "docs" / "ARCHITECTURE.md").is_file()
    assert (root / "docs" / "CURRENT_STATE.md").is_file()
    assert (root / "docs" / "BUG_REPORTS.md").is_file()
    assert (root / "diagnostics" / "check_architecture.py").is_file()
    assert (root / "diagnostics" / "benchmarks" / "benchmark_model_roles.py").is_file()
    assert (root / "diagnostics" / "soak" / "soak_test_tools.py").is_file()
    general_scripts = {p.name for p in (root / "scripts").iterdir() if p.is_file()}
    assert not any(name.startswith("benchmark_") for name in general_scripts)
    assert "soak_test_tools.py" not in general_scripts
    assert "check_architecture.py" not in general_scripts


def test_readme_is_end_user_facing_and_contains_simple_setup():
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    for heading in ("## Features", "## Prerequisites", "## Quick start", "## Generate a bug report"):
        assert heading in readme
    assert "docker compose up -d --build" in readme
    assert "Generate Bug Report" in readme
    assert "docs/ARCHITECTURE.md" in readme


def test_webui_source_bind_is_writable_for_root_bug_report_generation():
    root = Path(__file__).resolve().parents[1]
    compose = (root / "docker-compose.yml").read_text(encoding="utf-8")
    webui = compose[compose.index("  webui:\n"):]
    assert "- AGENT_SOURCE_ROOT=/app/source" in webui
    assert "- ./:/app/source\n" in webui
    assert "- ./:/app/source:ro" not in webui
