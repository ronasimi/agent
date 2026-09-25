from __future__ import annotations

import time
from pathlib import Path


def test_runtime_facade_passes_overrides_without_mutating_turn_engine(monkeypatch):
    from al_agent import runtime

    sentinel_engine_client = object()
    sentinel_facade_client = object()
    captured = {}
    monkeypatch.setattr(runtime.turn_engine.state, "OLLAMA", sentinel_engine_client)
    monkeypatch.setattr(runtime, "OLLAMA", sentinel_facade_client)
    monkeypatch.setattr(
        runtime.turn_engine,
        "handle_user_turn",
        lambda *args, **kwargs: captured.update(kwargs),
    )

    runtime.handle_user_turn([], "hello", False)
    assert runtime.turn_engine.state.OLLAMA is sentinel_engine_client
    assert captured["runtime_overrides"]["OLLAMA"] is sentinel_facade_client


def test_filesystem_usage_rejects_host_traversal_before_disk_probe(monkeypatch, tmp_path):
    import tools.primitive_modules.system as system

    real_path = Path

    def fake_path(value):
        if str(value) == "/host":
            return tmp_path
        return real_path(value)

    monkeypatch.setattr(system, "Path", fake_path)
    result = system.filesystem_usage("../../etc", host=True)
    assert result == "Error: path escapes /host boundary."


def test_pressure_info_turns_malformed_psi_into_structured_error(monkeypatch):
    import tools.primitive_modules.system as system

    class FakeNode:
        def __init__(self, value=""):
            self.value = value

        def exists(self):
            return self.value.endswith("pressure")

        def __truediv__(self, child):
            return FakeNode(f"{self.value}/{child}")

        def read_text(self):
            return "some avg10=not-a-number total=12\n"

        def __str__(self):
            return self.value

    monkeypatch.setattr(system, "Path", FakeNode)
    result = system.pressure_info("cpu")
    assert '"error"' in result
    assert "not-a-number" in result


def test_isolated_tool_timeout_is_killable(monkeypatch, tmp_path):
    import tools.executor as executor
    from tools.subprocess_utils import ProcessResult

    monkeypatch.setattr(executor.tempfile, "TemporaryDirectory", lambda prefix="": _TempDir(tmp_path))
    monkeypatch.setattr(
        executor,
        "run_argv",
        lambda *a, **k: ProcessResult(returncode=-9, stdout="", stderr="", timed_out=True),
    )
    try:
        executor._execute_isolated_tool("demo", {}, 1)
    except TimeoutError as exc:
        assert "process group was terminated" in str(exc)
    else:
        raise AssertionError("expected isolated tool timeout")


def test_background_supervisor_heartbeats_while_job_runs(monkeypatch):
    import al_agent.background.runner as runner

    beats = []
    monkeypatch.setattr(runner, "HEARTBEAT_SECONDS", 0.05)
    monkeypatch.setattr(runner, "MONITOR_INTERVAL", 9999)
    monkeypatch.setattr(runner, "MAINTENANCE_INTERVAL", 9999)
    monkeypatch.setattr(runner, "run_job", lambda *_: time.sleep(0.18))
    monkeypatch.setattr(runner, "heartbeat_job", lambda job_id, worker_id: beats.append((job_id, worker_id)) or True)

    job = {"id": "abcdef1234", "job_type": "research", "attempts": 1, "max_attempts": 3, "title": "demo"}
    runner._run_job_supervised(job, "worker-test", last_monitor=time.monotonic(), last_maintenance=time.monotonic())
    assert beats


def test_compaction_client_uses_transport_timeout(monkeypatch):
    import al_agent.background.maintenance as maintenance

    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def generate(self, **kwargs):
            return {"response": "summary"}

    monkeypatch.setattr(maintenance, "Client", FakeClient)
    monkeypatch.setattr(maintenance, "get_job", lambda job_id: {"payload": {"through_id": 3, "conversation_id": "x"}})
    monkeypatch.setattr(maintenance, "get_messages_for_compaction", lambda *a, **k: [{"role": "user", "content": "hello"}])
    monkeypatch.setattr(maintenance, "get_conversation_summary", lambda *a, **k: "")
    monkeypatch.setattr(maintenance, "_ensure_interactive_idle", lambda: None)
    monkeypatch.setattr(maintenance, "apply_conversation_compaction", lambda *a, **k: True)
    monkeypatch.setattr(maintenance, "complete_job", lambda *a, **k: True)

    maintenance.run_context_compaction_job("job")
    assert captured["timeout"] == maintenance.COMPACTION_TIMEOUT_SECONDS


class _TempDir:
    def __init__(self, path: Path):
        self.path = path

    def __enter__(self):
        return str(self.path)

    def __exit__(self, *exc):
        return False
