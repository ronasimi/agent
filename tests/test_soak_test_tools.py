from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from scripts import soak_test_tools as soak


def test_parse_duration_units():
    assert soak.parse_duration("30") == 30
    assert soak.parse_duration("2m") == 120
    assert soak.parse_duration("1.5h") == 5400
    assert soak.parse_duration("1d") == 86400


def test_probe_argument_contract_covers_all_registered_tools_and_builtin_recipes(tmp_path):
    fixtures = soak.create_fixtures(tmp_path / "fixtures")
    ids = {"job_id": "audit-job", "observation_old": "old", "observation_new": "new"}
    targets, inventory = soak.discover_targets(saved_recipes=[])
    assert inventory["registered_tools"] >= 220
    assert inventory["builtin_recipes"] >= 20
    errors = []
    for target in targets:
        ok, reason, _args = soak.contract_check(target, fixtures, ids)
        if not ok:
            errors.append((target.kind, target.name, reason))
    assert errors == []


def test_mutating_default_isolation_policy():
    isolated = soak.Target(kind="tool", name="write_file", readonly=False)
    unsafe = soak.Target(kind="tool", name="install_package", readonly=False)
    readonly = soak.Target(kind="tool", name="calculate", readonly=True)
    assert soak._mutation_action(readonly, "isolated") == "invoke"
    assert soak._mutation_action(isolated, "isolated") == "invoke"
    assert soak._mutation_action(unsafe, "isolated") == "contract_only"
    assert soak._mutation_action(unsafe, "all") == "invoke"


def test_summary_tracks_flaky_errors_and_latency():
    rows = [
        {"kind":"primitive", "name":"calculate", "module":"primitive_modules.transform", "status":"success", "reason":"ok", "latency_ms":10},
        {"kind":"primitive", "name":"calculate", "module":"primitive_modules.transform", "status":"error", "reason":"tool_reported_error", "latency_ms":30, "error_class":"runtime"},
        {"kind":"recipe_builtin", "name":"compat.test", "module":"recipe_compat", "status":"success", "reason":"recipe_ok", "latency_ms":20},
    ]
    summary = soak.summarize(rows, {"registered_tools": 223}, "2026-09-21T00:00:00+00:00", 60)
    assert summary["totals"]["attempted"] == 3
    assert summary["totals"]["success"] == 2
    assert summary["totals"]["error"] == 1
    assert summary["totals"]["latency_ms"]["p50"] == 20
    assert summary["flaky_targets"][0]["name"] == "calculate"


def test_read_saved_recipes_direct_does_not_require_tool_import(tmp_path):
    database = tmp_path / "recipes.db"
    with sqlite3.connect(database) as conn:
        conn.execute("CREATE TABLE recipes(id INTEGER PRIMARY KEY, name TEXT, description TEXT, pipeline_json TEXT, parameters_json TEXT, tags_json TEXT, origin TEXT, target_tool TEXT)")
        conn.execute(
            "INSERT INTO recipes(name,description,pipeline_json,parameters_json,tags_json,origin,target_tool) VALUES(?,?,?,?,?,?,?)",
            ("User audit", "test", json.dumps([{"tool":"calculate","args":{"expression":"1+1"}}]), "{}", "[]", "user", ""),
        )
        conn.execute(
            "INSERT INTO recipes(name,description,pipeline_json,parameters_json,tags_json,origin,target_tool) VALUES(?,?,?,?,?,?,?)",
            ("Builtin", "test", "[]", "{}", "[]", "builtin", "calculate"),
        )
        conn.commit()
    rows = soak.read_saved_recipes_direct(database)
    assert [row["name"] for row in rows] == ["User audit"]



def test_all_declared_mutators_have_an_explicit_soak_policy():
    from tools.providers import MUTATING_TOOLS
    covered = soak.ISOLATABLE_MUTATORS | soak.NONISOLATABLE_MUTATORS
    assert covered == set(MUTATING_TOOLS)
    assert soak.ISOLATABLE_MUTATORS.isdisjoint(soak.NONISOLATABLE_MUTATORS)


def test_redaction_strips_url_query_values():
    value = soak._redact({"url": "https://example.com/path?token=secret&x=1"})
    assert value["url"] == "https://example.com/path?<redacted-query>"

def test_child_probe_executes_local_primitive(tmp_path):
    report = tmp_path / "report"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    fixtures = soak.create_fixtures(workspace / "fixtures")
    env = soak.build_env(report, workspace)
    target = soak.Target(kind="primitive", name="calculate", module="primitive_modules.transform", readonly=True)
    # Fetch the real schema so argument normalization is identical to production.
    targets, _ = soak.discover_targets(saved_recipes=[])
    target = next(item for item in targets if item.name == "calculate")
    result = soak.invoke_child(target, {"expression":"6*7"}, fixtures, {}, env, timeout=10)
    assert result["status"] == "success"
    assert result["latency_ms"] >= 0


def test_summary_coverage_uses_selected_inventory_denominator():
    rows = [
        {"kind":"primitive", "name":"calculate", "module":"primitive_modules.transform", "status":"success", "reason":"ok", "latency_ms":5},
    ]
    summary = soak.summarize(rows, {"selected_targets": 10}, "2026-09-21T00:00:00+00:00", 1)
    assert summary["totals"]["observed_targets"] == 1
    assert summary["totals"]["attempted_targets"] == 1
    assert summary["totals"]["unprobed_targets"] == 9
    assert summary["totals"]["runtime_target_coverage"] == 0.1
    assert summary["totals"]["contract_target_coverage"] == 0.1


def test_summary_reports_environment_blocks_separately():
    rows = [
        {"kind":"primitive", "name":"calculate", "status":"success", "reason":"ok", "latency_ms":5},
        {"kind":"tool", "name":"market_quote", "status":"error", "reason":"exception", "latency_ms":10, "error_class":"network"},
    ]
    summary = soak.summarize(rows, {"selected_targets": 2}, "2026-09-21T00:00:00+00:00", 1)
    assert summary["totals"]["success_rate"] == 0.5
    assert summary["totals"]["environment_blocked"] == 1
    assert summary["totals"]["harness_evaluable"] == 1
    assert summary["totals"]["harness_success_rate"] == 1.0


def test_streaming_metrics_matches_batch_summary():
    rows = [
        {"kind":"primitive", "name":"calculate", "status":"success", "reason":"ok", "latency_ms":4},
        {"kind":"primitive", "name":"calculate", "status":"timeout", "reason":"controller_timeout", "latency_ms":20, "error_class":"timeout"},
        {"kind":"recipe_builtin", "name":"compat.test", "status":"partial", "reason":"recipe_partial", "latency_ms":8},
    ]
    inventory = {"selected_targets": 2}
    batch = soak.summarize(rows, inventory, "2026-09-21T00:00:00+00:00", 10)
    metrics = soak.MetricsAccumulator()
    for row in rows:
        metrics.add(row)
    streamed = metrics.to_summary(inventory, "2026-09-21T00:00:00+00:00", 10)
    assert streamed["totals"] == batch["totals"]
    assert streamed["by_kind"] == batch["by_kind"]
    assert streamed["failure_reasons"] == batch["failure_reasons"]
    assert streamed["targets"] == batch["targets"]


def test_network_classifier_recognizes_requests_dns_failures():
    assert soak._classify_environment(
        "exception",
        "Max retries exceeded with url: /quote (Caused by NameResolutionError: failed to resolve host)",
    ) == "network"


def test_declared_distribution_names_parses_requirements_specs(tmp_path):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text(
        "pyyaml\nollama>=0.5.2\ncryptography>=43,<47  # bounded\n\n# comment\n",
        encoding="utf-8",
    )
    assert soak._declared_distribution_names(requirements) == ["pyyaml", "ollama", "cryptography"]


def test_dependency_preflight_message_is_actionable():
    message = soak._dependency_preflight_message(["pyyaml", "ollama"])
    assert "pyyaml" in message
    assert "ollama" in message
    assert "bootstrap_venv.sh" in message
    assert ".venv/bin/python" in message


def test_controller_fails_cleanly_before_tool_import_when_python_deps_missing(monkeypatch, capsys):
    monkeypatch.setattr(soak, "_preflight_repository_dependencies", lambda: (False, ["pyyaml"]))
    args = soak.parse_args(["--duration", "1s", "--passes", "1"])
    assert soak.controller_main(args) == 2
    captured = capsys.readouterr()
    assert "Missing Python package(s): pyyaml" in captured.err
    assert "Traceback" not in captured.err


def test_maybe_reexec_uses_repo_venv_when_available(tmp_path, monkeypatch):
    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("", encoding="utf-8")
    monkeypatch.setattr(soak, "VENV_PYTHON", venv_python)
    monkeypatch.delenv("AGENT_VENV_REEXEC", raising=False)

    called = {}

    class ReexecCalled(RuntimeError):
        pass

    def fake_execve(path, argv, env):
        called.update(path=path, argv=argv, env=env)
        raise ReexecCalled

    monkeypatch.setattr(soak.os, "execve", fake_execve)
    try:
        soak._maybe_reexec_in_repo_venv(["--duration", "1h"])
    except ReexecCalled:
        pass
    else:
        raise AssertionError("expected venv re-exec")

    assert called["path"] == str(venv_python.resolve())
    assert called["argv"][0] == str(venv_python.resolve())
    assert called["argv"][-2:] == ["--duration", "1h"]
    assert called["env"]["AGENT_VENV_REEXEC"] == "1"


def test_default_run_is_five_passes_without_duration_limit():
    args = soak.parse_args([])
    assert args.passes == 5
    assert args.duration == 0.0


def test_pass_fixture_profiles_rotate_deterministically(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    first = soak.create_fixtures(workspace / "p1", workspace, pass_no=1)
    second = soak.create_fixtures(workspace / "p2", workspace, pass_no=2)
    sixth = soak.create_fixtures(workspace / "p6", workspace, pass_no=6)
    assert first["profile"] != second["profile"]
    assert first["expression"] != second["expression"]
    assert first["endpoint_tls"] is False and second["endpoint_tls"] is True
    assert sixth["profile"] == first["profile"]


def test_process_fixture_uses_probeable_same_uid_process(tmp_path):
    from tools.primitive_modules import process as process_ops
    proc = soak._start_process_fixture(tmp_path, 1)
    try:
        assert not process_ops.process_info(proc.pid).startswith("Error:")
        assert not process_ops.process_io(proc.pid).startswith("Error:")
        assert not process_ops.process_fds(proc.pid, 10).startswith("Error:")
    finally:
        soak._stop_process_fixture(proc)
