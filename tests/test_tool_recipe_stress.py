import json


def _prompt() -> str:
    sections = [
        ("A. DIRECT TOOL ROUTING", range(1, 7)),
        ("B. SIMILAR-TOOL DISAMBIGUATION", range(7, 12)),
        ("C. TOOL DISCOVERY / RECOVERY PATH", range(12, 15)),
        ("D. WORKSPACE PATH AND FILE-TOOL TEST", range(15, 18)),
        ("E. RECIPE DISCOVERY AND DUPLICATION CHECK", range(18, 20)),
        ("F. RECIPE CREATION", range(20, 22)),
        ("G. RECIPE REPLAY", range(22, 25)),
        ("H. RECIPE FAILURE / FALLBACK BEHAVIOR", range(25, 28)),
        ("I. OBSERVATION / TRUNCATION PATH", range(28, 30)),
        ("J. RECIPE STORAGE INTEGRITY", range(30, 33)),
        ("K. CLEANUP", range(33, 35)),
        ("L. REQUIREMENT / EVIDENCE AUDIT", range(35, 38)),
    ]
    lines = [
        "I want you to perform a tool-routing and recipe-system stress test.",
        "Treat every numbered item below as an independent requirement.",
        "GENERAL SAFETY RULES",
        "1. Do not modify system configuration.",
        "2. Do not install packages.",
    ]
    for heading, numbers in sections:
        lines += ["==================================================", heading, "=================================================="]
        for number in numbers:
            lines.append(f"{number}. Requirement {number} for deterministic tool/recipe validation.")
    lines += ["FINAL OUTPUT", "Return the requested report."]
    return "\n".join(lines)


def test_tool_recipe_stress_compiles_all_37_requirements():
    from tools.task_requirements import derive_requirements

    rows = derive_requirements(_prompt())
    assert len(rows) == 37
    assert [row.key for row in rows] == [f"tooltest:{i:02d}" for i in range(1, 38)]
    assert rows[0].tool == "current_time"
    assert rows[6].tool == "dns_query"
    assert rows[14].tool == "write_file"
    assert rows[17].tool == "search_recipes"
    assert rows[18].tool == "load_recipe"
    assert rows[19].tool == "save_recipe"
    assert rows[20].tool == "search_recipes"
    assert rows[21].tool == "run_recipe"
    assert rows[29].tool == "load_recipe"
    assert rows[32].tool == "remove_path"
    assert all("Do not install packages" not in str(row.scope.get("source_text") or "") for row in rows)


def test_key_scoped_requirement_recording_keeps_repeated_phase_pending():
    from tools.task_requirements import Requirement, TaskRequirementLedger

    ledger = TaskRequirementLedger([
        Requirement("tooltest:18", "search_recipes", "pre-search"),
        Requirement("tooltest:21", "search_recipes", "post-search"),
    ])
    ledger.record_tool_for_key(
        "tooltest:18", "search_recipes", status="ok", reason="ok",
        arguments={"query": "health"}, result_text="[]",
    )
    assert ledger.requirements[0].status == "satisfied"
    assert ledger.requirements[1].status == "pending"


def test_workspace_remove_path_is_workspace_bounded(tmp_path, monkeypatch):
    import tools.workspace as ws

    root = tmp_path / "workspace"
    root.mkdir()
    monkeypatch.setattr(ws, "WORKSPACE_DIR", str(root))
    nested = root / "harness_tool_recipe_test"
    nested.mkdir()
    (nested / "input.txt").write_text("x", encoding="utf-8")
    assert ws.remove_path("harness_tool_recipe_test", recursive=True).startswith("Successfully removed")
    assert not nested.exists()
    assert ws.remove_path("../outside", recursive=True).startswith("Error:")


def test_load_recipe_tool_exposes_pipeline(tmp_path, monkeypatch):
    import tools.recipe_store as store
    from tools.recipe_store import save_recipe
    from tools.pipeline import load_recipe

    db = tmp_path / "recipes.db"
    monkeypatch.setenv("AGENT_RECIPE_DB", str(db))
    save_recipe("r", "demo", [{"id": "s1", "tool": "calculate", "args": {"expression": "1+1"}}])
    payload = json.loads(load_recipe("r"))
    assert payload["name"] == "r"
    assert payload["pipeline"][0]["tool"] == "calculate"


def test_tool_recipe_stress_finishes_without_model_loop(monkeypatch):
    from al_agent import turn_engine as te

    prompt = _prompt()
    recipe_saved = {"value": False}
    recipe_definition = {
        "id": 1,
        "name": "quick_local_agent_health_check",
        "description": "Quick local agent health check using current time, host resources, CPU identity, and Ollama runtime state.",
        "pipeline": [
            {"id": "time", "tool": "current_time", "args": {}},
            {"id": "host", "tool": "host_snapshot", "args": {}},
            {"id": "cpu", "tool": "cpu_info", "args": {}},
            {"id": "ollama", "tool": "ollama_runtime_snapshot", "args": {}},
            {"id": "ollama_count", "tool": "json_count", "args": {"data": {"$ref": "ollama", "path": "models", "default": []}}},
            {"id": "summary", "tool": "compose_object", "args": {"data": {}}},
        ],
        "parameters": {}, "tags": ["health"], "origin": "user", "target_tool": "",
    }
    calls = []

    def fake_execute(name, args):
        calls.append((name, dict(args)))
        if name == "current_time":
            return json.dumps({
                "utc": "2026-09-23T17:00:00+00:00", "local": "2026-09-23T13:00:00-04:00",
                "date": "2026-09-23", "time": "13:00:00", "timezone": "America/Toronto", "utc_offset": "-0400",
            })
        if name == "environment_summary":
            return json.dumps({"host_hostname": "muninn", "platform": "Linux-test", "kernel": "test", "architecture": "x86_64"})
        if name == "cpu_info":
            return json.dumps({"models": ["AMD Ryzen 5 PRO 4650U with Radeon Graphics"], "physical_cores": 6, "logical_cpus": 12})
        if name == "host_snapshot":
            return json.dumps({"hostname": "muninn", "uptime_seconds": 1234, "load_average": [0.2, 0.1, 0.1],
                               "memory": {"total_mb": 15200, "available_mb": 7000}, "disk": {"used_percent": 80.0}})
        if name == "temperature_sensors":
            return json.dumps({"k10temp": [{"label": "Tctl", "current": 50.0}]})
        if name == "ollama_runtime_snapshot":
            return json.dumps({"models": [{"name": "agent-main:4b"}, {"name": "agent-main:2b"}]})
        if name == "dns_query":
            return json.dumps({"ok": True, "status": "NOERROR", "answers": ["example.com. A 93.184.216.34"], "elapsed_ms": 2.0})
        if name == "tcp_connect":
            return json.dumps({"ok": True, "connected_address": "93.184.216.34", "tcp_connect_ms": 15.0})
        if name == "http_probe":
            return json.dumps({"ok": True, "http_status": 200, "time_to_headers_ms": 25.0, "tls_version": "TLSv1.3"})
        if name == "page_metadata":
            return json.dumps({"url": "https://example.com", "canonical": "https://example.com/", "http_status": 200, "title": "Example Domain"})
        if name == "tool_search":
            query = str(args.get("query") or "")
            if "skills" in query:
                return json.dumps([{"name": "search_skills", "readonly": True, "required": ["query"]}])
            if "observation" in query:
                return json.dumps([{"name": "read_observation", "readonly": True, "required": ["observation_id"]}])
            if "recipe" in query:
                return json.dumps([
                    {"name": name, "readonly": name != "save_recipe", "required": []}
                    for name in ("search_recipes", "list_recipes", "load_recipe", "save_recipe", "run_recipe")
                ])
            return json.dumps([])
        if name == "write_file":
            return "Successfully wrote 35 characters to harness_tool_recipe_test/input.txt"
        if name == "read_file":
            return "TOOL_PATH_TEST_OK\nalpha\nbeta\ngamma\n"
        if name == "search_recipes":
            return json.dumps([
                {"id": 1, "name": "quick_local_agent_health_check", "description": recipe_definition["description"],
                 "tags": ["health"], "origin": "user", "target_tool": "", "semantic_score": 0.9}
            ] if recipe_saved["value"] else [])
        if name == "save_recipe":
            recipe_saved["value"] = True
            return json.dumps({"saved": True, "id": 1, "name": "quick_local_agent_health_check"})
        if name == "load_recipe":
            return json.dumps(recipe_definition)
        if name == "run_recipe":
            return json.dumps({
                "ok": True,
                "result": {
                    "local_time": "2026-09-23T13:00:01-04:00", "hostname": "muninn", "uptime_seconds": 1235,
                    "memory_total_mb": 15200, "memory_available_mb": 6990, "load_average": [0.2, 0.1, 0.1],
                    "cpu_model": "AMD Ryzen 5 PRO 4650U with Radeon Graphics", "logical_cpus": 12,
                    "ollama_state": {"models": [{"name": "agent-main:4b"}, {"name": "agent-main:2b"}]},
                    "loaded_model_count": 2,
                },
                "recipe": {"id": 1, "name": "quick_local_agent_health_check"},
            })
        if name == "remove_path":
            return "Successfully removed harness_tool_recipe_test"
        raise AssertionError((name, args))

    class NoModel:
        def chat(self, **kwargs):
            raise AssertionError("main model must not be used for deterministic tool/recipe stress plan")

    monkeypatch.setattr(te, "_execute_registered_tool", fake_execute)
    monkeypatch.setattr(te, "WORKING_STATE_ENABLED", False)
    monkeypatch.setattr(te, "RECIPES_ENABLED", False)
    monkeypatch.setattr(te, "LOOP_VALIDATOR_ENABLED", False)
    monkeypatch.setattr(te, "get_conversation_summary", lambda: "")
    monkeypatch.setattr(te, "build_memory_context", lambda *_: "")
    monkeypatch.setattr(te, "get_relevant_user_prompt_context", lambda *_: "")
    monkeypatch.setattr(te, "get_user_location", lambda: "London, Ontario, Canada")
    monkeypatch.setattr(te, "_bounded_tool_result_with_ref", lambda _name, text: (text, ""))
    monkeypatch.setattr(te, "evict_report_model_for_interactive", lambda: None)
    monkeypatch.setattr(te, "_prune_compacted_history", lambda _messages: None)
    monkeypatch.setattr(te, "log_perf_stats", lambda *a, **k: None)

    messages = [{"role": "system", "content": "system"}]
    te.handle_user_turn(
        messages, prompt, False,
        runtime_overrides={
            "OLLAMA": NoModel(), "record_monitor_state": lambda *a, **k: None,
            "append_and_save": lambda rows, item: rows.append(item),
            "acquire_turn_lock": lambda: object(), "release_turn_lock": lambda _lock: None,
            "acquire_inference_lock": lambda: (_ for _ in ()).throw(AssertionError("model lock should not be acquired")),
            "release_inference_lock": lambda _lock: None, "queue_compaction_if_needed": lambda *a, **k: None,
        },
    )

    content = messages[-1]["content"]
    for heading in (
        "## Direct tool routing", "## Tool disambiguation", "## Tool discovery",
        "## Workspace file path", "## Recipe discovery", "## Recipe creation",
        "## Recipe replay", "## Routing/fallback audit", "## Observation audit",
        "## Recipe integrity", "## Unresolved requirements",
    ):
        assert heading in content
    assert "Deterministic finalization — PASS" in content
    assert "## Unresolved requirements\nNone." in content
    assert "hard turn/model-call budget exhausted" not in content
    assert recipe_saved["value"] is True
    assert sum(1 for name, _ in calls if name == "save_recipe") == 1
    assert sum(1 for name, _ in calls if name == "run_recipe") == 1
    assert sum(1 for name, _ in calls if name == "tool_search") == 3
    assert calls[-1][0] == "remove_path"
