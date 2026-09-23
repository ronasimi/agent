import json


def _prompt() -> str:
    sections = [
        ("A. LEDGER AND SYSTEM BASELINE", range(16, 21)),
        ("B. TOOL LAYER SEPARATION", range(21, 26)),
        ("C. CAPABILITY DISCOVERY AND PROVENANCE", range(26, 30)),
        ("D. DISPOSABLE WORKSPACE TEST", range(30, 33)),
        ("E. SEMANTIC RECIPE SEARCH", range(33, 36)),
        ("F. GENERALIZED RECIPE CREATION", range(36, 39)),
        ("G. FIRST PARAMETERIZED REPLAY", range(39, 43)),
        ("H. SECOND PARAMETERIZED REPLAY", range(43, 47)),
        ("I. GENERALIZATION AUDIT", range(47, 51)),
        ("J. ROUTING AND DISCOVERY AUDIT", range(51, 55)),
        ("K. OBSERVATION AND FAILURE AUDIT", range(55, 59)),
        ("L. WORKSPACE CLEANUP", range(59, 63)),
        ("M. PERSISTED LEDGER AUDIT", range(63, 68)),
        ("N. FINAL AUDITS", range(68, 73)),
    ]
    lines = [
        "I want a generalized recipe, parameterization, provenance, and requirement-ledger stress test.",
        "Treat every numbered item as an independent requirement.",
        "GENERAL RULES",
    ]
    for n in range(1, 16):
        lines.append(f"{n}. General safety and deterministic execution rule {n}.")
    for heading, numbers in sections:
        lines += ["==================================================", heading, "=================================================="]
        for n in numbers:
            text = f"Requirement {n}."
            if n == 53:
                text = "Verify tool_search provenance is persisted."
            lines.append(f"{n}. {text}")
    lines += ["FINAL OUTPUT", "Return the requested deterministic report."]
    return "\n".join(lines)


def test_generalized_recipe_stress_compiles_72_requirements():
    from tools.task_requirements import derive_requirements

    rows = derive_requirements(_prompt())
    assert len(rows) == 72
    assert [row.key for row in rows] == [f"genrecipe:{i:02d}" for i in range(1, 73)]
    assert rows[0].scope["policy_rule"] is True
    assert rows[15].tool == "current_time"
    assert rows[20].tool == "dns_query"
    assert rows[29].tool == "write_file"
    assert rows[35].tool == "save_recipe"
    assert rows[38].tool == "run_recipe"
    assert rows[42].tool == "run_recipe"
    assert rows[58].tool == "remove_path"
    assert rows[59].tool == "path_stat"


def test_pipeline_template_resolves_runtime_parameter():
    from tools.pipeline import _resolve

    resolved = _resolve(
        {"$template": "https://{hostname}", "vars": {"hostname": {"$param": "hostname"}}},
        {}, {"hostname": "www.iana.org"},
    )
    assert resolved == "https://www.iana.org"


def test_generalized_recipe_stress_finishes_without_model_loop(monkeypatch):
    from al_agent import turn_engine as te

    prompt = _prompt()
    calls = []
    saved = {"value": False, "definition": None}
    late_truncation = {"cleanup_seen": False, "armed": True, "raw_len": 0}

    def definition():
        return saved["definition"] or {}

    def candidate():
        d = definition()
        return {
            "id": 77, "name": "public_endpoint_health_check",
            "description": d.get("description", "public endpoint health check"),
            "tags": ["network", "endpoint", "dns", "tcp", "https", "metadata", "hostname"],
            "origin": "user", "target_tool": "", "semantic_score": 0.9,
        }

    def fake_execute(name, args):
        calls.append((name, dict(args)))
        if name == "current_time":
            return json.dumps({"utc": "2026-09-23T18:00:00+00:00", "local": "2026-09-23T14:00:00-04:00", "date": "2026-09-23", "time": "14:00:00", "timezone": "America/Toronto", "utc_offset": "-0400"})
        if name == "environment_summary":
            return json.dumps({"host_hostname": "muninn", "kernel": "7.2.6-test", "platform": "Linux-test", "architecture": "x86_64"})
        if name == "cpu_info":
            return json.dumps({"models": ["AMD Ryzen 5 PRO 4650U with Radeon Graphics"], "physical_cores": 6, "logical_cpus": 12})
        if name == "host_snapshot":
            return json.dumps({"hostname": "muninn", "uptime_seconds": 1000, "load_average": [0.2, 0.1, 0.1], "memory": {"total_mb": 15200, "available_mb": 7000}, "disk": {"used_percent": 80.0}})
        if name == "ollama_runtime_snapshot":
            return json.dumps({"models": [{"name": "agent-main:4b"}, {"name": "agent-main:2b"}]})
        if name == "dns_query":
            return json.dumps({"ok": True, "status": "NOERROR", "answers": ["example.com A 93.184.216.34"], "elapsed_ms": 2.0})
        if name == "tcp_connect":
            return json.dumps({"ok": True, "connected_address": "93.184.216.34", "tcp_connect_ms": 15.0})
        if name == "http_probe":
            return json.dumps({"ok": True, "http_status": 200, "time_to_headers_ms": 25.0, "tls_version": "TLSv1.3"})
        if name == "page_metadata":
            return json.dumps({"url": "https://example.com", "canonical": "https://example.com/", "http_status": 200, "title": "Example Domain"})
        if name == "tool_search":
            q = str(args.get("query") or "")
            if "observation" in q:
                return json.dumps([{"name": "read_observation"}])
            if "skills" in q:
                return json.dumps([{"name": "search_skills"}])
            return json.dumps([{"name": x} for x in ("search_recipes", "list_recipes", "load_recipe", "save_recipe", "run_recipe")])
        if name == "write_file":
            return "Successfully wrote 25 characters to generalized_recipe_test/targets.txt"
        if name == "read_file":
            return "example.com\nwww.iana.org\n"
        if name == "search_recipes":
            return json.dumps([candidate()] if saved["value"] else [])
        if name == "save_recipe":
            saved["value"] = True
            saved["definition"] = {
                "id": 77, "name": str(args["name"]), "description": str(args["description"]),
                "pipeline": list(args["stages"]), "parameters": dict(args.get("parameters") or {}),
                "tags": list(args.get("tags") or []), "origin": "user", "target_tool": "",
            }
            return json.dumps({"saved": True, "id": 77, "name": args["name"]})
        if name == "load_recipe":
            return json.dumps(definition())
        if name == "run_recipe":
            host = str((args.get("parameters") or {}).get("hostname") or "")
            title = "Example Domain" if host == "example.com" else "Internet Assigned Numbers Authority"
            return json.dumps({
                "ok": True,
                "stages": [
                    {"id": "dns", "tool": "dns_query", "ok": True, "args": {"name": host, "record_type": "A"}},
                    {"id": "tcp", "tool": "tcp_connect", "ok": True, "args": {"host": host, "port": 443, "timeout": 5.0}},
                    {"id": "https", "tool": "http_probe", "ok": True, "args": {"url": f"https://{host}", "timeout": 8.0, "allow_private": False}},
                    {"id": "page", "tool": "page_metadata", "ok": True, "args": {"url": f"https://{host}"}},
                    {"id": "summary", "tool": "compose_object", "ok": True, "args": {"data": {}}},
                ],
                "result": {
                    "hostname": host,
                    "dns": {"status": "NOERROR", "answers": [f"{host} A 93.184.216.34"]},
                    "tcp": {"ok": True, "tcp_connect_ms": 15.0},
                    "https": {"ok": True, "http_status": 200, "tls_version": "TLSv1.3"},
                    "page": {"title": title, "url": f"https://{host}"},
                },
                "recipe": {"id": 77, "name": "public_endpoint_health_check"},
            })
        if name == "read_observation":
            raw_len = int(late_truncation.get("raw_len") or 0)
            offset = int(args.get("offset") or 0)
            returned = max(1, raw_len - offset)
            return json.dumps({
                "observation_id": str(args.get("observation_id") or ""),
                "offset": offset, "returned_chars": returned,
                "total_chars": raw_len, "has_more": False,
                "content": "x" * min(returned, 32),
            })
        if name == "remove_path":
            late_truncation["cleanup_seen"] = True
            return "Successfully removed generalized_recipe_test"
        if name == "path_stat":
            return json.dumps({"path": "generalized_recipe_test", "exists": False})
        raise AssertionError((name, args))

    class NoModel:
        def chat(self, **kwargs):
            raise AssertionError("model must not be used for deterministic generalized recipe stress plan")

    monkeypatch.setattr(te, "_execute_registered_tool", fake_execute)
    monkeypatch.setattr(te, "WORKING_STATE_ENABLED", False)
    monkeypatch.setattr(te, "RECIPES_ENABLED", False)
    monkeypatch.setattr(te, "LOOP_VALIDATOR_ENABLED", False)
    monkeypatch.setattr(te, "get_conversation_summary", lambda: "")
    monkeypatch.setattr(te, "build_memory_context", lambda *_: "")
    monkeypatch.setattr(te, "get_relevant_user_prompt_context", lambda *_: "")
    monkeypatch.setattr(te, "get_user_location", lambda: "London, Ontario, Canada")
    def bounded_result(name, text):
        # Reproduce the real failure: a late recipe-retention search after the
        # initial truncation audit creates a new archived middle.  Final
        # settlement must recover it before rule 13 / requirement 72.
        if name == "search_recipes" and late_truncation["cleanup_seen"] and late_truncation["armed"]:
            late_truncation["armed"] = False
            late_truncation["raw_len"] = len(text)
            marker = (
                "HEAD\n\n[Harness: middle truncated; full "
                f"{len(text)}-character result stored as observation obs-late. offset=10]\n\nTAIL"
            )
            return marker, "obs-late"
        return text, ""

    monkeypatch.setattr(te, "_bounded_tool_result_with_ref", bounded_result)
    monkeypatch.setattr(te, "evict_report_model_for_interactive", lambda: None)
    monkeypatch.setattr(te, "_prune_compacted_history", lambda _messages: None)
    monkeypatch.setattr(te, "log_perf_stats", lambda *a, **k: None)

    from tools.task_requirements import TaskRequirementLedger
    captured = {}

    class CapturingLedger(TaskRequirementLedger):
        @classmethod
        def from_request(cls, user_text):
            from tools.task_requirements import derive_requirements
            ledger = cls(derive_requirements(user_text))
            captured["ledger"] = ledger
            return ledger

    messages = [{"role": "system", "content": "system"}]
    te.handle_user_turn(
        messages, prompt, False,
        runtime_overrides={
            "TaskRequirementLedger": CapturingLedger,
            "OLLAMA": NoModel(), "record_monitor_state": lambda *a, **k: None,
            "append_and_save": lambda rows, item: rows.append(item),
            "acquire_turn_lock": lambda: object(), "release_turn_lock": lambda _lock: None,
            "acquire_inference_lock": lambda: (_ for _ in ()).throw(AssertionError("model lock should not be acquired")),
            "release_inference_lock": lambda _lock: None, "queue_compaction_if_needed": lambda *a, **k: None,
        },
    )

    content = messages[-1]["content"]
    for heading in (
        "## System baseline", "## Tool-layer routing", "## Capability discovery",
        "## Workspace test", "## Recipe discovery", "## Recipe definition",
        "## Replay: example.com", "## Replay: www.iana.org", "## Generalization audit",
        "## Routing/provenance audit", "## Observation audit", "## Persistent ledger audit",
        "## Mutation/cleanup audit", "## Unresolved requirements",
    ):
        assert heading in content
    assert "72. Deterministic finalization — PASS" in content
    assert "## Unresolved requirements\nNone." in content
    assert "hard turn/model-call budget exhausted" not in content
    assert saved["value"] is True
    assert sum(1 for name, _ in calls if name == "save_recipe") == 1
    runs = [args for name, args in calls if name == "run_recipe"]
    assert [row["parameters"]["hostname"] for row in runs] == ["example.com", "www.iana.org"]
    assert sum(1 for name, _ in calls if name == "tool_search") == 3
    assert any(name == "read_observation" for name, _ in calls)

    ledger = captured["ledger"]
    rule13 = next(item for item in ledger.requirements if item.key == "genrecipe:13")
    assert rule13.status == "satisfied"
    assert "recovered" in rule13.last_reason
    assert len(ledger.requirements) == 72
    assert all(row.status in {"satisfied", "partial"} for row in ledger.requirements)
    for key in ("genrecipe:26", "genrecipe:27", "genrecipe:28"):
        row = next(item for item in ledger.requirements if item.key == key)
        assert row.attempts == 1
        assert row.evidence
        assert row.evidence[0]["source"] == "tool_call"
        assert row.evidence[0]["tool"] == "tool_search"

    # Small direct results are archived specifically for requirement durability,
    # even when they are far below the normal large-observation threshold.
    for key in tuple(f"genrecipe:{n:02d}" for n in range(16, 25)):
        row = next(item for item in ledger.requirements if item.key == key)
        assert row.evidence, key
        direct = row.evidence[-1]
        assert direct["source"] == "tool_call"
        assert direct.get("evidence_ref"), key
        assert direct.get("evidence_preview"), key


def test_generalized_recipe_targets_fixture_is_hidden_from_fallback_file_summary():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "al_agent" / "turn_engine.py").read_text(encoding="utf-8")
    assert 'return value == "generalized_recipe_test/targets.txt"' in source
    assert "if _hide_inline_file_summary(target):" in source


def test_generalized_truncation_audit_recovers_before_finalization():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "al_agent" / "turn_engine.py").read_text(encoding="utf-8")
    cleanup = source.index('requirement_key="genrecipe:61"', source.index('# 59-62:'))
    final_checkpoint = source.index(
        'if pending_truncated_observations:\n                recover_pending_truncated_observations()',
        cleanup,
    )
    rule13 = source.index('"genrecipe:13", "satisfied"', final_checkpoint)
    finalization = source.index('_genrecipe_mark("genrecipe:72"', rule13)
    assert cleanup < final_checkpoint < rule13 < finalization
    assert "unresolved_truncated_observations" in source
    assert "Recovery failures are terminal evidence gaps" in source
