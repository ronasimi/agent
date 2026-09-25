import json
import os
from pathlib import Path


def _set_recipe_db(tmp_path, monkeypatch):
    db = tmp_path / "recipes.db"
    monkeypatch.setenv("AGENT_RECIPE_DB", str(db))
    return db


def test_primitive_registry_and_readonly_metadata():
    from tools import AVAILABLE_TOOLS_MAP, TOOL_METADATA, load_tools
    load_tools()
    expected = {
        "path_stat", "find_paths", "read_text", "tail_file", "file_hash", "text_search",
        "json_query", "json_filter", "json_sort", "json_diff", "list_processes", "process_info",
        "resolve_host", "route_lookup", "tcp_connect", "tls_handshake", "http_request",
        "document_info", "document_text", "archive_list", "calculate", "command_available",
        "run_pipeline", "run_recipe", "search_recipes", "list_recipes", "save_recipe",
        "read_lines", "directory_size", "regex_replace", "text_split", "json_keys",
        "csv_summary", "cpu_info", "os_release", "process_tree", "ping_host",
        "extract_tables", "git_branches",
    }
    assert expected <= set(AVAILABLE_TOOLS_MAP)
    assert TOOL_METADATA["run_pipeline"]["readonly"] is True
    assert TOOL_METADATA["save_recipe"]["readonly"] is False
    assert TOOL_METADATA["archive_extract"]["readonly"] is False
    assert TOOL_METADATA["image_resize"]["readonly"] is False


def test_file_text_json_and_calculator_primitives(tmp_path, monkeypatch):
    import tools.workspace as ws
    import tools.primitive_ops as p
    root = tmp_path / "workspace"; root.mkdir()
    monkeypatch.setattr(ws, "WORKSPACE_DIR", str(root))
    (root / "a.txt").write_text("alpha\nbeta error\ngamma\n", encoding="utf-8")
    (root / "data.json").write_text('[{"v":3},{"v":1},{"v":2}]', encoding="utf-8")
    assert json.loads(p.path_stat("a.txt"))["exists"] is True
    assert json.loads(p.find_paths(".", "*.txt", "file"))["matches"]
    assert "beta error" in p.tail_file("a.txt", 2)
    assert json.loads(p.text_search("error", path="a.txt"))["matches"][0]["line"] == 2
    assert json.loads(p.json_sort("v", path="data.json"))[0]["v"] == 1
    assert json.loads(p.calculate("111042/3600"))["result"] == 30.845


def test_additional_text_structured_and_filesystem_primitives(tmp_path, monkeypatch):
    import tools.workspace as ws
    import tools.primitive_ops as p
    root = tmp_path / "workspace"; root.mkdir()
    monkeypatch.setattr(ws, "WORKSPACE_DIR", str(root))
    (root / "lines.txt").write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")
    (root / "rows.csv").write_text("name,value\na,2\nb,4\nc,\n", encoding="utf-8")

    selected = json.loads(p.read_lines("lines.txt", 2, 3))
    assert [row["text"] for row in selected["lines"]] == ["two", "three"]
    assert json.loads(p.directory_size("."))["files"] == 2
    replaced = json.loads(p.regex_replace(r"t\w+", "T", text="one two three"))
    assert replaced["text"] == "one T T" and replaced["replacements"] == 2
    assert json.loads(p.text_split(text="a|b|c", delimiter="|"))["parts"] == ["a", "b", "c"]
    assert json.loads(p.json_keys(data={"a": 1, "b": 2}))["keys"] == ["a", "b"]
    summary = json.loads(p.csv_summary(path="rows.csv"))
    assert summary["row_count"] == 3 and summary["numeric"]["value"]["mean"] == 3.0


def test_extract_tables_primitive():
    from tools.primitive_ops import extract_tables
    payload = json.loads(extract_tables("<table><caption>T</caption><tr><th>A</th></tr><tr><td>1</td></tr></table>"))
    assert payload["tables"][0]["caption"] == "T"
    assert payload["tables"][0]["headers"] == ["A"]


def test_pipeline_passes_intermediate_refs_without_model_roundtrip(monkeypatch):
    from tools import load_tools
    from tools.pipeline import execute_pipeline
    import tools.primitive_ops as p
    load_tools()
    # Use deterministic in-memory JSON primitives. s2 consumes s1's parsed output.
    result = execute_pipeline([
        {"id": "s1", "tool": "calculate", "args": {"expression": "6*7"}},
        {"id": "s2", "tool": "compare_values", "args": {"a": {"$ref": "s1", "path": "result"}, "b": "41"}},
    ])
    assert result["ok"] is True
    assert result["result"]["equal"] is False
    assert len(result["stages"]) == 2


def test_pipeline_rejects_mutating_tools():
    from tools import load_tools
    from tools.pipeline import execute_pipeline
    load_tools()
    result = execute_pipeline([{"tool": "write_file", "args": {"filename": "x", "content": "y"}}])
    assert result["ok"] is False
    assert "read-only" in result["error"]


def test_semantic_recipe_store_and_execution(tmp_path, monkeypatch):
    _set_recipe_db(tmp_path, monkeypatch)
    from tools import load_tools
    from tools.recipe_store import save_recipe, search_recipes, get_recipe
    from tools.pipeline import run_recipe
    load_tools()
    save_recipe(
        "Convert uptime to hours",
        "calculate an uptime duration in hours",
        [{"id": "s1", "tool": "calculate", "args": {"expression": {"$param": "expression", "default": "3600/3600"}}}],
        {"expression": {"default": "3600/3600", "description": "arithmetic expression"}},
        ["uptime", "calculate", "hours"],
    )
    matches = search_recipes("calculate uptime hours")
    assert matches and matches[0]["name"] == "Convert uptime to hours"
    payload = json.loads(run_recipe("Convert uptime to hours", {"expression": "7200/3600"}))
    assert payload["ok"] is True
    assert payload["result"]["result"] == 2.0
    assert get_recipe("Convert uptime to hours")["use_count"] == 1


def test_recipe_preflight_checks_and_renders_relevant_recipe(tmp_path, monkeypatch):
    _set_recipe_db(tmp_path, monkeypatch)
    from tools.recipe_store import save_recipe, check_recipes_for_task, render_recipe_preflight
    save_recipe(
        "Check website connectivity", "check website connectivity with DNS and TCP",
        [{"tool": "resolve_host", "args": {"host": {"$param": "host"}}}],
        {"host": {"description": "hostname"}}, ["website", "connectivity"],
    )
    report = check_recipes_for_task("check website connectivity", threshold=0.2)
    assert report["checked"] is True
    assert report["relevant"][0]["name"] == "Check website connectivity"
    rendered = render_recipe_preflight(report)
    assert "Harness recipe preflight" in rendered
    assert "Check website connectivity" in rendered


def test_successful_trace_candidate_and_confirmation(tmp_path, monkeypatch):
    _set_recipe_db(tmp_path, monkeypatch)
    from tools.recipe_learning import maybe_create_recipe_candidate, pending_recipe_prompt, handle_recipe_confirmation
    from tools.recipe_store import list_recipes
    trace = [
        {"tool": "resolve_host", "args": {"host": "example.com"}, "success": True, "readonly": True},
        {"tool": "tcp_connect", "args": {"host": "example.com", "port": 443}, "success": True, "readonly": True},
    ]
    candidate = maybe_create_recipe_candidate("check example.com connectivity", trace, 2)
    assert candidate and candidate["stages"] == 2
    assert pending_recipe_prompt()
    handled, reply = handle_recipe_confirmation("yes save it")
    assert handled and "Saved recipe" in reply
    assert len(list_recipes()) == 1


def test_candidate_is_expired_by_unrelated_response(tmp_path, monkeypatch):
    _set_recipe_db(tmp_path, monkeypatch)
    from tools.recipe_learning import maybe_create_recipe_candidate, handle_recipe_confirmation
    from tools.recipe_store import pending_candidate
    trace = [
        {"tool": "resolve_host", "args": {"host": "example.com"}, "success": True, "readonly": True},
        {"tool": "tcp_connect", "args": {"host": "example.com", "port": 443}, "success": True, "readonly": True},
    ]
    assert maybe_create_recipe_candidate("connectivity test", trace, 2)
    handled, _ = handle_recipe_confirmation("what time is it")
    assert handled is False
    assert pending_candidate() is None


def test_recipe_confirmation_does_not_steal_artifact_save_request(tmp_path, monkeypatch):
    _set_recipe_db(tmp_path, monkeypatch)
    from tools.recipe_learning import maybe_create_recipe_candidate, handle_recipe_confirmation
    trace = [
        {"tool": "resolve_host", "args": {"host": "example.com"}, "success": True, "readonly": True},
        {"tool": "tcp_connect", "args": {"host": "example.com", "port": 443}, "success": True, "readonly": True},
    ]
    assert maybe_create_recipe_candidate("connectivity test", trace, 2)
    handled, reply = handle_recipe_confirmation("yes save it as report.md")
    assert handled is False and reply == ""


def test_builtin_weather_workflow_is_not_suggested_as_duplicate_recipe(tmp_path, monkeypatch):
    _set_recipe_db(tmp_path, monkeypatch)
    from tools.recipe_learning import maybe_create_recipe_candidate
    trace = [
        {"tool": "geocode_location", "args": {"query": "London ON"}, "success": True, "readonly": True},
        {
            "tool": "weather_forecast",
            "args": {"latitude": 42.98, "longitude": -81.23, "forecast_days": 7},
            "success": True,
            "readonly": True,
        },
    ]
    assert maybe_create_recipe_candidate("weather for London ON", trace, 2) is None




def test_pipeline_schema_is_bounded():
    from tools.pipeline import execute_pipeline
    result = execute_pipeline([{"tool": "calculate", "args": {"expression": "1+1"}}] * 17)
    assert result["ok"] is False
    assert "exceeds" in result["error"]





def test_pipeline_extended_bound_and_foreach_condition(monkeypatch):
    from tools import AVAILABLE_TOOLS_MAP, TOOL_METADATA, load_tools
    from tools.pipeline import execute_pipeline, MAX_STAGES
    load_tools()
    assert MAX_STAGES == 16
    too_many = execute_pipeline([{"tool": "calculate", "args": {"expression": "1+1"}}] * 17)
    assert too_many["ok"] is False and "exceeds" in too_many["error"]

    result = execute_pipeline([
        {"id": "rows", "tool": "calculate", "foreach": [1, 2, 3], "args": {"expression": {"$item": "$"}}},
        {"id": "skipped", "tool": "calculate", "when": False, "args": {"expression": "99"}},
        {"id": "result", "tool": "compose_object", "args": {"data": {"rows": {"$ref": "rows"}, "skipped": {"$ref": "skipped"}}}},
    ])
    assert result["ok"] is True
    assert [x["result"] for x in result["result"]["rows"]] == [1.0, 2.0, 3.0]
    assert result["result"]["skipped"] is None


def test_builtin_compatibility_recipes_seed_and_cover_monoliths(tmp_path, monkeypatch):
    _set_recipe_db(tmp_path, monkeypatch)
    from tools import load_tools
    from tools.recipe_compat import seed_builtin_recipes, compatibility_coverage
    from tools.recipe_store import list_recipes
    load_tools()
    seeded = seed_builtin_recipes()
    assert seeded["errors"] == []
    rows = list_recipes(100)
    builtins = [r for r in rows if r.get("origin") == "builtin"]
    assert len(builtins) >= 20
    assert any(r["name"] == "compat.host_snapshot" and r["target_tool"] == "host_snapshot" for r in builtins)
    coverage = compatibility_coverage()
    assert coverage["complete"] is True
    assert coverage["unclassified"] == []
    assert coverage["expected_monolithic_count"] == coverage["classified_count"]


def test_builtin_recipe_seeding_is_idempotent(tmp_path, monkeypatch):
    _set_recipe_db(tmp_path, monkeypatch)
    from tools.recipe_compat import seed_builtin_recipes
    from tools.recipe_store import list_recipes
    seed_builtin_recipes(); first = [r for r in list_recipes(100) if r.get("origin") == "builtin"]
    seed_builtin_recipes(); second = [r for r in list_recipes(100) if r.get("origin") == "builtin"]
    assert len(first) == len(second)
    assert {r["builtin_key"] for r in first} == {r["builtin_key"] for r in second}


def test_process_snapshot_compat_recipe_executes(tmp_path, monkeypatch):
    _set_recipe_db(tmp_path, monkeypatch)
    from tools import load_tools
    from tools.recipe_compat import seed_builtin_recipes
    from tools.pipeline import run_recipe
    load_tools(); seed_builtin_recipes()
    payload = json.loads(run_recipe("compat.process_snapshot", {"limit": 2, "sort_by": "memory"}))
    assert payload["ok"] is True
    assert payload["recipe"]["origin"] == "builtin"
    assert payload["result"]["sort_by"] == "memory"
    assert len(payload["result"]["processes"]) <= 2


def test_builtin_recipe_names_are_reserved(tmp_path, monkeypatch):
    _set_recipe_db(tmp_path, monkeypatch)
    from tools.recipe_compat import seed_builtin_recipes
    from tools.recipe_store import save_recipe
    seed_builtin_recipes()
    import pytest
    with pytest.raises(ValueError):
        save_recipe("compat.host_snapshot", "overwrite builtin", [{"tool": "calculate", "args": {"expression": "1"}}])


def test_pipeline_treats_soft_tool_failure_as_recipe_failure(monkeypatch):
    from tools import AVAILABLE_TOOLS_MAP, TOOL_METADATA
    from tools.pipeline import execute_pipeline

    monkeypatch.setitem(AVAILABLE_TOOLS_MAP, "fake_empty_search", lambda: "No search results found for query: x")
    monkeypatch.setitem(TOOL_METADATA, "fake_empty_search", {"readonly": True})
    result = execute_pipeline([{"tool": "fake_empty_search", "args": {}}])
    assert result["ok"] is False
    assert "no_progress_result" in result["error"]




def test_optional_failed_pipeline_stage_is_not_credited_as_provenance(monkeypatch):
    import json
    from tools import AVAILABLE_TOOLS_MAP, TOOL_METADATA, load_tools
    from tools.grounding import make_observation
    from tools.pipeline import execute_pipeline

    load_tools()
    monkeypatch.setitem(AVAILABLE_TOOLS_MAP, "fake_optional_failure", lambda: "Error: optional provider unavailable")
    monkeypatch.setitem(TOOL_METADATA, "fake_optional_failure", {"readonly": True})
    result = execute_pipeline([
        {"id": "optional", "tool": "fake_optional_failure", "args": {}, "optional": True},
        {"id": "final", "tool": "calculate", "args": {"expression": "1+1"}},
    ])
    assert result["ok"] is True
    assert result["stages"][0]["ok"] is False
    assert result["stages"][0]["optional_failure"] is True
    observation = make_observation("run_pipeline", json.dumps(result), turn_id=1)
    assert "fake_optional_failure" not in observation["source_tools"]
    assert "calculate" in observation["source_tools"]


def test_pipeline_choose_value_preserves_json_values_from_conditional_stages():
    from tools import load_tools
    from tools.pipeline import execute_pipeline
    load_tools()
    stages = [
        {"id": "left", "tool": "compose_object", "when": {"$param": "use_left"}, "args": {"data": {"side": "left"}}},
        {"id": "right", "tool": "compose_object", "when": {"$not": {"$param": "use_left"}}, "args": {"data": {"side": "right"}}},
        {"id": "result", "tool": "choose_value", "args": {
            "condition": {"$param": "use_left"}, "if_true": {"$ref": "left"}, "if_false": {"$ref": "right"},
        }},
    ]
    left = execute_pipeline(stages, {"use_left": True})
    right = execute_pipeline(stages, {"use_left": False})
    assert left["ok"] is True and left["result"] == {"side": "left"}
    assert right["ok"] is True and right["result"] == {"side": "right"}


def test_choose_value_manifest_schema_accepts_unconstrained_json_values():
    from tools import AVAILABLE_TOOLS_MAP, TOOL_SCHEMAS, load_tools
    from tools.tool_registry import normalize_arguments
    load_tools()
    schema = next(row for row in TOOL_SCHEMAS if row["function"]["name"] == "choose_value")
    props = schema["function"]["parameters"]["properties"]
    assert "type" not in props["if_true"]
    assert "type" not in props["if_false"]
    normalized = normalize_arguments(
        AVAILABLE_TOOLS_MAP["choose_value"],
        {"condition": False, "if_true": None, "if_false": {"ok": True}},
    )
    assert normalized["if_true"] is None
    assert normalized["if_false"] == {"ok": True}


def test_fixed_builtin_recipe_versions_and_positive_defaults():
    from tools.recipe_compat import discover_recipe_specs
    specs, _ = discover_recipe_specs()
    by_name = {row["name"]: row for row in specs}
    assert by_name["compat.endpoint_probe"]["version"] >= 2
    assert by_name["compat.http_probe"]["version"] >= 2
    assert by_name["compat.read_feed"]["version"] >= 2
    assert by_name["compat.read_feed"]["parameters"]["url"]["default"].startswith("https://feeds.bbci.co.uk/")
    assert by_name["compat.read_host_file"]["version"] >= 2
    assert by_name["compat.read_host_file"]["parameters"]["filepath"]["default"] == "/etc/os-release"


def test_fixed_probe_recipes_execute_both_transport_branches_without_type_loss(monkeypatch):
    from tools import load_tools
    from tools import executor
    from tools.pipeline import execute_pipeline
    from tools.recipe_compat import discover_recipe_specs

    load_tools()
    specs, _ = discover_recipe_specs()
    by_name = {row["name"]: row for row in specs}
    original = executor.execute_registered_tool

    def fake_execute(name, args):
        if name == "tcp_connect":
            return json.dumps({"host": args["host"], "port": args["port"], "connected": True})
        if name == "tls_handshake":
            return json.dumps({"host": args["host"], "port": args["port"], "tls": True})
        if name == "http_request":
            return json.dumps({"url": args["url"], "status": 200})
        return original(name, args)

    monkeypatch.setattr(executor, "execute_registered_tool", fake_execute)

    endpoint = by_name["compat.endpoint_probe"]["pipeline"]
    tcp = execute_pipeline(endpoint, {"host": "example.com", "port": 80, "tls": False, "timeout": 5.0})
    tls = execute_pipeline(endpoint, {"host": "example.com", "port": 443, "tls": True, "timeout": 5.0})
    assert tcp["ok"] is True and tcp["result"]["connected"] is True
    assert tls["ok"] is True and tls["result"]["tls"] is True

    http = by_name["compat.http_probe"]["pipeline"]
    plain = execute_pipeline(http, {"url": "http://example.com/", "timeout": 5.0, "allow_private": False})
    secure = execute_pipeline(http, {"url": "https://example.com/", "timeout": 5.0, "allow_private": False})
    assert plain["ok"] is True and plain["result"]["transport"]["connected"] is True
    assert secure["ok"] is True and secure["result"]["transport"]["tls"] is True


def test_recipe_learning_generalizes_hostname_across_host_and_url_args():
    from tools.recipe_learning import build_candidate

    trace = [
        {"tool": "dns_query", "args": {"name": "example.com", "record_type": "A"}, "success": True, "readonly": True},
        {"tool": "tcp_connect", "args": {"host": "example.com", "port": 443, "timeout": 5.0}, "success": True, "readonly": True},
        {"tool": "http_probe", "args": {"url": "https://example.com", "timeout": 8.0, "allow_private": False}, "success": True, "readonly": True},
        {"tool": "page_metadata", "args": {"url": "https://example.com"}, "success": True, "readonly": True},
    ]
    stages, params = build_candidate("check example.com endpoint health", trace)
    assert "hostname" in params
    assert params["hostname"]["default"] == "example.com"
    assert stages[0]["args"]["name"]["$param"] == "hostname"
    assert stages[0]["args"]["record_type"] == "A"
    assert params["hostname"]["type"] == "string"
    assert stages[1]["args"]["host"]["$param"] == "hostname"
    assert stages[1]["args"]["port"] == 443
    assert stages[1]["args"]["timeout"] == 5.0
    assert stages[2]["args"]["allow_private"] is False
    assert stages[2]["args"]["url"]["$template"] == "https://{hostname}"
    assert stages[3]["args"]["url"]["$template"] == "https://{hostname}"


def test_recipe_learning_fast_hints_are_advisory_and_cannot_inject_values():
    from tools.recipe_learning import build_candidate

    trace = [
        {"tool": "dns_query", "args": {"name": "example.com", "record_type": "A"}, "success": True, "readonly": True},
        {"tool": "http_probe", "args": {"url": "https://example.com", "timeout": 8.0}, "success": True, "readonly": True},
    ]
    stages, params = build_candidate(
        "check example.com",
        trace,
        semantic_hints=[
            {"name": "site", "value": "example.com"},
            {"name": "injected", "value": "evil.example"},
            {"name": "token", "value": "abcdefghijklmnopqrstuvwxyz0123456789"},
        ],
    )
    assert "site" in params
    assert params["site"]["default"] == "example.com"
    assert all(meta.get("default") != "evil.example" for meta in params.values())
    assert "token" not in params
    assert stages[1]["args"]["url"]["$template"] == "https://{site}"
