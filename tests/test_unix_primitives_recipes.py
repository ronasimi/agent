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


def test_recipe_selector_bundle():
    from tools import load_tools, select_tool_schemas
    load_tools()
    names = {s["function"]["name"] for s in select_tool_schemas("find and run my saved recipe for checking a website", max_tools=12)}
    assert "run_recipe" in names
    assert "search_recipes" in names


def test_pipeline_schema_is_bounded():
    from tools.pipeline import execute_pipeline
    result = execute_pipeline([{"tool": "calculate", "args": {"expression": "1+1"}}] * 17)
    assert result["ok"] is False
    assert "exceeds" in result["error"]

def test_successful_turn_emits_recipe_suggestion(tmp_path, monkeypatch):
    _set_recipe_db(tmp_path, monkeypatch)
    import agent
    from tools import AVAILABLE_TOOLS_MAP

    def fake_resolve_host(host: str, record_type: str = "any") -> str:
        return json.dumps({"host": host, "addresses": ["93.184.216.34"]})

    def fake_tcp_connect(host: str, port: int, timeout: float = 5.0) -> str:
        return json.dumps({"host": host, "port": port, "ok": True})

    monkeypatch.setitem(AVAILABLE_TOOLS_MAP, "resolve_host", fake_resolve_host)
    monkeypatch.setitem(AVAILABLE_TOOLS_MAP, "tcp_connect", fake_tcp_connect)
    monkeypatch.setattr(agent, "_acquire_inference_lock", lambda: None)
    monkeypatch.setattr(agent, "_release_inference_lock", lambda lock: None)
    monkeypatch.setattr(agent, "record_monitor_state", lambda *a, **k: None)
    monkeypatch.setattr(agent, "_queue_compaction_if_needed", lambda *a, **k: None)
    monkeypatch.setattr(agent, "append_and_save", lambda messages, msg: messages.append(msg))
    monkeypatch.setattr(agent, "RECIPE_MATCH_THRESHOLD", 2.0)  # do not match old recipes
    monkeypatch.setattr(agent.TaskRequirementLedger, "from_request", classmethod(lambda cls, text: cls([])))

    class FakeClient:
        def __init__(self): self.calls = 0
        def chat(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return iter([{"done": True, "message": {"content": "", "tool_calls": [{"id":"1","function":{"name":"resolve_host","arguments":{"host":"example.com"}}}]}}])
            if self.calls == 2:
                return iter([{"done": True, "message": {"content": "", "tool_calls": [{"id":"2","function":{"name":"tcp_connect","arguments":{"host":"example.com","port":443}}}]}}])
            return iter([{"done": True, "message": {"content": "Connectivity verified.", "tool_calls": []}}])

    monkeypatch.setattr(agent, "OLLAMA", FakeClient())
    events=[]
    messages=[{"role":"system","content":"system"}]
    with agent.frontend_event_context(lambda e: events.append(e)):
        agent.handle_user_turn(messages, "Resolve example.com and test TCP connectivity to port 443", False)
    assert any(e.get("type") == "recipe_suggestion" for e in events)


def test_turn_preflights_recipes_before_model_planning(tmp_path, monkeypatch):
    _set_recipe_db(tmp_path, monkeypatch)
    from tools.recipe_store import save_recipe
    save_recipe(
        "Check website connectivity", "check website connectivity with DNS and TCP",
        [{"tool": "resolve_host", "args": {"host": {"$param": "host"}}}],
        {"host": {"description": "hostname"}}, ["website", "connectivity"],
    )
    import agent
    monkeypatch.setattr(agent, "_acquire_inference_lock", lambda: None)
    monkeypatch.setattr(agent, "_release_inference_lock", lambda lock: None)
    monkeypatch.setattr(agent, "record_monitor_state", lambda *a, **k: None)
    monkeypatch.setattr(agent, "_queue_compaction_if_needed", lambda *a, **k: None)
    monkeypatch.setattr(agent, "append_and_save", lambda messages, msg: messages.append(msg))
    monkeypatch.setattr(agent, "RECIPE_MATCH_THRESHOLD", 0.1)
    monkeypatch.setattr(agent.TaskRequirementLedger, "from_request", classmethod(lambda cls, text: cls([])))

    class FakeClient:
        def __init__(self): self.requests = []
        def chat(self, **kwargs):
            self.requests.append(kwargs)
            return iter([{"done": True, "message": {"content": "Connectivity task considered.", "tool_calls": []}}])

    client = FakeClient(); monkeypatch.setattr(agent, "OLLAMA", client)
    events = []; messages = [{"role": "system", "content": "system"}]
    with agent.frontend_event_context(lambda event: events.append(event)):
        agent.handle_user_turn(messages, "check website connectivity", False)

    event_types = [event.get("type") for event in events]
    assert event_types.index("recipe_check") < event_types.index("model_start")
    assert next(event for event in events if event.get("type") == "recipe_check")["best_match"] == "Check website connectivity"
    prompt = "\n".join(str(item.get("content") or "") for item in client.requests[0]["messages"])
    assert "[Harness recipe preflight]" in prompt and "Check website connectivity" in prompt
    exposed = {item["function"]["name"] for item in client.requests[0]["tools"]}
    assert "run_recipe" in exposed


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


def test_fast_model_keep_alive_uses_short_residency_ttl_in_config():
    import yaml
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "config" / "config.yaml").read_text(encoding="utf-8"))
    assert config["agent"]["fast_model_keep_alive"] == "2m"
    assert config["agent"]["tool_loop_validator"]["keep_alive"] == "2m"
    assert config["worker"]["fast_model_keep_alive"] == "2m"
