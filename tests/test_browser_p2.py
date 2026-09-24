import json
from pathlib import Path


def _row(ref, role, name, *, y=0, in_viewport=True, input_type="", disabled=False):
    return {
        "ref": ref,
        "role": role,
        "name": name,
        "value": "",
        "disabled": disabled,
        "checked": None,
        "selected": None,
        "expanded": None,
        "input_type": input_type,
        "region": "Main",
        "in_viewport": in_viewport,
        "viewport_distance": 0 if in_viewport else max(0, y - 800),
        "bbox": {"x": 10, "y": y, "w": 120, "h": 30},
    }


def _snapshot(elements=None, *, text="", url="https://example.com/", focused_ref=""):
    return {
        "url": url,
        "title": "Example",
        "text": text,
        "viewport": {"width": 1000, "height": 800, "scroll_x": 0, "scroll_y": 0},
        "focused_ref": focused_ref,
        "elements": elements or [],
        "tabs": [],
        "downloads": [],
        "popups": [],
    }


def test_p2_safety_boundaries_and_submit_detection():
    from tools.browser_ui import classify_action_safety

    snap = _snapshot([
        _row("e1", "button", "View details"),
        _row("e2", "button", "Save draft"),
        _row("e3", "button", "Submit"),
        _row("e4", "button", "Confirm"),
        _row("e5", "button", "Delete account"),
    ])
    assert classify_action_safety("observe", snap)["level"] == "read_only"
    assert classify_action_safety("click", snap, ref="e1")["level"] == "reversible"
    assert classify_action_safety("click", snap, ref="e2")["level"] == "reversible"
    for ref in ("e3", "e4", "e5"):
        result = classify_action_safety("click", snap, ref=ref)
        assert result["level"] == "consequential"
        assert result["requires_pre_submit_verification"] is True


def test_p2_enter_key_uses_focused_element_for_safety():
    from tools.browser_ui import classify_action_safety

    snap = _snapshot([_row("e9", "button", "Purchase now")], focused_ref="e9")
    assert classify_action_safety("key", snap, value="Enter")["level"] == "consequential"
    assert classify_action_safety("key", snap, value="Escape")["level"] == "reversible"


def test_p2_auth_state_detection_is_conservative():
    from tools.browser_ui import _detect_auth_state

    login = _snapshot([_row("e1", "textbox", "Password", input_type="password")])
    assert _detect_auth_state(login)["state"] == "login_required"

    mfa = _snapshot([_row("e1", "textbox", "Verification code")])
    assert _detect_auth_state(mfa)["state"] == "mfa_required"

    signed = _snapshot([_row("e1", "button", "Sign out")], text="Account settings")
    assert _detect_auth_state(signed)["state"] == "signed_in"


def test_p2_viewport_pruning_and_hierarchical_regions():
    from tools.browser_ui import project_snapshot

    elements = [
        {**_row("e1", "button", "Menu"), "region": "Navigation"},
        {**_row("e2", "textbox", "Search"), "region": "Main"},
        {**_row("e3", "button", "Nearby next", y=1100, in_viewport=False), "region": "Main"},
        {**_row("e4", "button", "Far unrelated", y=4000, in_viewport=False), "region": "Footer"},
    ]
    projected = project_snapshot(_snapshot(elements), task_hint="search", max_candidates=8)
    refs = {row["ref"] for row in projected["elements"]}
    assert {"e1", "e2", "e3"} <= refs
    assert "e4" not in refs
    regions = {region["name"]: region for region in projected["regions"]}
    assert "Navigation" in regions and "Main" in regions
    assert "e1" in regions["Navigation"]["refs"]


def test_p2_browser_tool_schema_includes_tab_download_and_timeout_controls():
    import tools

    tools.load_tools()
    schema = tools.get_tool_schema("browser_step")
    props = schema["function"]["parameters"]["properties"]
    assert {"new_tab", "list_tabs", "switch_tab", "close_tab", "wait_download"} <= set(props["op"]["enum"])
    assert props["tab_index"]["minimum"] == -1
    assert props["timeout_ms"]["maximum"] == 15000
    assert "pre-submit" in props["checks"]["description"].lower()


def test_p2_benchmark_budget_enforces_each_resource_dimension():
    from tools.browser_benchmark import BenchmarkBudget

    budget = BenchmarkBudget(
        max_model_calls=1,
        max_browser_actions=1,
        max_wall_time_s=999,
        max_prompt_tokens=10,
        max_output_tokens=5,
        max_recovery_attempts=1,
    )
    budget.consume_model(prompt_tokens=8, output_tokens=2)
    budget.consume_browser()
    assert budget.violation() is None
    budget.consume_model(prompt_tokens=3, output_tokens=0)
    assert budget.violation() in {"model_calls", "prompt_tokens"}

    recovery = BenchmarkBudget(max_recovery_attempts=1, max_wall_time_s=999)
    recovery.consume_browser(recovery=True)
    assert recovery.violation() is None
    recovery.consume_browser(recovery=True)
    assert recovery.violation() == "recovery_attempts"


def test_p2_browsergym_adapter_translates_compact_actions_without_dependency():
    from tools.browser_benchmark import BrowserGymAdapter

    assert BrowserGymAdapter.to_browsergym_action({"op": "click", "ref": "42"}) == 'click("42")'
    assert BrowserGymAdapter.to_browsergym_action({"op": "type", "ref": "42", "value": "abc"}) == 'fill("42", "abc")'
    assert BrowserGymAdapter.to_browsergym_action({"op": "key", "value": "Enter"}) == 'keyboard_press("Enter")'
    assert BrowserGymAdapter.to_browsergym_action({"op": "key", "ref": "42", "value": "Enter"}) == 'press("42", "Enter")'
    assert BrowserGymAdapter.to_browsergym_action({"op": "scroll", "direction": "down", "amount": 350}) == "scroll(0, 350)"
    assert BrowserGymAdapter.to_browsergym_action({"op": "scroll", "direction": "left", "amount": 200}) == "scroll(-200, 0)"
    assert BrowserGymAdapter.to_browsergym_action({"op": "navigate", "url": "https://example.com"}) == 'goto("https://example.com")'
    assert BrowserGymAdapter.to_browsergym_action({"op": "new_tab"}) == "new_tab()"
    assert BrowserGymAdapter.to_browsergym_action({"op": "switch_tab", "tab_index": 2}) == "tab_focus(2)"
    assert BrowserGymAdapter.to_browsergym_action({"op": "close_tab", "tab_index": -1}) == "tab_close()"
    assert BrowserGymAdapter.to_browsergym_actions({"op": "close_tab", "tab_index": 2}) == ["tab_focus(2)", "tab_close()"]

    normalized = BrowserGymAdapter.normalize_observation({
        "url": "https://example.com",
        "axtree_object": {"node": "root"},
        "dom_object": {"node": "html"},
        "extra_element_properties": {"42": {"visibility": 1}},
    })
    assert normalized["axtree_object"] == {"node": "root"}
    assert "raw_html" not in normalized


def test_p2_benchmark_store_and_regression_dashboard(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_DB_PATH", str(tmp_path / "agent.db"))
    from tools.browser_benchmark import BrowserBenchmarkStore, regression_dashboard

    store = BrowserBenchmarkStore()
    store.record_run(
        benchmark="unit",
        task_id="task-1",
        success=True,
        reward=1.0,
        metrics={"task_ms": 100, "steps": 3, "prompt_tokens": 200, "output_tokens": 40},
        budget={"violation": None},
    )
    store.record_run(
        benchmark="unit",
        task_id="task-2",
        success=False,
        reward=0.0,
        metrics={"task_ms": 300, "steps": 5, "prompt_tokens": 300, "output_tokens": 60},
        budget={"violation": None},
        error="failed",
    )
    dashboard = regression_dashboard(limit=10)
    assert dashboard["current"]["runs"] == 1
    assert dashboard["previous"]["runs"] == 1
    assert len(dashboard["recent_runs"]) == 2
    assert dashboard["live_ui"]["steps"] == 0


def test_p2_optional_benchmark_files_and_dashboard_ui_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "diagnostics/requirements-benchmark.txt").read_text(encoding="utf-8").strip()
    assert (root / "diagnostics" / "benchmarks" / "benchmark_browsergym.py").is_file()
    assert (root / "diagnostics" / "browser_regression_dashboard.py").is_file()
    html = (root / "webui" / "static" / "index.html").read_text(encoding="utf-8")
    js = (root / "webui" / "static" / "app.js").read_text(encoding="utf-8")
    server = (root / "webui" / "server.py").read_text(encoding="utf-8")
    assert 'id="benchmarksPanel"' in html
    assert "/api/browser-benchmarks" in js
    assert '@app.get("/api/browser-benchmarks")' in server


def test_p2_browser_state_audit_retains_safety_and_recovery_when_bounded():
    from tools.browser_state import _bounded

    payload = {
        "ok": True,
        "operation": "click",
        "state_version": 5,
        "safety": {"level": "consequential", "pre_submit_verified": True},
        "recovery": {"attempted": True, "events": [{"kind": "scroll_target_into_view"}]},
        "metrics": {"timings_ms": {"action_ms": 2}},
        "padding": "x" * 20000,
    }
    bounded = _bounded(payload, limit=500)
    # P2 diagnostic fields must survive bounded trajectory records.
    assert bounded.get("safety", {}).get("level") == "consequential"
    assert bounded.get("recovery", {}).get("attempted") is True


def test_p2_late_popup_adoption_switches_without_extra_model_roundtrip():
    import asyncio
    from tools.browser_ui import BrowserSession, _adopt_pending_popup

    class FakePage:
        def __init__(self, url):
            self.url = url
            self.front = False
        def is_closed(self):
            return False
        async def bring_to_front(self):
            self.front = True
        async def wait_for_load_state(self, *_args, **_kwargs):
            return None

    opener = FakePage("https://example.com/")
    popup = FakePage("https://example.com/popup")

    class FakeContext:
        pages = [opener, popup]

    session = BrowserSession(context=FakeContext(), page=opener)
    session.popup_events.append({"index": 1, "url": popup.url})
    event = asyncio.run(_adopt_pending_popup(session))
    assert event and event["kind"] == "popup_registered_and_switched"
    assert session.page is popup
    assert popup.front is True
    assert session.handled_popup_events == 1
    assert session.recovery_attempts == 1
    assert asyncio.run(_adopt_pending_popup(session)) is None
