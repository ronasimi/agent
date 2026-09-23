import json


def _snapshot(*, url="https://example.com/", text="", elements=None):
    return {
        "url": url,
        "title": "Example",
        "viewport": {"width": 1000, "height": 800, "scroll_x": 0, "scroll_y": 0},
        "elements": elements or [],
        "text": text,
    }


def test_semantic_diff_is_ref_keyed_and_delta_first():
    from tools.browser_ui import semantic_diff

    before = _snapshot(elements=[
        {"ref": "e1", "role": "textbox", "name": "Email", "value": "", "disabled": False, "checked": None, "selected": None, "expanded": None, "bbox": {"x": 1}, "in_viewport": True},
        {"ref": "e2", "role": "button", "name": "Continue", "value": "", "disabled": True, "checked": None, "selected": None, "expanded": None, "bbox": {"x": 2}, "in_viewport": True},
    ])
    after = _snapshot(text="ready", elements=[
        {"ref": "e1", "role": "textbox", "name": "Email", "value": "a@example.com", "disabled": False, "checked": None, "selected": None, "expanded": None, "bbox": {"x": 1}, "in_viewport": True},
        {"ref": "e2", "role": "button", "name": "Continue", "value": "", "disabled": False, "checked": None, "selected": None, "expanded": None, "bbox": {"x": 2}, "in_viewport": True},
        {"ref": "e3", "role": "alert", "name": "Ready", "value": "", "disabled": False, "checked": None, "selected": None, "expanded": None, "bbox": {"x": 3}, "in_viewport": True},
    ])

    delta = semantic_diff(before, after)
    assert delta["full"] is False
    assert {row["ref"] for row in delta["added"]} == {"e3"}
    changed = {row["ref"]: row for row in delta["changed"]}
    assert changed["e1"]["value"] == "a@example.com"
    assert changed["e2"]["disabled"] is False
    assert delta["text_changed"] is True


def test_verify_snapshot_requires_all_explicit_end_state_predicates():
    from tools.browser_ui import verify_snapshot

    snap = _snapshot(
        url="https://example.com/settings?saved=1",
        text="Settings saved successfully",
        elements=[
            {"ref": "e7", "role": "checkbox", "name": "Notifications", "value": "", "checked": True, "disabled": False, "expanded": None}
        ],
    )
    result = verify_snapshot(snap, [
        {"type": "url_contains", "value": "/settings"},
        {"type": "text_present", "value": "saved successfully"},
        {"type": "element_checked", "ref": "e7", "value": True},
    ])
    assert result["passed"] is True
    assert all(row["passed"] for row in result["checks"])

    failed = verify_snapshot(snap, [{"type": "element_checked", "ref": "e7", "value": False}])
    assert failed["passed"] is False


def test_verify_snapshot_rejects_empty_checks_as_completion_proof():
    from tools.browser_ui import verify_snapshot
    assert verify_snapshot(_snapshot(), [])["passed"] is False


def test_browser_structured_error_flows_into_loop_validator():
    from tools.loop_validator import classify_tool_outcome

    content = json.dumps({
        "ok": False,
        "state_version": 4,
        "error": {"code": "STALE_OBSERVATION", "retryable": True},
    })
    outcome = classify_tool_outcome(content, tool_name="browser_step")
    assert outcome["success"] is False
    assert outcome["reason"] == "browser_stale_observation"

    success = classify_tool_outcome(json.dumps({"ok": True, "state_version": 5}), tool_name="browser_step")
    assert success["success"] is True


def test_browser_tool_schema_is_small_typed_and_mutating():
    import tools

    tools.load_tools()
    schema = tools.get_tool_schema("browser_step")
    props = schema["function"]["parameters"]["properties"]
    assert set(props["op"]["enum"]) == {"observe", "navigate", "click", "type", "select", "scroll", "key", "back", "verify", "new_tab", "list_tabs", "switch_tab", "close_tab", "wait_download"}
    assert props["checks"]["items"]["type"] == "object"
    assert tools.TOOL_METADATA["browser_step"]["readonly"] is False
    assert tools.TOOL_METADATA["browser_step"]["repeat_safe"] is False


def test_ui_intent_exposes_browser_step():
    import tools
    tools.load_tools()
    names = {schema["function"]["name"] for schema in tools.select_tool_schemas(
        "Open the webpage, fill the form, and click the Continue button", max_tools=16
    )}
    assert "browser_step" in names


def test_project_snapshot_prunes_by_task_relevance_and_can_expand():
    from tools.browser_ui import project_snapshot

    elements = []
    for i in range(20):
        elements.append({
            "ref": f"e{i}", "role": "button", "name": f"Generic {i}", "value": "",
            "disabled": False, "checked": None, "selected": None, "expanded": None,
            "bbox": {"x": i, "y": i, "w": 10, "h": 10}, "in_viewport": i < 5,
        })
    elements.append({
        "ref": "e99", "role": "textbox", "name": "Billing postal code", "value": "",
        "disabled": False, "checked": None, "selected": None, "expanded": None,
        "bbox": {"x": 1, "y": 900, "w": 10, "h": 10}, "in_viewport": False,
    })
    snap = _snapshot(elements=elements)
    projected = project_snapshot(snap, task_hint="Enter the billing postal code", max_candidates=8)
    refs = {row["ref"] for row in projected["elements"]}
    assert "e99" in refs
    assert projected["candidate_count_exposed"] <= 8
    assert projected["candidate_pruned"] > 0

    expanded = project_snapshot(snap, task_hint="", max_candidates=200, include_offscreen=True)
    assert expanded["candidate_count_exposed"] == len(elements)


def test_verify_snapshot_p1_predicate_aliases_and_negative_checks():
    from tools.browser_ui import verify_snapshot

    snap = _snapshot(
        url="https://example.com/settings?saved=1",
        text="Settings saved",
        elements=[{
            "ref": "e7", "role": "checkbox", "name": "Notifications", "value": "yes",
            "checked": True, "disabled": False, "expanded": False, "selected": True,
        }],
    )
    snap["tabs"] = [{"url": "https://example.com/settings", "title": "Settings", "active": True}]
    snap["downloads"] = [{"filename": "report.pdf", "completed": True}]
    result = verify_snapshot(snap, [
        {"type": "url_matches", "value": r"/settings"},
        {"type": "page_title_matches", "value": "Example"},
        {"type": "text_absent", "value": "failure"},
        {"type": "element_value_equals", "ref": "e7", "value": "yes"},
        {"type": "element_selected", "ref": "e7", "value": True},
        {"type": "element_not_visible", "ref": "missing"},
        {"type": "tab_open", "value": "/settings"},
        {"type": "download_exists", "value": "report.pdf"},
    ])
    assert result["passed"] is True


def test_browser_state_store_tracks_process_and_outcome_separately(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_DB_PATH", str(tmp_path / "agent.db"))
    from tools.browser_state import BrowserStateStore

    store = BrowserStateStore("browser-state-test")
    store.save_state(state_version=3, state={"url": "https://example.com"}, requirements=[])
    store.record_step(
        state_version=3,
        operation="click",
        action={"op": "click", "ref": "e1"},
        process_success=True,
        outcome_success=None,
        timings={"action_ms": 10},
        token_metrics={"observation_tokens": 20},
        result={"ok": True},
    )
    store.record_step(
        state_version=3,
        operation="verify",
        action={"op": "verify"},
        process_success=True,
        outcome_success=False,
        timings={"verification_ms": 2},
        token_metrics={"observation_tokens": 4},
        result={"ok": False},
    )
    rows = store.trajectory()
    assert rows[0]["process_success"] is True
    assert rows[0]["outcome_success"] is None
    assert rows[1]["outcome_success"] is False
    summary = store.metrics_summary()
    assert summary["process_success_rate"] == 1.0
    assert summary["outcome_success_rate"] == 0.0


def test_browser_tool_schema_exposes_p1_expansion_coordinate_and_screenshot_controls():
    import tools

    tools.load_tools()
    schema = tools.get_tool_schema("browser_step")
    props = schema["function"]["parameters"]["properties"]
    assert props["max_candidates"]["minimum"] == 8
    assert props["max_candidates"]["maximum"] == 200
    assert "include_offscreen" in props
    assert "screenshot" in props
    assert props["x"]["minimum"] == -1
    assert props["y"]["minimum"] == -1
