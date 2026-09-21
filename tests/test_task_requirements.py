from tools.task_requirements import (
    TaskRequirementLedger, build_news_query, derive_task_frame, is_evidence_reuse_request,
    is_followup_request, is_task_continuation,
)
from tools.turn_policy import derive_turn_tool_policy


def test_broad_health_prompt_requires_all_explicit_capabilities():
    text = """
    Inspect host CPU, memory, disk, temperature and pressure state; top resource-consuming processes;
    filesystem inode usage; failed services; network interfaces, routes, neighbors, listening sockets and established connections;
    current tool/dependency health. Resolve example.com, probe HTTPS connectivity, inspect the network path.
    Research the current official documentation and include source URLs. Take a screenshot of the documentation page.
    Report repository status and run compile/config/lint/test checks. Do not modify any files.
    """
    tools = set(TaskRequirementLedger.from_request(text).required_tools())
    expected = {
        "host_snapshot", "pressure_snapshot", "process_snapshot", "filesystem_snapshot", "service_health",
        "network_snapshot", "neighbor_snapshot", "connection_snapshot", "dns_diagnose", "http_probe", "network_path",
        "tool_health", "dependency_audit", "web_search", "browse_url", "take_web_screenshot", "repo_status", "repo_checks",
    }
    assert expected <= tools


def test_requirement_ledger_tracks_completion_without_inference():
    ledger = TaskRequirementLedger.from_request("Resolve example.com and take a screenshot of the page")
    assert set(ledger.required_tools()) == {"dns_diagnose", "take_web_screenshot"}
    ledger.record_tool("dns_diagnose", status="ok", fingerprint="one")
    assert ledger.required_tools(pending_only=True) == ["take_web_screenshot"]
    ledger.record_tool("take_web_screenshot", status="partial", reason="blocked image")
    assert ledger.pending() == []


def test_followup_detection_is_conservative():
    assert is_followup_request("Without rerunning tools, summarize those results") is True
    assert is_followup_request("Perform a structured health and capability assessment of this system with host and network diagnostics") is False


def test_readonly_policy_can_exempt_explicit_safe_screenshot_artifact():
    metadata = {
        "take_web_screenshot": {"readonly": False, "safe_artifact": True},
        "write_file": {"readonly": False, "safe_artifact": False},
    }
    policy = derive_turn_tool_policy(
        "Take a screenshot. Inspect the repository and do not modify any files.",
        set(metadata),
        metadata,
    )
    assert policy.readonly_only is False
    policy.allow_explicit_requirements({"take_web_screenshot"}, metadata)
    assert policy.allowed("take_web_screenshot", metadata["take_web_screenshot"]) is True
    assert policy.allowed("write_file", metadata["write_file"]) is False


def test_turn_policy_supports_counted_conditional_tool_restriction():
    metadata = {
        "execute_shell": {"readonly": False, "safe_artifact": False},
        "execute_python": {"readonly": False, "safe_artifact": False},
        "read_file": {"readonly": True, "safe_artifact": False},
    }
    policy = derive_turn_tool_policy(
        "Do not use execute_shell or execute_python unless three structured-tool attempts fail.",
        set(metadata),
        metadata,
    )
    assert policy.delayed["execute_shell"] == 3
    assert policy.delayed["execute_python"] == 3
    for _ in range(2):
        assert not policy.allowed("execute_shell", metadata["execute_shell"])
        policy.record_iteration(False)
    assert not policy.allowed("execute_shell", metadata["execute_shell"])
    policy.record_iteration(False)
    assert policy.allowed("execute_shell", metadata["execute_shell"])


def test_http_probe_satisfies_explicit_dns_resolution_requirement():
    ledger = TaskRequirementLedger.from_request(
        "Resolve example.com and probe HTTPS connectivity to https://example.com"
    )
    assert {"dns_diagnose", "http_probe"} <= set(ledger.required_tools())
    ledger.record_tool("http_probe", status="ok", fingerprint="probe-one")
    assert ledger.status_for_tool("http_probe") == "satisfied"
    assert ledger.status_for_tool("dns_diagnose") == "satisfied"
    assert ledger.pending() == []


def test_blocked_requirement_is_closed_and_not_pending():
    ledger = TaskRequirementLedger.from_request("Inspect the network path to example.com")
    ledger.mark_blocked("network_path", "tool parser unavailable")
    assert ledger.status_for_tool("network_path") == "blocked"
    assert "network_path" in ledger.closed_tools()
    assert ledger.pending() == []


def test_pending_hint_is_ephemeral_style_and_lists_only_unfinished_checks():
    ledger = TaskRequirementLedger.from_request(
        "Inspect host CPU and memory and pressure state"
    )
    ledger.record_tool("host_snapshot", status="ok", fingerprint="host")
    hint = ledger.pending_hint()
    assert "pressure_snapshot" in hint
    assert "host_snapshot" not in hint
    assert "Do not draft the final report yet" in hint


def test_weather_completion_is_owned_by_fact_grounding_not_generic_tool_ledger():
    ledger = TaskRequirementLedger.from_request("What is the weather forecast for the next five days?")
    # Weather may be satisfied by the structured provider or by verified web
    # fallback, so the generic completion ledger must not hard-code one path.
    assert ledger.required_tools() == []


def test_display_forecast_is_evidence_reuse_followup_not_new_weather_requirement():
    text = "display the forecast"
    assert is_evidence_reuse_request(text) is True
    assert is_followup_request(text) is True
    assert TaskRequirementLedger.from_request(text).required_tools() == []


def test_local_network_scan_has_deterministic_discovery_requirements():
    ledger = TaskRequirementLedger.from_request(
        "scan the local network and subnets for hosts, then compile a list of hosts"
    )
    assert {"local_subnets", "scan_subnet"} <= set(ledger.required_tools())


def test_live_fact_prompts_are_distinct_from_implementation_prompts():
    pairs = (
        ("What is the weather in London ON?", "Refactor the weather validator and update its tests", "weather"),
        ("What time is it?", "Fix the current time tool", "current_time"),
        ("What are the latest headlines in London ON?", "Debug the latest-headlines formatter", "news"),
    )
    for live_prompt, implementation_prompt, expected in pairs:
        assert derive_task_frame(live_prompt).get("intent") == expected
        assert derive_task_frame(implementation_prompt).get("intent") is None
        assert TaskRequirementLedger.from_request(implementation_prompt).required_tools() == []


def test_local_news_followup_inherits_location_but_topical_news_does_not():
    first = derive_task_frame("What are the latest headlines in London ON?")
    assert first["entity"] == "London, Ontario, Canada"
    assert is_task_continuation("What are the latest local headlines?", first) is True
    followup = derive_task_frame("What are the latest local headlines?", first)
    assert followup["entity"] == first["entity"]
    assert build_news_query("What are the latest local headlines?", followup) == (
        "London, Ontario, Canada local latest news"
    )
    assert is_task_continuation("What are the latest AI headlines?", first) is False
    assert is_task_continuation("What are the latest headlines?", first) is False
    general = derive_task_frame("What are the latest headlines?", {}, default_location="London, Ontario, Canada")
    assert general == {"intent": "news", "time_scope": "latest"}
    assert build_news_query("What are the latest headlines?", general, "London, Ontario, Canada") == "latest news"


def test_market_requirement_ledger_uses_returned_rows_not_requested_arguments():
    import json

    request = "What is the current price of Brent crude and WTI?"
    ledger = TaskRequirementLedger.from_request(request)
    partial = json.dumps({
        "quotes": [{"instrument": "brent", "symbol": "BZ=F", "price": 97.43}],
        "errors": [{"instrument": "wti", "error": "provider unavailable"}],
    })
    ledger.record_tool(
        "market_quote", status="partial", reason="partial_result", arguments={"instruments": ["BZ=F", "CL=F"]},
        result_text=partial,
    )
    assert ledger.status_for_tool("market_quote") == "failed"

    complete = json.dumps({
        "quotes": [
            {"instrument": "brent", "symbol": "BZ=F", "price": 97.43},
            {"instrument": "wti", "symbol": "CL=F", "price": 93.95},
        ],
        "errors": [],
    })
    ledger.record_tool(
        "market_quote", status="ok", reason="ok", arguments={"instruments": ["BZ=F", "CL=F"]},
        result_text=complete,
    )
    assert ledger.status_for_tool("market_quote") == "satisfied"


def test_market_requirement_rejects_malformed_or_empty_result_provenance():
    ledger = TaskRequirementLedger.from_request("What are the current Brent and WTI prices?")
    assert "market_quote" in ledger.required_tools()
    ledger.record_tool(
        "market_quote", status="ok", fingerprint="bad-json",
        arguments={"instruments": ["BZ=F", "CL=F"]}, result_text="{not json",
    )
    assert ledger.status_for_tool("market_quote") != "satisfied"

    ledger = TaskRequirementLedger.from_request("What are the current Brent and WTI prices?")
    ledger.record_tool(
        "market_quote", status="ok", fingerprint="empty-meta",
        arguments={"instruments": ["BZ=F", "CL=F"]},
        result_metadata={"market_instruments": []},
    )
    assert ledger.status_for_tool("market_quote") != "satisfied"


def test_news_current_scope_accepts_latest_query_and_empty_completed_retrieval():
    from tools.task_requirements import TaskRequirementLedger

    ledger = TaskRequirementLedger.from_request("what are the local headlines in London ON?")
    ledger.record_tool(
        "news_search", status="ok", reason="empty_result",
        arguments={
            "query": "London, Ontario, Canada local latest news",
            "location": "London, Ontario, Canada",
            "timelimit": "d",
        },
        result_text="[]",
    )
    assert ledger.status_for_tool("news_search") == "satisfied"
    assert ledger.pending() == []
