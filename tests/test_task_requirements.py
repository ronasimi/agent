from tools.task_requirements import (
    TaskRequirementLedger, build_news_query, derive_task_frame, is_evidence_reuse_request,
    is_followup_request, is_task_continuation,
)


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


def test_weather_requirement_is_exposed_but_any_grounded_weather_path_can_close_it():
    ledger = TaskRequirementLedger.from_request("What is the weather forecast for the next five days?")
    assert ledger.required_tools() == ["weather_forecast"]
    # The hard grounding gate owns evidence validity. Once it proves weather via
    # a recipe/API/web fallback, the tool ledger closes the weather capability so
    # the main model cannot redundantly call weather_forecast again.
    ledger.mark_fact_satisfied("weather")
    assert ledger.pending() == []


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




def test_weather_entity_stops_at_following_report_sentence():
    from tools.task_requirements import derive_task_frame

    frame = derive_task_frame("Determine the current weather for London, Ontario.\n Report:")
    assert frame["entity"] == "London Ontario"

def test_effective_request_uses_fact_specific_source_for_compound_prompt():
    from tools.task_requirements import derive_fact_frames, effective_request_for_frame

    request = """Tasks:\n1. Get the current weather for London, Ontario.\n2. Get the latest local London, Ontario headlines.\n3. Get the current Brent crude price."""
    frame = derive_fact_frames(request, default_location="London, ON")["weather"]
    effective = effective_request_for_frame(request, frame)
    assert "Get the current weather for London, Ontario." in effective
    assert "headlines" not in effective.lower()
    assert "brent" not in effective.lower()


def test_scheduler_runtime_and_registry_steps_compile_explicit_requirements():
    from tools.task_requirements import derive_requirements

    runtime = (
        "1. Runtime Identity: Determine using runtime/system tools: hostname, operating system, "
        "kernel version, architecture, uptime, current local time, configured timezone. "
        "Verify each value from tool output."
    )
    tools = {item.tool for item in derive_requirements(runtime)}
    assert {"hostname", "environment_summary", "uptime", "current_time"}.issubset(tools)

    registry = (
        "2. Tool Registry: Inspect the available tool registry. Report total registered tools, "
        "healthy tools, degraded tools, unavailable tools, and missing dependencies."
    )
    assert [item.tool for item in derive_requirements(registry)] == ["tool_health"]

    host = (
        "4. Host Snapshot: Collect a read-only host snapshot. Report CPU, load average, memory, "
        "uptime, filesystem usage, kernel, and hostname."
    )
    host_tools = {item.tool for item in derive_requirements(host)}
    assert "host_snapshot" in host_tools
    assert "filesystem_snapshot" in host_tools
    assert "hostname" in host_tools

    cpu = "5. CPU: Use the dedicated CPU tool. Report model, architecture, logical CPU count, and physical cores."
    memory = "6. Memory: Use the dedicated memory primitive. Report total memory, available memory, used memory, and swap."
    assert "cpu_info" in {item.tool for item in derive_requirements(cpu)}
    assert "memory_info" in {item.tool for item in derive_requirements(memory)}


def test_sectioned_capability_stress_prompt_compiles_all_24_requirements():
    from tools.task_requirements import derive_requirements, detect_fact_frame_types

    request = '''EXECUTION RULES
1. Use the most specific available tool.
2. Do not repeat equivalent failed calls indefinitely.

REMOTE / INTERNET TOOLS
1. Determine the current weather for London, Ontario.
2. Retrieve exactly 3 of the latest local London, Ontario news headlines.
3. Retrieve the current/latest available Brent crude oil price.
4. Retrieve https://example.com and report HTTP status, title, and canonical URL.

SYSTEM TOOLS
5. Determine the current system/local time using a system capability.
6. Report basic operating-system information: kernel/system name, kernel release, architecture, hostname.
7. Perform a harmless shell/system execution test and verify HARNESS_SYSTEM_TOOL_OK.
8. Inspect available filesystem information and report free space for the agent workspace filesystem.

HOST TOOLS
9. Obtain a host snapshot and report uptime, total memory, available memory, load, and root filesystem utilization.
10. Report CPU identity/model and logical CPU count.
11. Report host thermal information if available; check temperature sensors.
12. Determine whether Ollama is currently running on the host.

GOOGLE ACCOUNT TOOLS
13. Gmail: determine access and report the 3 most recent inbox messages.
14. Google Calendar: report the next 3 upcoming calendar events.
15. Google Drive: list the 3 most recently modified files.

NETWORK TOOLS
16. Resolve example.com with the available DNS/network tool.
17. Probe TCP connectivity to example.com port 443.
18. Perform an HTTPS probe against https://example.com.
19. Test DNS resolution for harness-stress-test-invalid.example and treat NXDOMAIN as PASS.
20. Check localhost connectivity to http://127.0.0.1:11434/api/version.

CROSS-CAPABILITY CONSISTENCY TESTS
21. Compare the system time result with timestamps returned by at least one current remote source.
22. Verify that the HTTP page retrieval result for example.com and the network HTTPS probe agree about basic reachability.
23. Verify that every successful requirement has actual tool evidence.
24. Check the observations generated during this test for truncation warnings and recover any middle truncation.

FINAL OUTPUT
Return the requested sectioned report.'''

    requirements = derive_requirements(request)
    assert len(requirements) == 24
    assert [item.key for item in requirements] == [f"stress:{i:02d}" for i in range(1, 25)]
    tools = [item.tool for item in requirements]
    assert tools.count("dns_query") == 2
    assert tools.count("http_probe") == 2
    assert tools[14] == "google_drive_list_files"
    assert tools[20:24] == [
        "__derived_time_consistency__",
        "__derived_reachability_consistency__",
        "__derived_evidence_audit__",
        "__derived_truncation_audit__",
    ]
    assert detect_fact_frame_types(request) == {"weather", "news", "market_price", "current_time"}
    # Execution rules are not compiled as tasks.
    assert all("Use the most specific" not in str(item.scope.get("source_text") or "") for item in requirements)


def test_duplicate_primitives_track_scoped_requirements_independently():
    from tools.task_requirements import Requirement, TaskRequirementLedger

    ledger = TaskRequirementLedger([
        Requirement("one", "dns_query", "example", scope={"target": "example.com"}),
        Requirement("two", "dns_query", "negative", scope={"target": "invalid.example"}),
    ])
    ledger.record_tool(
        "dns_query", status="ok", reason="ok",
        arguments={"name": "example.com", "record_type": "A"},
        result_text='{"status":"NOERROR","answers":["example.com. A 1.2.3.4"]}',
    )
    assert ledger.requirements[0].status == "satisfied"
    assert ledger.requirements[0].attempts == 1
    assert ledger.requirements[1].status == "pending"
    assert ledger.requirements[1].attempts == 0


def test_direct_requirement_retains_durable_evidence_reference_and_excerpt():
    ledger = TaskRequirementLedger.from_request("What time is it?")
    ledger.record_tool(
        "current_time",
        status="ok",
        reason="ok",
        fingerprint="clock-fingerprint",
        arguments={},
        result_text='{"local":"2026-09-23T16:22:36-04:00"}',
        evidence_ref="observation-clock-1",
    )
    row = ledger.as_list()[0]
    assert row["status"] == "satisfied"
    assert row["evidence"]
    evidence = row["evidence"][-1]
    assert evidence["source"] == "tool_call"
    assert evidence["tool"] == "current_time"
    assert evidence["evidence_ref"] == "observation-clock-1"
    assert "2026-09-23" in evidence["evidence_preview"]
    assert evidence["arguments_digest"]


def test_ui_requirements_are_derived_and_only_closed_by_verifier():
    from tools.task_requirements import TaskRequirementLedger

    ledger = TaskRequirementLedger([])
    checks = [{"type": "text_present", "value": "Saved"}]
    ledger.ensure_ui_requirements(checks)
    assert "browser_step" not in ledger.required_tools()
    assert len(ledger.pending()) == 2  # outcome + explicit predicate

    # Ordinary successful browser actions cannot satisfy independent UI proof.
    ledger.record_tool("browser_step", status="ok", arguments={"op": "click"}, result_text='{"ok":true}')
    assert len(ledger.pending()) == 2

    ledger.record_ui_verification({
        "passed": True,
        "checks": [{"check": checks[0], "passed": True, "actual": "Saved"}],
    })
    assert ledger.pending() == []

    ledger.invalidate_ui_outcome()
    assert len(ledger.pending()) == 2


def test_generic_weather_followups_inherit_location_instead_of_question_words():
    previous = derive_task_frame(
        "What's the weather in London, Ontario?",
        default_location="London, Ontario, Canada",
    )
    for request in (
        "what is the current weather?",
        "what is the weather?",
        "weather right now",
        "what's the forecast?",
        "forecast please",
    ):
        frame = derive_task_frame(request, previous, default_location="London, Ontario, Canada")
        assert frame["intent"] == "weather"
        assert frame["entity"] == previous["entity"]

    tomorrow = derive_task_frame("What about tomorrow?", previous, default_location="London, Ontario, Canada")
    assert tomorrow["entity"] == previous["entity"]
    assert tomorrow["time_scope"] == "tomorrow"

    toronto = derive_task_frame("And Toronto?", tomorrow, default_location="London, Ontario, Canada")
    assert toronto["entity"] == "Toronto"
    assert toronto["time_scope"] == "tomorrow"


def test_tool_selection_audit_is_not_reparsed_as_live_fact_work():
    from tools.grounding import requested_fact_types
    from tools.task_requirements import (
        derive_fact_frames,
        derive_requirements,
        derive_task_frame,
        is_nonexecuting_tool_selection_request,
    )

    request = (
        "3. Tool Selection: For each of these intents, identify the most appropriate primitive "
        "without executing it yet: * current time * CPU information * memory usage * network routes "
        "* DNS lookup * current weather * public webpage retrieval * repository status "
        "* reading a local file * parsing JSON * running a calculation * historical conversation recall."
    )
    assert is_nonexecuting_tool_selection_request(request)
    assert derive_task_frame(request) == {}
    assert derive_fact_frames(request) == {}
    assert requested_fact_types(request) == set()
    assert derive_requirements(request) == []


def test_target_extraction_rejects_instruction_prose_and_strips_markdown():
    from tools.task_requirements import _extract_target

    assert _extract_target("dns_diagnose", "Do not probe systems except through ordinary public HTTP/DNS requests.") == ""
    assert _extract_target("network_path", "Network Path: Use a safe path/traceroute-style primitive if available.") == ""
    assert _extract_target("dns_diagnose", "Resolve example.com using the DNS primitive.") == "example.com"
    assert _extract_target("network_path", "Use the network path to example.com and report the hops.") == "example.com"
    assert _extract_target("browse_url", "Fetch `https://example.com` and report the title.") == "https://example.com"


def test_cpu_architecture_request_requires_environment_evidence():
    tools = set(TaskRequirementLedger.from_request(
        "Use the dedicated CPU tool. Report model, architecture, logical CPU count, and frequency."
    ).required_tools())
    assert "cpu_info" in tools
    assert "environment_summary" in tools


def test_browser_semantic_observation_requires_browser_step():
    text = (
        "Semantic Observation: Identify at least three interactive elements semantically. "
        "For each report stable reference, role, accessible name, and state."
    )
    assert "browser_step" in TaskRequirementLedger.from_request(text).required_tools()


def test_browser_session_requires_browser_step():
    text = "Browser Session: Open a safe public demo page and report the active tab and state version."
    assert "browser_step" in TaskRequirementLedger.from_request(text).required_tools()


def test_routes_compile_to_deterministic_route_list_and_lookup_requirements():
    """The stress-suite route step should not require model-based tool routing."""
    ledger = TaskRequirementLedger.from_request(
        "Inspect the routing table. Identify the default route, default gateway, and interface used by the "
        "default route. Cross-check using route lookup against a harmless public address."
    )
    assert ledger.required_tools() == ["route_list", "route_lookup"]
    lookup = next(item for item in ledger.requirements if item.tool == "route_lookup")
    assert lookup.scope.get("target") == "8.8.8.8"
