from tools.task_requirements import TaskRequirementLedger, is_followup_request
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
