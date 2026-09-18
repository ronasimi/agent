import json
from pathlib import Path

import pytest

from tools import AVAILABLE_TOOLS_MAP, select_tool_schemas
from tools.loop_validator import classify_tool_outcome
from tools.network_diagnostics import _safe_host
from tools.turn_policy import derive_turn_tool_policy


def test_all_registered_tools_selectable_by_literal_name():
    for name in AVAILABLE_TOOLS_MAP:
        selected = {schema["function"]["name"] for schema in select_tool_schemas(name, max_tools=12)}
        assert name in selected, name


def test_new_tool_set_is_registered():
    expected = {
        "process_snapshot", "pressure_snapshot", "filesystem_snapshot", "service_health",
        "neighbor_snapshot", "connection_snapshot", "dns_diagnose", "network_path",
        "endpoint_probe", "http_probe", "page_metadata", "page_links", "discover_site",
        "read_feed", "extract_document", "page_fingerprint", "page_diff",
        "repo_status", "repo_diff", "repo_checks", "dependency_audit", "tool_health",
        "diff_observations",
    }
    assert expected <= set(AVAILABLE_TOOLS_MAP)


def test_partial_result_is_progress_not_failure():
    outcome = classify_tool_outcome("Partial: command exited with status 1 but produced usable stdout.\nSTDOUT:\n/path")
    assert outcome["success"] is True
    assert outcome["status"] == "partial"
    assert outcome["reason"] == "nonzero_with_output"


def test_conditional_tool_policy_hides_then_unlocks_shell():
    policy = derive_turn_tool_policy(
        "Do not use execute_shell unless your first approach fails.",
        set(AVAILABLE_TOOLS_MAP),
        {name: {"readonly": True} for name in AVAILABLE_TOOLS_MAP} | {"execute_shell": {"readonly": False}},
    )
    assert not policy.allowed("execute_shell", {"readonly": False})
    assert policy.record_iteration(False) is True
    assert policy.allowed("execute_shell", {"readonly": False})


def test_read_only_policy_blocks_mutating_tools():
    metadata = {
        "read_file": {"readonly": True},
        "write_file": {"readonly": False},
        "execute_shell": {"readonly": False},
    }
    policy = derive_turn_tool_policy("Read-only: do not modify any files.", set(metadata), metadata)
    assert policy.allowed("read_file", metadata["read_file"])
    assert not policy.allowed("write_file", metadata["write_file"])
    assert not policy.allowed("execute_shell", metadata["execute_shell"])


def test_safe_host_rejects_command_like_input():
    assert _safe_host("example.com") == "example.com"
    assert _safe_host("192.168.1.1") == "192.168.1.1"
    with pytest.raises(ValueError):
        _safe_host("example.com; rm -rf /")


def test_read_only_question_does_not_trigger_turn_lockdown():
    metadata = {"read_file": {"readonly": True}, "write_file": {"readonly": False}}
    policy = derive_turn_tool_policy("Why is this filesystem read-only?", set(metadata), metadata)
    assert policy.readonly_only is False
    assert policy.allowed("write_file", metadata["write_file"])
