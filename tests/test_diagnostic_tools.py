import json
from pathlib import Path

import pytest

from tools import AVAILABLE_TOOLS_MAP, select_tool_schemas
from tools.loop_validator import classify_tool_outcome
from tools.network_diagnostics import _safe_host


def test_all_registered_tools_selectable_by_literal_name():
    from al_agent.tool_session import ToolSession
    import tools
    session = ToolSession(tools.TOOL_SCHEMAS, lambda *a: None, tools.TOOL_METADATA)
    for name in tools.AVAILABLE_TOOLS_MAP:
        loaded = session.invoke("load_tools", {"names": [name]})
        expected = [] if name in {"tool_search", "load_tools"} else [name]
        assert loaded["activated"] == expected
        assert "schemas" not in loaded


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






def test_safe_host_rejects_command_like_input():
    assert _safe_host("example.com") == "example.com"
    assert _safe_host("192.168.1.1") == "192.168.1.1"
    with pytest.raises(ValueError):
        _safe_host("example.com; rm -rf /")




def test_mtr_text_report_fallback_is_structured():
    from tools.network_diagnostics import _parse_mtr_report_text

    sample = """Start: 2026-09-18T11:23:08-0400
HOST: muninn                      Loss%   Snt   Last   Avg  Best  Wrst StDev
  1.|-- _gateway                   0.0%     3    1.3   1.3   1.3   1.4   0.1
  2.|-- dhcp-198-2-106-1.example   0.0%     3   15.2  14.9  14.1  15.2   0.6
"""
    payload = _parse_mtr_report_text(sample, "example.com")
    assert payload is not None
    assert payload["format"] == "mtr_report_text_fallback"
    assert payload["source_host"] == "muninn"
    assert payload["hops"][0]["hop"] == 1
    assert payload["hops"][0]["host"] == "_gateway"
    assert payload["hops"][1]["avg_ms"] == 14.9


def test_web_search_result_filter_rejects_ad_redirects():
    from tools.web import _search_result_url_allowed

    assert _search_result_url_allowed("https://weather.gc.ca/en/location/index.html") is True
    assert _search_result_url_allowed("https://www.bing.com/aclick?x=1") is False
    assert _search_result_url_allowed("javascript:alert(1)") is False


def test_network_mapper_autodetects_private_subnet(monkeypatch):
    from tools import network_mapper

    monkeypatch.setattr(network_mapper, "_local_ipv4_networks", lambda include_virtual=False: [
        {"interface": "eth0", "address": "192.168.50.10", "prefixlen": 24, "network": "192.168.50.0/24", "virtual": False, "operstate": "UP"}
    ])
    assert network_mapper._resolve_network("") == ("192.168.50.0/24", "")
    network, error = network_mapper._resolve_network("8.8.8.0/24")
    assert network == ""
    assert "private/link-local" in error


def test_scan_subnet_returns_structured_hosts_without_creating_map(monkeypatch):
    from tools import network_mapper

    monkeypatch.setattr(network_mapper, "_resolve_network", lambda network: ("192.168.1.0/24", ""))
    monkeypatch.setattr(network_mapper, "_scan_hosts", lambda network: ["192.168.1.1", "192.168.1.2"])
    monkeypatch.setattr(network_mapper, "_get_os_info", lambda host, top_ports=50: {
        "ip": host, "hostname": f"host-{host.rsplit('.', 1)[-1]}", "open_ports": []
    })
    payload = json.loads(network_mapper.scan_subnet("192.168.1.0/24", max_detail_hosts=32, top_ports=50))
    assert payload["network"] == "192.168.1.0/24"
    assert payload["host_count"] == 2
    assert payload["detailed_hosts"] == 2
    assert [row["ip"] for row in payload["hosts"]] == ["192.168.1.1", "192.168.1.2"]
    assert "map_path" not in payload
