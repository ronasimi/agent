import json
from datetime import datetime, timezone
from pathlib import Path

from tools import get_tools_prompt_summary, select_tool_schemas
from tools.host_tools import host_snapshot
from tools.primitives import current_time, environment_summary, hostname
from tools.task_requirements import TaskRequirementLedger


def _tool_names(schemas):
    return {str(schema.get("function", {}).get("name") or "") for schema in schemas}


def test_current_time_returns_structured_current_clock():
    before = datetime.now(timezone.utc).timestamp()
    payload = json.loads(current_time())
    after = datetime.now(timezone.utc).timestamp()

    stamp = datetime.fromisoformat(payload["utc"]).timestamp()
    assert before - 1 <= stamp <= after + 1
    assert payload["timezone"]
    assert payload["local"]
    assert payload["date"]
    assert payload["time"]
    assert payload["day_of_week"]
    assert isinstance(payload["unix_timestamp"], int)


def test_time_request_forces_current_time_without_shell():
    names = _tool_names(select_tool_schemas("What time is it?", max_tools=12))
    assert "current_time" in names
    assert "execute_shell" not in names
    assert "execute_python" not in names
    assert "enqueue_self_optimization" not in names


def test_current_time_is_present_even_with_tiny_schema_budget():
    names = _tool_names(select_tool_schemas("What time is it?", max_tools=1))
    assert names == {"current_time"}


def test_direct_time_request_creates_completion_requirement():
    for request in ("What is the current time?", "What day is it?", "What timezone am I in?", "UTC time please"):
        ledger = TaskRequirementLedger.from_request(request)
        assert "current_time" in ledger.required_tools()


def test_hostname_is_structured_and_bounded():
    payload = json.loads(hostname())
    assert payload["host_hostname"]
    assert payload["runtime_hostname"]
    assert isinstance(payload["same_hostname"], bool)


def test_environment_summary_does_not_dump_arbitrary_environment(monkeypatch):
    monkeypatch.setenv("AGENT_TEST_SUPER_SECRET", "do-not-leak-this-value")
    payload_text = environment_summary()
    payload = json.loads(payload_text)
    assert "do-not-leak-this-value" not in payload_text
    assert payload["clock"]["timezone"]
    assert payload["main_model"]
    assert payload["fast_model"]
    assert payload["context_tokens"] >= 1


def test_host_snapshot_has_observation_timestamps():
    payload = json.loads(host_snapshot())
    assert datetime.fromisoformat(payload["observed_at"]).tzinfo is not None
    assert datetime.fromisoformat(payload["observed_at_local"]).tzinfo is not None
    assert payload["timezone"]


def test_compact_tool_policy_mentions_withheld_capabilities():
    text = get_tools_prompt_summary(compact=True)
    assert "execute_shell" in text
    assert "schema absence does not prove" in text


def test_system_policy_forbids_stale_clock_inference():
    source = Path("al_agent/runtime.py").read_text(encoding="utf-8")
    assert "Never infer the current clock from uptime" in source
    assert "Their absence from the currently supplied schemas" in source


def test_runtime_services_mount_host_localtime():
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")
    assert compose.count("/etc/localtime:/etc/localtime:ro") >= 2
