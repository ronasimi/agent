from __future__ import annotations

import json
import tempfile
from pathlib import Path

from tools import runtime
from tools.job_tools import cancel_computation, get_computation_status, start_computation
from tools.loop_validator import classify_tool_outcome


def _program():
    return {
        "initial_state": "run",
        "halt_states": ["HALT"],
        "transitions": {
            "run": {"_": {"write": "1", "move": "N", "next": "HALT"}}
        },
    }


def _db(monkeypatch):
    td = tempfile.TemporaryDirectory()
    db = str(Path(td.name) / "agent.db")
    monkeypatch.setenv("AGENT_DB_PATH", db)
    runtime.DB_PATH = db
    runtime.init_runtime_db()
    return td


def test_start_status_cancel_and_active_creation_deduplication(monkeypatch):
    td = _db(monkeypatch)
    try:
        first = json.loads(start_computation(_program(), input_text=""))
        second = json.loads(start_computation(_program(), input_text=""))
        assert first["job_id"] == second["job_id"]
        assert first["deduplicated"] is False
        assert second["deduplicated"] is True
        assert first["unbounded"] is True

        status = json.loads(get_computation_status(first["job_id"], tape_cells=8))
        assert status["runtime_status"] == "pending"
        assert status["machine_status"] == "not_started"
        assert status["tape_window"]["end_exclusive"] - status["tape_window"]["start"] == 8

        cancelled = json.loads(cancel_computation(first["job_id"]))
        assert cancelled["status"] == "cancelled"
        assert runtime.get_job(first["job_id"])["status"] == "cancelled"
    finally:
        td.cleanup()


def test_compute_tools_are_structured_validator_successes():
    for tool, payload in (
        ("start_computation", {"job_id": "abc", "status": "pending"}),
        ("get_computation_status", {"job_id": "abc", "runtime_status": "pending"}),
        ("cancel_computation", {"job_id": "abc", "status": "cancelled"}),
    ):
        outcome = classify_tool_outcome(json.dumps(payload), tool_name=tool)
        assert outcome["success"] is True
        assert outcome["status"] == "ok"


def test_compute_routing_exposes_durable_tools_without_generic_execution():
    from tools.catalog import select_tool_schemas

    names = {
        item["function"]["name"]
        for item in select_tool_schemas("start an unbounded durable computation that runs until halt", max_tools=12)
    }
    assert {"start_computation", "get_computation_status", "cancel_computation"} <= names
    assert "execute_shell" not in names
