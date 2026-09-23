from __future__ import annotations

import json
import tempfile
from pathlib import Path

from al_agent.background.job_providers import p15_compute
from al_agent.compute import inputs as compute_inputs
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
        assert status["policy"]["max_recovery_failures"] == 10
        assert status["recovery_failures"] == 0
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


def test_start_computation_accepts_sparse_initial_tape_and_head(monkeypatch):
    td = _db(monkeypatch)
    try:
        program = {
            "initial_state": "read",
            "halt_states": ["HALT"],
            "transitions": {
                "read": {"TOKEN": {"write": "DONE", "move": "N", "next": "HALT"}}
            },
        }
        created = json.loads(
            start_computation(program, initial_tape={"-2": "TOKEN"}, initial_head=-2)
        )
        claimed = runtime.claim_next_job("worker", ["durable_compute"])
        assert claimed and claimed["id"] == created["job_id"]
        p15_compute.run_durable_compute_job(created["job_id"], "worker")
        job = runtime.get_job(created["job_id"])
        assert job["status"] == "completed"
        assert job["state"]["head"] == -2
        assert runtime.get_compute_tape_window(created["job_id"], -3, 0) == {"-2": "DONE"}
    finally:
        td.cleanup()


def test_workspace_input_file_is_hash_pinned_and_loaded_outside_model_context(monkeypatch):
    td = _db(monkeypatch)
    try:
        workspace = Path(td.name) / "workspace"
        workspace.mkdir()
        monkeypatch.setattr(compute_inputs, "WORKSPACE_ROOT", workspace)
        source = workspace / "input.txt"
        source.write_text("111", encoding="utf-8")
        created = json.loads(start_computation(_program() | {
            "transitions": {
                "run": {
                    "1": {"write": "1", "move": "R", "next": "run"},
                    "_": {"write": "_", "move": "N", "next": "HALT"},
                }
            }
        }, input_file="input.txt", input_file_format="text"))
        payload = runtime.get_job(created["job_id"])["payload"]
        assert payload["input_file"]["sha256"]
        assert "111" not in json.dumps(payload["input_file"])

        claimed = runtime.claim_next_job("worker", ["durable_compute"])
        assert claimed and claimed["id"] == created["job_id"]
        p15_compute.run_durable_compute_job(created["job_id"], "worker")
        assert runtime.get_job(created["job_id"])["status"] == "completed"
        assert runtime.get_compute_tape_window(created["job_id"], 0, 4) == {"0": "1", "1": "1", "2": "1"}
    finally:
        td.cleanup()


def test_workspace_input_file_change_is_rejected_before_execution(monkeypatch):
    td = _db(monkeypatch)
    try:
        workspace = Path(td.name) / "workspace"
        workspace.mkdir()
        monkeypatch.setattr(compute_inputs, "WORKSPACE_ROOT", workspace)
        source = workspace / "input.txt"
        source.write_text("A", encoding="utf-8")
        created = json.loads(start_computation(_program(), input_file="input.txt"))
        source.write_text("B", encoding="utf-8")
        runtime.claim_next_job("worker", ["durable_compute"])
        try:
            p15_compute.run_durable_compute_job(created["job_id"], "worker")
            raise AssertionError("changed external input should have been rejected")
        except Exception as exc:
            assert "changed after the computation was queued" in str(exc)
    finally:
        td.cleanup()
