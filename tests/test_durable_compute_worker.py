from __future__ import annotations

import tempfile
from pathlib import Path

from al_agent.background.job_providers import p15_compute
from tools import runtime


def _scan_program():
    return {
        "initial_state": "scan",
        "halt_states": ["HALT"],
        "transitions": {
            "scan": {
                "1": {"write": "1", "move": "R", "next": "scan"},
                "_": {"write": "_", "move": "N", "next": "HALT"},
            }
        },
    }


def _forever_program():
    return {
        "initial_state": "run",
        "halt_states": ["HALT"],
        "transitions": {
            "run": {"_": {"write": "1", "move": "R", "next": "run"}}
        },
    }


def _db(monkeypatch):
    td = tempfile.TemporaryDirectory()
    db = str(Path(td.name) / "agent.db")
    monkeypatch.setenv("AGENT_DB_PATH", db)
    runtime.DB_PATH = db
    runtime.init_runtime_db()
    return td


def _claim_and_run(job_id: str):
    claimed = runtime.claim_next_job("test-worker", ["durable_compute"])
    assert claimed and claimed["id"] == job_id
    p15_compute.run_durable_compute_job(job_id, "test-worker")
    return runtime.get_job(job_id)


def test_durable_compute_resumes_until_halt_without_spending_retries(monkeypatch):
    td = _db(monkeypatch)
    try:
        monkeypatch.setattr(p15_compute, "DURABLE_COMPUTE_YIELD_DELAY_SECONDS", 0)
        job_id = runtime.create_job(
            "durable_compute",
            "scan",
            {"program": _scan_program(), "input_text": "11111", "quantum": 2},
        )
        first = _claim_and_run(job_id)
        assert first["status"] == "pending"
        assert first["attempts"] == 0
        assert first["state"]["steps"] == 2
        assert first["state"]["checkpoint_generation"] == 1
        assert "tape" not in first["state"]
        assert runtime.get_compute_tape_window(job_id, 0, 8) == {
            "0": "1", "1": "1", "2": "1", "3": "1", "4": "1"
        }

        while runtime.get_job(job_id)["status"] != "completed":
            _claim_and_run(job_id)
        finished = runtime.get_job(job_id)
        assert finished["state"]["steps"] == 6
        assert finished["state"]["yield_count"] == 2
        assert finished["state"]["checkpoint_generation"] == 3
    finally:
        td.cleanup()


def test_durable_compute_recovers_latest_checkpoint_after_stale_claim(monkeypatch):
    td = _db(monkeypatch)
    try:
        monkeypatch.setattr(p15_compute, "DURABLE_COMPUTE_YIELD_DELAY_SECONDS", 0)
        job_id = runtime.create_job(
            "durable_compute",
            "recover",
            {"program": _scan_program(), "input_text": "1111", "quantum": 2},
        )
        yielded = _claim_and_run(job_id)
        assert yielded["state"]["steps"] == 2

        claimed = runtime.claim_next_job("worker-before-crash", ["durable_compute"])
        assert claimed and claimed["id"] == job_id
        with runtime._connect() as conn:
            conn.execute(
                "UPDATE agent_jobs SET heartbeat_at = '2000-01-01T00:00:00+00:00' WHERE id = ?",
                (job_id,),
            )
        assert runtime.recover_stale_jobs(30) == 1
        assert runtime.load_checkpoint(job_id)["steps"] == 2

        resumed = _claim_and_run(job_id)
        assert resumed["state"]["steps"] == 4
    finally:
        td.cleanup()


def test_non_halting_durable_compute_runs_across_many_claims_until_cancelled(monkeypatch):
    td = _db(monkeypatch)
    try:
        monkeypatch.setattr(p15_compute, "DURABLE_COMPUTE_YIELD_DELAY_SECONDS", 0)
        job_id = runtime.create_job(
            "durable_compute",
            "forever",
            {"program": _forever_program(), "quantum": 3},
        )
        for _ in range(25):
            job = _claim_and_run(job_id)
            assert job["status"] == "pending"
            assert job["attempts"] == 0
        job = runtime.get_job(job_id)
        assert job["state"]["steps"] == 75
        assert job["state"]["yield_count"] == 25
        assert "tape" not in job["state"]
        assert runtime.get_compute_tape_cell_count(job_id) == 75
        assert runtime.cancel_job(job_id)
        assert runtime.get_job(job_id)["status"] == "cancelled"
    finally:
        td.cleanup()


def test_optional_step_policy_is_explicit_and_terminal(monkeypatch):
    td = _db(monkeypatch)
    try:
        monkeypatch.setattr(p15_compute, "DURABLE_COMPUTE_YIELD_DELAY_SECONDS", 0)
        job_id = runtime.create_job(
            "durable_compute",
            "bounded",
            {"program": _forever_program(), "quantum": 10, "max_steps": 7},
        )
        job = _claim_and_run(job_id)
        assert job["status"] == "pending"
        assert job["state"]["steps"] == 7
        job = _claim_and_run(job_id)
        assert job["status"] == "failed"
        assert "max_steps" in job["error"]
    finally:
        td.cleanup()


def test_legacy_inline_tape_checkpoint_migrates_to_sparse_table(monkeypatch):
    td = _db(monkeypatch)
    try:
        monkeypatch.setattr(p15_compute, "DURABLE_COMPUTE_YIELD_DELAY_SECONDS", 0)
        job_id = runtime.create_job(
            "durable_compute",
            "legacy-inline",
            {"program": _scan_program(), "quantum": 10},
        )
        legacy_state = {
            "checkpoint_version": 1,
            "machine_state": "scan",
            "head": 2,
            "steps": 2,
            "yield_count": 1,
            "checkpoint_generation": 1,
            "status": "yielded",
            "tape": {"0": "1", "1": "1", "2": "1"},
            "tape_cells": 3,
            "last_quantum_transitions": 2,
            "last_error": None,
        }
        import json
        with runtime._connect() as conn:
            conn.execute(
                "INSERT INTO agent_job_checkpoints(job_id, step, state_json, created_at) VALUES (?, ?, ?, ?)",
                (job_id, 1, json.dumps(legacy_state), runtime.utc_now()),
            )
            conn.execute(
                "UPDATE agent_jobs SET state_json = ? WHERE id = ?",
                (json.dumps(legacy_state), job_id),
            )

        finished = _claim_and_run(job_id)
        assert finished["status"] == "completed"
        assert "tape" not in finished["state"]
        assert runtime.get_compute_tape_window(job_id, 0, 4) == {"0": "1", "1": "1", "2": "1"}
    finally:
        td.cleanup()
