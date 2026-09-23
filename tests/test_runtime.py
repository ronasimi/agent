import tempfile
from pathlib import Path


def test_runtime_job_lifecycle(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        db = str(Path(td) / "agent.db")
        monkeypatch.setenv("AGENT_DB_PATH", db)
        from tools import runtime
        runtime.DB_PATH = db
        runtime.init_runtime_db()
        job_id = runtime.create_job("research", "demo", {"topic": "demo"})
        job = runtime.claim_next_job("test-worker", ["research"])
        assert job["id"] == job_id
        assert runtime.save_checkpoint(job_id, {"phase": "search"})
        assert runtime.load_checkpoint(job_id)["phase"] == "search"
        assert runtime.complete_job(job_id, "done")
        assert runtime.get_job(job_id)["status"] == "completed"


def test_stale_job_is_recovered(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        db = str(Path(td) / "agent.db")
        monkeypatch.setenv("AGENT_DB_PATH", db)
        from tools import runtime
        runtime.DB_PATH = db
        runtime.init_runtime_db()
        job_id = runtime.create_job("research", "stale", {"topic": "stale"}, max_attempts=3)
        runtime.claim_next_job("worker-a", ["research"])
        with runtime._connect() as conn:
            conn.execute(
                "UPDATE agent_jobs SET heartbeat_at = '2000-01-01T00:00:00+00:00' WHERE id = ?",
                (job_id,),
            )
        assert runtime.recover_stale_jobs(30) == 1
        assert runtime.get_job(job_id)["status"] == "pending"


def test_stale_recovery_uses_separate_recovery_budget(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        db = str(Path(td) / "agent.db")
        monkeypatch.setenv("AGENT_DB_PATH", db)
        from tools import runtime
        runtime.DB_PATH = db
        runtime.init_runtime_db()
        job_id = runtime.create_job(
            "durable_compute", "stale-budget", {}, max_attempts=3, max_recovery_failures=1
        )
        runtime.claim_next_job("worker-a", ["durable_compute"])
        with runtime._connect() as conn:
            conn.execute(
                "UPDATE agent_jobs SET heartbeat_at = '2000-01-01T00:00:00+00:00' WHERE id = ?",
                (job_id,),
            )
        assert runtime.recover_stale_jobs(30) == 1
        job = runtime.get_job(job_id)
        assert job["status"] == "failed"
        assert job["attempts"] == 0
        assert job["recovery_failures"] == 1
        assert "Infrastructure recovery limit" in job["error"]


def test_singleton_job_and_foreground_deferral(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        db = str(Path(td) / "agent.db")
        monkeypatch.setenv("AGENT_DB_PATH", db)
        from tools import runtime
        runtime.DB_PATH = db
        runtime.init_runtime_db()
        first = runtime.create_singleton_job("context_compaction", "compact", {"through_id": 4})
        assert first
        assert runtime.create_singleton_job("context_compaction", "duplicate") is None
        claimed = runtime.claim_next_job("worker", ["context_compaction"])
        assert claimed["id"] == first
        assert runtime.defer_job(first, delay_seconds=1, state={"phase": "waiting"})
        deferred = runtime.get_job(first)
        assert deferred["status"] == "pending"
        assert deferred["attempts"] == 0
        assert deferred["state"] == {"phase": "waiting"}


def test_optimization_candidate_audit_lifecycle(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        db = str(Path(td) / "agent.db")
        monkeypatch.setenv("AGENT_DB_PATH", db)
        from tools import runtime
        runtime.DB_PATH = db
        runtime.init_runtime_db()
        job_id = runtime.create_job("self_optimization", "optimize", {"objective": "smaller context"})
        candidate_id = runtime.create_optimization_candidate(job_id, "smaller context", "prompt tokens")
        digest = "a" * 64
        assert runtime.update_optimization_candidate(
            candidate_id,
            status="awaiting_approval",
            patch_sha256=digest,
            baseline_json={"passed": True},
        )
        assert not runtime.approve_optimization_candidate(candidate_id, "b" * 64)
        assert runtime.approve_optimization_candidate(candidate_id, digest)
        candidate = runtime.get_optimization_candidate(candidate_id)
        assert candidate["status"] == "approved"
        assert candidate["baseline"] == {"passed": True}


def test_checkpoint_and_defer_is_atomic_and_does_not_consume_retry(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        db = str(Path(td) / "agent.db")
        monkeypatch.setenv("AGENT_DB_PATH", db)
        from tools import runtime
        runtime.DB_PATH = db
        runtime.init_runtime_db()
        job_id = runtime.create_job("durable_compute", "compute", {"program": {}})
        runtime.claim_next_job("worker", ["durable_compute"])
        state = {"checkpoint_version": 1, "checkpoint_generation": 1, "steps": 7}
        assert runtime.checkpoint_and_defer_job(job_id, state, step=1, delay_seconds=0)
        job = runtime.get_job(job_id)
        assert job["status"] == "pending"
        assert job["attempts"] == 0
        assert job["state"] == state
        assert runtime.load_checkpoint(job_id) == state


def test_cancelled_job_cannot_be_resurrected_by_late_compute_checkpoint(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        db = str(Path(td) / "agent.db")
        monkeypatch.setenv("AGENT_DB_PATH", db)
        from tools import runtime
        runtime.DB_PATH = db
        runtime.init_runtime_db()
        job_id = runtime.create_job("durable_compute", "race", {"program": {}})
        runtime.claim_next_job("worker", ["durable_compute"])
        assert runtime.cancel_job(job_id)
        state = {"checkpoint_version": 1, "checkpoint_generation": 1, "steps": 100}
        assert not runtime.checkpoint_and_defer_job(job_id, state, step=1, delay_seconds=0)
        assert runtime.get_job(job_id)["status"] == "cancelled"


def test_compute_tape_delta_and_checkpoint_commit_together(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        db = str(Path(td) / "agent.db")
        monkeypatch.setenv("AGENT_DB_PATH", db)
        from tools import runtime
        runtime.DB_PATH = db
        runtime.init_runtime_db()
        job_id = runtime.create_job("durable_compute", "sparse", {"program": {}})
        runtime.replace_compute_tape(job_id, {0: "A", 10_000: "FAR"})
        runtime.claim_next_job("worker", ["durable_compute"])
        state = {
            "checkpoint_version": 1,
            "checkpoint_generation": 1,
            "machine_state": "q",
            "head": 1,
            "steps": 2,
            "yield_count": 1,
            "status": "yielded",
            "tape_cells": 2,
            "tape": {"this": "must not be serialized"},
        }
        assert runtime.checkpoint_and_defer_job(
            job_id,
            state,
            step=1,
            delay_seconds=0,
            tape_updates={0: None, 1: "B"},
        )
        stored = runtime.get_job(job_id)["state"]
        assert "tape" not in stored
        assert runtime.load_checkpoint(job_id) == stored
        assert runtime.get_compute_tape_window(job_id, -1, 3) == {"1": "B"}
        assert runtime.get_compute_tape_window(job_id, 9_999, 10_001) == {"10000": "FAR"}


def test_infrastructure_recovery_has_a_separate_consecutive_budget(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        db = str(Path(td) / "agent.db")
        monkeypatch.setenv("AGENT_DB_PATH", db)
        from tools import runtime
        runtime.DB_PATH = db
        runtime.init_runtime_db()
        job_id = runtime.create_job(
            "durable_compute",
            "recoveries",
            {"program": {}},
            max_attempts=3,
            max_recovery_failures=2,
        )

        runtime.claim_next_job("worker-a", ["durable_compute"])
        assert runtime.recover_job_after_infrastructure_failure(job_id, "worker died", retry_delay_seconds=1)
        first = runtime.get_job(job_id)
        assert first["status"] == "pending"
        assert first["attempts"] == 0
        assert first["recovery_failures"] == 1

        with runtime._connect() as conn:
            conn.execute("UPDATE agent_jobs SET next_run_at = ? WHERE id = ?", (runtime.utc_now(), job_id))
        runtime.claim_next_job("worker-b", ["durable_compute"])
        assert runtime.recover_job_after_infrastructure_failure(job_id, "worker died again", retry_delay_seconds=1)
        second = runtime.get_job(job_id)
        assert second["status"] == "failed"
        assert second["attempts"] == 0
        assert second["recovery_failures"] == 2


def test_successful_compute_yield_resets_consecutive_recovery_failures(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        db = str(Path(td) / "agent.db")
        monkeypatch.setenv("AGENT_DB_PATH", db)
        from tools import runtime
        runtime.DB_PATH = db
        runtime.init_runtime_db()
        job_id = runtime.create_job("durable_compute", "reset", {"program": {}}, max_recovery_failures=5)
        with runtime._connect() as conn:
            conn.execute("UPDATE agent_jobs SET recovery_failures = 3 WHERE id = ?", (job_id,))
        runtime.claim_next_job("worker", ["durable_compute"])
        state = {"checkpoint_version": 1, "checkpoint_generation": 1, "steps": 1}
        assert runtime.checkpoint_and_defer_job(job_id, state, step=1, delay_seconds=0)
        assert runtime.get_job(job_id)["recovery_failures"] == 0


def test_zero_recovery_limit_explicitly_allows_unlimited_infrastructure_recovery(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        db = str(Path(td) / "agent.db")
        monkeypatch.setenv("AGENT_DB_PATH", db)
        from tools import runtime
        runtime.DB_PATH = db
        runtime.init_runtime_db()
        job_id = runtime.create_job("durable_compute", "unlimited-recovery", {}, max_recovery_failures=0)
        for expected in range(1, 6):
            with runtime._connect() as conn:
                conn.execute("UPDATE agent_jobs SET next_run_at = ? WHERE id = ?", (runtime.utc_now(), job_id))
            assert runtime.claim_next_job("worker", ["durable_compute"])
            assert runtime.recover_job_after_infrastructure_failure(job_id, "restart", retry_delay_seconds=1)
            job = runtime.get_job(job_id)
            assert job["status"] == "pending"
            assert job["attempts"] == 0
            assert job["recovery_failures"] == expected
