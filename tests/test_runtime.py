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
