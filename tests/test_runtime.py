import sys
import tempfile
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PKG = types.ModuleType("tools")
PKG.__path__ = [str(REPO / "tools")]
sys.modules.setdefault("tools", PKG)


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
