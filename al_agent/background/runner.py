"""Generic durable worker loop."""
from __future__ import annotations

import os
import socket
import time
import traceback

from tools.runtime import (
    DB_PATH,
    claim_next_job,
    defer_job,
    fail_job,
    get_job,
    init_runtime_db,
    maintain_runtime,
    recover_stale_jobs,
)
from tools.self_optimization import mark_self_optimization_failed

from .config import (
    HEARTBEAT_SECONDS,
    INTERACTIVE_COOLDOWN,
    MAINTENANCE_INTERVAL,
    MONITOR_INTERVAL,
    POLL_SECONDS,
    STALE_SECONDS,
    WORKER_CFG,
)
from .handlers import allowed_job_types, run_job
from .maintenance import monitor_once
from .resources import InferenceDeferred, _interactive_busy, _notify


def main() -> None:
    try:
        os.nice(5)
    except OSError:
        pass
    init_runtime_db()
    recovered = recover_stale_jobs(STALE_SECONDS)
    if recovered:
        print(f"[worker] recovered {recovered} stale job(s)")
    worker_id = f"{socket.gethostname()}:{os.getpid()}"
    last_monitor = 0.0
    last_heartbeat = 0.0
    last_maintenance = 0.0
    print(f"[worker] started as {worker_id}; database={DB_PATH}")

    while True:
        now = time.monotonic()
        if now - last_monitor >= MONITOR_INTERVAL:
            monitor_once()
            last_monitor = now

        if now - last_maintenance >= MAINTENANCE_INTERVAL and not _interactive_busy():
            try:
                maintain_runtime(
                    retention_days=int(WORKER_CFG.get("ephemeral_retention_days", 30)),
                    checkpoints_per_job=int(WORKER_CFG.get("checkpoints_per_job", 25)),
                )
            except Exception as exc:
                print(f"[worker] maintenance error: {exc}")
            last_maintenance = now

        if now - last_heartbeat >= HEARTBEAT_SECONDS:
            last_heartbeat = now
        try:
            job = claim_next_job(worker_id, allowed_types=allowed_job_types())
            if not job:
                time.sleep(POLL_SECONDS)
                continue
            print(f"[worker] claimed {job['id'][:8]}: {job['title']}")
            try:
                run_job(str(job.get("job_type") or ""), job["id"], worker_id)
            except InferenceDeferred:
                state = (get_job(job["id"]) or {}).get("state") or {}
                defer_job(job["id"], delay_seconds=max(2, int(INTERACTIVE_COOLDOWN)), state=state)
            except Exception as exc:
                detail = f"{exc}\n{traceback.format_exc(limit=5)}"
                print(f"[worker] job {job['id'][:8]} failed: {exc}")
                if job.get("job_type") == "self_optimization":
                    candidate_id = str((job.get("payload") or {}).get("candidate_id") or "")
                    mark_self_optimization_failed(candidate_id, detail)
                retry = int(job.get("attempts", 1)) < int(job.get("max_attempts", 3))
                fail_job(
                    job["id"], detail, retry=retry,
                    retry_delay_seconds=min(300, 30 * int(job.get("attempts", 1))),
                )
                if not retry:
                    _notify("Agent Job Failed", f"{job['title']}: {exc}")
        except KeyboardInterrupt:
            print("[worker] stopped")
            return
        except Exception as exc:
            print(f"[worker] loop error: {exc}")
            time.sleep(POLL_SECONDS)
