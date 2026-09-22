"""Generic durable worker loop."""
from __future__ import annotations

import concurrent.futures
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
    heartbeat_job,
    init_runtime_db,
    maintain_runtime,
    recover_stale_jobs,
)
from tools.self_optimization import mark_self_optimization_failed

from .config import (
    HEARTBEAT_SECONDS,
    INTERACTIVE_COOLDOWN,
    MAINTENANCE_INTERVAL,
    MAX_JOB_RUNTIME_SECONDS,
    MONITOR_INTERVAL,
    POLL_SECONDS,
    STALE_SECONDS,
    WORKER_CFG,
)
from .handlers import allowed_job_types, run_job
from .maintenance import monitor_once
from .resources import InferenceDeferred, _interactive_busy, _notify


def _run_maintenance_if_due(last_maintenance: float, now: float) -> float:
    if now - last_maintenance < MAINTENANCE_INTERVAL or _interactive_busy():
        return last_maintenance
    try:
        maintain_runtime(
            retention_days=int(WORKER_CFG.get("ephemeral_retention_days", 30)),
            checkpoints_per_job=int(WORKER_CFG.get("checkpoints_per_job", 25)),
        )
    except Exception as exc:
        print(f"[worker] maintenance error: {exc}")
    return now


def _run_monitor_if_due(last_monitor: float, now: float) -> float:
    if now - last_monitor < MONITOR_INTERVAL:
        return last_monitor
    monitor_once()
    return now


def _timeout_job_and_restart(job: dict, worker_id: str) -> None:
    """Fail/requeue a wedged job, then terminate the worker process.

    Python threads cannot be forcibly stopped safely.  Exiting the dedicated
    worker container is the only reliable way to terminate a job thread that
    ignored all lower-level timeouts.  Compose restarts the worker cleanly.
    """
    detail = (
        f"Background job exceeded the configured {MAX_JOB_RUNTIME_SECONDS:g}-second runtime ceiling; "
        "the worker is restarting to terminate the stuck execution context."
    )
    print(f"[worker] job {job['id'][:8]} timed out: {detail}")
    if job.get("job_type") == "self_optimization":
        candidate_id = str((job.get("payload") or {}).get("candidate_id") or "")
        mark_self_optimization_failed(candidate_id, detail)
    retry = int(job.get("attempts", 1)) < int(job.get("max_attempts", 3))
    fail_job(
        job["id"], detail, retry=retry,
        retry_delay_seconds=min(300, 30 * int(job.get("attempts", 1))),
    )
    if not retry:
        _notify("Agent Job Failed", f"{job['title']}: {detail}")
    # os._exit is deliberate: normal interpreter shutdown waits for executor
    # threads and would hang on the very thread we are trying to terminate.
    os._exit(70)


def _run_job_supervised(
    job: dict,
    worker_id: str,
    *,
    last_monitor: float,
    last_maintenance: float,
) -> tuple[float, float]:
    """Run a claimed job while the worker keeps heartbeat/maintenance alive."""
    started = time.monotonic()
    last_heartbeat = 0.0
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="background-job")
    future = executor.submit(run_job, str(job.get("job_type") or ""), job["id"], worker_id)
    try:
        while True:
            try:
                # Short polling keeps maintenance responsive without spinning.
                future.result(timeout=min(1.0, max(0.1, HEARTBEAT_SECONDS / 2.0)))
                return last_monitor, last_maintenance
            except concurrent.futures.TimeoutError:
                now = time.monotonic()
                last_monitor = _run_monitor_if_due(last_monitor, now)
                last_maintenance = _run_maintenance_if_due(last_maintenance, now)
                if now - last_heartbeat >= HEARTBEAT_SECONDS:
                    heartbeat_job(job["id"], worker_id)
                    last_heartbeat = now
                if now - started >= MAX_JOB_RUNTIME_SECONDS:
                    _timeout_job_and_restart(job, worker_id)
                    raise RuntimeError("worker restart did not terminate process")
    finally:
        # On the normal/exception path the job thread is complete, so waiting is
        # safe. The timeout path exits via os._exit before reaching shutdown.
        executor.shutdown(wait=True, cancel_futures=True)


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
    last_maintenance = 0.0
    print(f"[worker] started as {worker_id}; database={DB_PATH}")

    while True:
        now = time.monotonic()
        last_monitor = _run_monitor_if_due(last_monitor, now)
        last_maintenance = _run_maintenance_if_due(last_maintenance, now)
        try:
            job = claim_next_job(worker_id, allowed_types=allowed_job_types())
            if not job:
                time.sleep(POLL_SECONDS)
                continue
            print(f"[worker] claimed {job['id'][:8]}: {job['title']}")
            try:
                last_monitor, last_maintenance = _run_job_supervised(
                    job,
                    worker_id,
                    last_monitor=last_monitor,
                    last_maintenance=last_maintenance,
                )
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
