"""Durable runtime state for long-running agent jobs.

The runtime uses a single SQLite database as the source of truth for jobs,
checkpoints, heartbeats, and monitor state.  SQLite WAL mode allows the
interactive agent and the worker process to operate concurrently.
"""
from __future__ import annotations

import json
import os
import socket
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional

DB_PATH = os.environ.get("AGENT_DB_PATH", "/app/memory/knowledge.db")
DB_TIMEOUT = float(os.environ.get("AGENT_DB_TIMEOUT", "15"))

_SCHEMA_LOCK = threading.Lock()


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=15000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_runtime_db() -> None:
    """Create or migrate all durable runtime tables."""
    with _SCHEMA_LOCK, _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS agent_jobs (
                id TEXT PRIMARY KEY,
                job_type TEXT NOT NULL,
                title TEXT NOT NULL,
                payload_json TEXT NOT NULL DEFAULT '{}',
                state_json TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'pending',
                priority INTEGER NOT NULL DEFAULT 0,
                attempts INTEGER NOT NULL DEFAULT 0,
                max_attempts INTEGER NOT NULL DEFAULT 3,
                worker_id TEXT,
                heartbeat_at TEXT,
                next_run_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                result TEXT,
                error TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_agent_jobs_claim
              ON agent_jobs(status, priority DESC, next_run_at, created_at);
            CREATE INDEX IF NOT EXISTS idx_agent_jobs_heartbeat
              ON agent_jobs(status, heartbeat_at);

            CREATE TABLE IF NOT EXISTS agent_job_checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                step INTEGER NOT NULL,
                state_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(job_id) REFERENCES agent_jobs(id) ON DELETE CASCADE,
                UNIQUE(job_id, step)
            );
            CREATE INDEX IF NOT EXISTS idx_job_checkpoints_job
              ON agent_job_checkpoints(job_id, step DESC);

            CREATE TABLE IF NOT EXISTS monitor_state (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS monitor_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                summary TEXT NOT NULL,
                details_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_monitor_events_type_time
              ON monitor_events(event_type, created_at DESC);

            CREATE TABLE IF NOT EXISTS reminders (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                message TEXT NOT NULL DEFAULT '',
                when_iso TEXT NOT NULL,
                repeat_mode TEXT NOT NULL DEFAULT 'once',
                unit_name TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL DEFAULT 'scheduled',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_error TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_reminders_status_when
              ON reminders(status, when_iso);

            CREATE TABLE IF NOT EXISTS optimization_candidates (
                id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL UNIQUE,
                objective TEXT NOT NULL,
                target_metric TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'building',
                worktree_path TEXT,
                patch_path TEXT,
                patch_sha256 TEXT,
                baseline_json TEXT NOT NULL DEFAULT '{}',
                candidate_json TEXT NOT NULL DEFAULT '{}',
                report_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                approved_at TEXT,
                FOREIGN KEY(job_id) REFERENCES agent_jobs(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_optimization_candidates_status_time
              ON optimization_candidates(status, created_at DESC);
            """
        )


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _parse_json(value: Optional[str], fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return fallback


def create_job(
    job_type: str,
    title: str,
    payload: Optional[dict[str, Any]] = None,
    priority: int = 0,
    max_attempts: int = 3,
    run_at: Optional[str] = None,
) -> str:
    """Create a durable job and return its UUID."""
    init_runtime_db()
    job_id = str(uuid.uuid4())
    now = utc_now()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO agent_jobs (
                id, job_type, title, payload_json, state_json, status,
                priority, attempts, max_attempts, next_run_at,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, '{}', ?, ?, 0, ?, ?, ?, ?)
            """,
            (
                job_id,
                str(job_type),
                str(title),
                _json(payload or {}),
                JobStatus.PENDING.value,
                int(priority),
                max(1, int(max_attempts)),
                run_at or now,
                now,
                now,
            ),
        )
    return job_id


def create_singleton_job(
    job_type: str,
    title: str,
    payload: Optional[dict[str, Any]] = None,
    priority: int = 0,
    max_attempts: int = 3,
    singleton_key: str = "",
) -> Optional[str]:
    """Create one pending/running job per type, optionally scoped by a stable key."""
    init_runtime_db()
    job_id = str(uuid.uuid4())
    now = utc_now()
    with _connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        if singleton_key:
            scoped_title = f"{str(title)} [{str(singleton_key)}]"
            existing = conn.execute(
                "SELECT id FROM agent_jobs WHERE job_type = ? AND title = ? AND status IN ('pending', 'running') LIMIT 1",
                (str(job_type), scoped_title),
            ).fetchone()
            stored_title = scoped_title
        else:
            existing = conn.execute(
                "SELECT id FROM agent_jobs WHERE job_type = ? AND status IN ('pending', 'running') LIMIT 1",
                (str(job_type),),
            ).fetchone()
            stored_title = str(title)
        if existing:
            conn.commit()
            return None
        conn.execute(
            """
            INSERT INTO agent_jobs (
                id, job_type, title, payload_json, state_json, status,
                priority, attempts, max_attempts, next_run_at,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, '{}', ?, ?, 0, ?, ?, ?, ?)
            """,
            (
                job_id,
                str(job_type),
                stored_title,
                _json(payload or {}),
                JobStatus.PENDING.value,
                int(priority),
                max(1, int(max_attempts)),
                now,
                now,
                now,
            ),
        )
        conn.commit()
    return job_id


def get_job(job_id: str) -> Optional[dict[str, Any]]:
    init_runtime_db()
    with _connect() as conn:
        row = conn.execute("SELECT * FROM agent_jobs WHERE id = ?", (job_id,)).fetchone()
    if not row:
        return None
    item = dict(row)
    item["payload"] = _parse_json(item.pop("payload_json"), {})
    item["state"] = _parse_json(item.pop("state_json"), {})
    return item



def find_active_job_by_idempotency(job_type: str, idempotency_key: str) -> Optional[dict[str, Any]]:
    """Return a pending/running job with a matching payload idempotency key."""
    key = str(idempotency_key or "").strip()
    if not key:
        return None
    init_runtime_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM agent_jobs WHERE job_type = ? AND status IN ('pending', 'running') ORDER BY created_at ASC",
            (str(job_type),),
        ).fetchall()
    for row in rows:
        item = dict(row)
        payload = _parse_json(item.get("payload_json"), {})
        if str(payload.get("idempotency_key") or "") != key:
            continue
        item["payload"] = payload
        item["state"] = _parse_json(item.pop("state_json", "{}"), {})
        item.pop("payload_json", None)
        return item
    return None

def list_jobs(status: str = "", limit: int = 25) -> list[dict[str, Any]]:
    init_runtime_db()
    limit = max(1, min(int(limit), 100))
    with _connect() as conn:
        if status:
            rows = conn.execute(
                """
                SELECT id, job_type, title, status, priority, attempts, max_attempts,
                       worker_id, next_run_at, created_at, updated_at, started_at,
                       completed_at, result, error, state_json
                FROM agent_jobs
                WHERE status = ?
                ORDER BY priority DESC, created_at ASC
                LIMIT ?
                """,
                (status, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT id, job_type, title, status, priority, attempts, max_attempts,
                       worker_id, next_run_at, created_at, updated_at, started_at,
                       completed_at, result, error, state_json
                FROM agent_jobs
                ORDER BY
                  CASE status WHEN 'running' THEN 0 WHEN 'pending' THEN 1 ELSE 2 END,
                  priority DESC, created_at ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        state = _parse_json(item.pop("state_json", "{}"), {})
        if item.get("job_type") == "durable_compute":
            item["progress"] = {
                "machine_status": state.get("status") or ("not_started" if not state else "unknown"),
                "machine_state": state.get("machine_state"),
                "steps": int(state.get("steps", 0) or 0),
                "yield_count": int(state.get("yield_count", 0) or 0),
                "checkpoint_generation": int(state.get("checkpoint_generation", 0) or 0),
                "head": int(state.get("head", 0) or 0),
                "tape_cells": int(state.get("tape_cells", 0) or 0),
            }
        out.append(item)
    return out


def claim_next_job(worker_id: str, allowed_types: Optional[list[str]] = None) -> Optional[dict[str, Any]]:
    """Atomically claim the next due job for one worker."""
    init_runtime_db()
    worker_id = str(worker_id or socket.gethostname())
    now = utc_now()
    with _connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        where = "status = 'pending' AND (next_run_at IS NULL OR next_run_at <= ?)"
        params: list[Any] = [now]
        if allowed_types:
            placeholders = ",".join("?" for _ in allowed_types)
            where += f" AND job_type IN ({placeholders})"
            params.extend(allowed_types)
        row = conn.execute(
            f"""
            SELECT id
            FROM agent_jobs
            WHERE {where}
            ORDER BY priority DESC, created_at ASC
            LIMIT 1
            """,
            params,
        ).fetchone()
        if not row:
            conn.commit()
            return None
        job_id = row[0]
        conn.execute(
            """
            UPDATE agent_jobs
            SET status = ?, worker_id = ?, attempts = attempts + 1,
                heartbeat_at = ?, started_at = COALESCE(started_at, ?),
                updated_at = ?, error = NULL
            WHERE id = ? AND status = 'pending'
            """,
            (JobStatus.RUNNING.value, worker_id, now, now, now, job_id),
        )
        conn.commit()
    return get_job(job_id)


def heartbeat_job(job_id: str, worker_id: str, state: Optional[dict[str, Any]] = None) -> bool:
    now = utc_now()
    with _connect() as conn:
        if state is None:
            cur = conn.execute(
                "UPDATE agent_jobs SET heartbeat_at = ?, updated_at = ? WHERE id = ? AND status = 'running' AND worker_id = ?",
                (now, now, job_id, worker_id),
            )
        else:
            cur = conn.execute(
                """
                UPDATE agent_jobs
                SET heartbeat_at = ?, updated_at = ?, state_json = ?
                WHERE id = ? AND status = 'running' AND worker_id = ?
                """,
                (now, now, _json(state), job_id, worker_id),
            )
    return cur.rowcount == 1


def save_checkpoint(job_id: str, state: dict[str, Any], step: Optional[int] = None) -> bool:
    init_runtime_db()
    state = state or {}
    with _connect() as conn:
        if step is None:
            row = conn.execute(
                "SELECT COALESCE(MAX(step), 0) + 1 FROM agent_job_checkpoints WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            step = int(row[0] or 1)
        now = utc_now()
        conn.execute(
            "INSERT OR REPLACE INTO agent_job_checkpoints (job_id, step, state_json, created_at) VALUES (?, ?, ?, ?)",
            (job_id, int(step), _json(state), now),
        )
        conn.execute(
            "UPDATE agent_jobs SET state_json = ?, updated_at = ? WHERE id = ?",
            (_json(state), now, job_id),
        )
    return True



def _next_checkpoint_step(conn: sqlite3.Connection, job_id: str, requested: Optional[int]) -> int:
    """Resolve a monotonic checkpoint step inside an existing transaction."""
    if requested is not None:
        return max(1, int(requested))
    row = conn.execute(
        "SELECT COALESCE(MAX(step), 0) + 1 FROM agent_job_checkpoints WHERE job_id = ?",
        (job_id,),
    ).fetchone()
    return int(row[0] or 1)


def checkpoint_and_defer_job(
    job_id: str,
    state: dict[str, Any],
    *,
    step: Optional[int] = None,
    delay_seconds: float = 1.0,
) -> bool:
    """Atomically checkpoint healthy progress and release a running job.

    A cooperative yield is not a retry/failure, so the claim attempt consumed by
    this execution slice is returned.  Persisting the checkpoint and changing
    the queue state happen in one SQLite transaction to avoid exposing a newer
    queue state with an older machine checkpoint after a process crash.
    """
    init_runtime_db()
    now = datetime.now(timezone.utc)
    next_run = (now + timedelta(seconds=max(0.0, float(delay_seconds)))).isoformat(timespec="seconds")
    now_iso = now.isoformat(timespec="seconds")
    with _connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT status FROM agent_jobs WHERE id = ?", (job_id,)).fetchone()
        if not row or row[0] != JobStatus.RUNNING.value:
            conn.rollback()
            return False
        checkpoint_step = _next_checkpoint_step(conn, job_id, step)
        conn.execute(
            "INSERT OR REPLACE INTO agent_job_checkpoints (job_id, step, state_json, created_at) VALUES (?, ?, ?, ?)",
            (job_id, checkpoint_step, _json(state or {}), now_iso),
        )
        cur = conn.execute(
            """
            UPDATE agent_jobs
            SET status = ?, worker_id = NULL, heartbeat_at = NULL,
                attempts = MAX(0, attempts - 1), next_run_at = ?,
                updated_at = ?, state_json = ?, error = NULL
            WHERE id = ? AND status = ?
            """,
            (JobStatus.PENDING.value, next_run, now_iso, _json(state or {}), job_id, JobStatus.RUNNING.value),
        )
        if cur.rowcount != 1:
            conn.rollback()
            return False
        conn.commit()
    return True


def checkpoint_and_complete_job(
    job_id: str,
    state: dict[str, Any],
    result: str = "",
    *,
    step: Optional[int] = None,
) -> bool:
    """Atomically persist the terminal checkpoint and mark a job completed."""
    init_runtime_db()
    now = utc_now()
    with _connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT status FROM agent_jobs WHERE id = ?", (job_id,)).fetchone()
        if not row or row[0] != JobStatus.RUNNING.value:
            conn.rollback()
            return False
        checkpoint_step = _next_checkpoint_step(conn, job_id, step)
        conn.execute(
            "INSERT OR REPLACE INTO agent_job_checkpoints (job_id, step, state_json, created_at) VALUES (?, ?, ?, ?)",
            (job_id, checkpoint_step, _json(state or {}), now),
        )
        cur = conn.execute(
            """
            UPDATE agent_jobs
            SET status = ?, result = ?, completed_at = ?, heartbeat_at = NULL,
                worker_id = NULL, updated_at = ?, state_json = ?, error = NULL
            WHERE id = ? AND status = ?
            """,
            (JobStatus.COMPLETED.value, str(result), now, now, _json(state or {}), job_id, JobStatus.RUNNING.value),
        )
        if cur.rowcount != 1:
            conn.rollback()
            return False
        conn.commit()
    return True


def checkpoint_and_fail_job(
    job_id: str,
    state: dict[str, Any],
    error: str,
    *,
    step: Optional[int] = None,
) -> bool:
    """Atomically persist a terminal machine failure without retrying it."""
    init_runtime_db()
    now = utc_now()
    with _connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT status FROM agent_jobs WHERE id = ?", (job_id,)).fetchone()
        if not row or row[0] != JobStatus.RUNNING.value:
            conn.rollback()
            return False
        checkpoint_step = _next_checkpoint_step(conn, job_id, step)
        conn.execute(
            "INSERT OR REPLACE INTO agent_job_checkpoints (job_id, step, state_json, created_at) VALUES (?, ?, ?, ?)",
            (job_id, checkpoint_step, _json(state or {}), now),
        )
        cur = conn.execute(
            """
            UPDATE agent_jobs
            SET status = ?, heartbeat_at = NULL, worker_id = NULL, updated_at = ?,
                state_json = ?, error = ?
            WHERE id = ? AND status = ?
            """,
            (JobStatus.FAILED.value, now, _json(state or {}), str(error), job_id, JobStatus.RUNNING.value),
        )
        if cur.rowcount != 1:
            conn.rollback()
            return False
        conn.commit()
    return True

def load_checkpoint(job_id: str) -> dict[str, Any]:
    init_runtime_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT state_json FROM agent_job_checkpoints WHERE job_id = ? ORDER BY step DESC LIMIT 1",
            (job_id,),
        ).fetchone()
    if not row:
        job = get_job(job_id)
        return (job or {}).get("state", {})
    return _parse_json(row[0], {})


def complete_job(job_id: str, result: str = "") -> bool:
    now = utc_now()
    with _connect() as conn:
        cur = conn.execute(
            """
            UPDATE agent_jobs
            SET status = ?, result = ?, completed_at = ?, heartbeat_at = NULL,
                updated_at = ?, error = NULL
            WHERE id = ? AND status = 'running'
            """,
            (JobStatus.COMPLETED.value, str(result), now, now, job_id),
        )
    return cur.rowcount == 1


def fail_job(job_id: str, error: str, retry: bool = True, retry_delay_seconds: int = 30) -> bool:
    now = datetime.now(timezone.utc)
    with _connect() as conn:
        row = conn.execute(
            "SELECT attempts, max_attempts FROM agent_jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
        if not row:
            return False
        attempts, max_attempts = int(row[0]), int(row[1])
        can_retry = retry and attempts < max_attempts
        if can_retry:
            next_run = (now + timedelta(seconds=max(1, int(retry_delay_seconds)))).isoformat(timespec="seconds")
            cur = conn.execute(
                """
                UPDATE agent_jobs
                SET status = ?, next_run_at = ?, heartbeat_at = NULL,
                    updated_at = ?, error = ?
                WHERE id = ? AND status = 'running'
                """,
                (JobStatus.PENDING.value, next_run, now.isoformat(timespec="seconds"), str(error), job_id),
            )
        else:
            cur = conn.execute(
                """
                UPDATE agent_jobs
                SET status = ?, heartbeat_at = NULL, updated_at = ?, error = ?
                WHERE id = ? AND status = 'running'
                """,
                (JobStatus.FAILED.value, now.isoformat(timespec="seconds"), str(error), job_id),
            )
    return cur.rowcount == 1


def defer_job(job_id: str, delay_seconds: int = 3, state: Optional[dict[str, Any]] = None) -> bool:
    """Release a running job without consuming another retry after foreground contention."""
    now = datetime.now(timezone.utc)
    next_run = (now + timedelta(seconds=max(1, int(delay_seconds)))).isoformat(timespec="seconds")
    with _connect() as conn:
        if state is None:
            cursor = conn.execute(
                """
                UPDATE agent_jobs
                SET status = 'pending', worker_id = NULL, heartbeat_at = NULL,
                    attempts = MAX(0, attempts - 1), next_run_at = ?, updated_at = ?
                WHERE id = ? AND status = 'running'
                """,
                (next_run, now.isoformat(timespec="seconds"), job_id),
            )
        else:
            cursor = conn.execute(
                """
                UPDATE agent_jobs
                SET status = 'pending', worker_id = NULL, heartbeat_at = NULL,
                    attempts = MAX(0, attempts - 1), next_run_at = ?, updated_at = ?, state_json = ?
                WHERE id = ? AND status = 'running'
                """,
                (next_run, now.isoformat(timespec="seconds"), _json(state), job_id),
            )
    return cursor.rowcount == 1


def cancel_job(job_id: str) -> bool:
    now = utc_now()
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE agent_jobs SET status = ?, heartbeat_at = NULL, updated_at = ? WHERE id = ? AND status IN ('pending', 'running')",
            (JobStatus.CANCELLED.value, now, job_id),
        )
    return cur.rowcount == 1


def recover_stale_jobs(stale_after_seconds: int = 180) -> int:
    """Return stale running jobs to pending after a process crash."""
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=max(30, int(stale_after_seconds)))
    cutoff_iso = cutoff.isoformat(timespec="seconds")
    now = utc_now()
    with _connect() as conn:
        cur = conn.execute(
            """
            UPDATE agent_jobs
            SET status = ?, worker_id = NULL, heartbeat_at = NULL,
                next_run_at = ?, updated_at = ?, error = COALESCE(error, 'Recovered after stale worker heartbeat.')
            WHERE status = 'running'
              AND (heartbeat_at IS NULL OR heartbeat_at < ?)
            """,
            (JobStatus.PENDING.value, now, now, cutoff_iso),
        )
    return cur.rowcount


def record_monitor_state(key: str, value: Any) -> None:
    now = utc_now()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO monitor_state(key, value_json, updated_at) VALUES(?, ?, ?) ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_at=excluded.updated_at",
            (str(key), _json(value), now),
        )


def get_monitor_state(key: str, default: Any = None) -> Any:
    with _connect() as conn:
        row = conn.execute("SELECT value_json FROM monitor_state WHERE key = ?", (str(key),)).fetchone()
    return _parse_json(row[0], default) if row else default


def record_monitor_event(event_type: str, summary: str, details: Optional[dict[str, Any]] = None) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO monitor_events(event_type, summary, details_json, created_at) VALUES (?, ?, ?, ?)",
            (str(event_type), str(summary), _json(details or {}), utc_now()),
        )


def list_monitor_events(event_type: str = "", limit: int = 25) -> list[dict[str, Any]]:
    limit = max(1, min(int(limit), 100))
    with _connect() as conn:
        if event_type:
            rows = conn.execute(
                "SELECT id, event_type, summary, details_json, created_at FROM monitor_events WHERE event_type = ? ORDER BY id DESC LIMIT ?",
                (event_type, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, event_type, summary, details_json, created_at FROM monitor_events ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
    out = []
    for row in rows:
        item = dict(row)
        item["details"] = _parse_json(item.pop("details_json"), {})
        out.append(item)
    return out


def create_optimization_candidate(
    job_id: str,
    objective: str,
    target_metric: str = "",
    candidate_id: str | None = None,
) -> str:
    """Create the durable audit record for one isolated optimization candidate."""
    init_runtime_db()
    candidate_id = candidate_id or str(uuid.uuid4())
    now = utc_now()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO optimization_candidates(
                id, job_id, objective, target_metric, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, 'building', ?, ?)
            """,
            (candidate_id, str(job_id), str(objective), str(target_metric), now, now),
        )
    return candidate_id


def update_optimization_candidate(candidate_id: str, **fields: Any) -> bool:
    """Update only explicitly permitted fields on a candidate audit record."""
    allowed = {
        "status", "worktree_path", "patch_path", "patch_sha256",
        "baseline_json", "candidate_json", "report_json", "approved_at",
    }
    updates: list[str] = []
    values: list[Any] = []
    for key, value in fields.items():
        if key not in allowed:
            raise ValueError(f"Unsupported candidate field: {key}")
        if key.endswith("_json") and not isinstance(value, str):
            value = _json(value)
        updates.append(f"{key} = ?")
        values.append(value)
    if not updates:
        return False
    updates.append("updated_at = ?")
    values.extend([utc_now(), str(candidate_id)])
    with _connect() as conn:
        cur = conn.execute(
            f"UPDATE optimization_candidates SET {', '.join(updates)} WHERE id = ?",
            values,
        )
    return cur.rowcount == 1


def _candidate_row(row: sqlite3.Row | None) -> Optional[dict[str, Any]]:
    if not row:
        return None
    item = dict(row)
    for key in ("baseline_json", "candidate_json", "report_json"):
        item[key.removesuffix("_json")] = _parse_json(item.pop(key), {})
    return item


def get_optimization_candidate(candidate_id: str) -> Optional[dict[str, Any]]:
    init_runtime_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM optimization_candidates WHERE id = ?",
            (str(candidate_id),),
        ).fetchone()
    return _candidate_row(row)


def list_optimization_candidates(status: str = "", limit: int = 20) -> list[dict[str, Any]]:
    init_runtime_db()
    limit = max(1, min(int(limit), 100))
    with _connect() as conn:
        if status:
            rows = conn.execute(
                "SELECT * FROM optimization_candidates WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                (str(status), limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM optimization_candidates ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
    return [_candidate_row(row) for row in rows if row]


def approve_optimization_candidate(candidate_id: str, expected_sha256: str) -> bool:
    """Approve an awaiting candidate only when its displayed patch digest matches."""
    now = utc_now()
    with _connect() as conn:
        cur = conn.execute(
            """
            UPDATE optimization_candidates
            SET status = 'approved', approved_at = ?, updated_at = ?
            WHERE id = ? AND status = 'awaiting_approval' AND patch_sha256 = ?
            """,
            (now, now, str(candidate_id), str(expected_sha256).lower()),
        )
    return cur.rowcount == 1


def maintain_runtime(retention_days: int = 30, checkpoints_per_job: int = 25) -> dict[str, int]:
    """Prune bounded ephemeral records and checkpoint the WAL during worker idle time."""
    retention_days = max(1, min(int(retention_days), 3650))
    checkpoints_per_job = max(1, min(int(checkpoints_per_job), 1000))
    deleted = {"monitor_events": 0, "tool_observations": 0, "checkpoints": 0}
    cutoff = f"-{retention_days} days"
    with _connect() as conn:
        deleted["monitor_events"] = conn.execute(
            "DELETE FROM monitor_events WHERE created_at < datetime('now', ?)", (cutoff,)
        ).rowcount
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "tool_observations" in tables:
            deleted["tool_observations"] = conn.execute(
                "DELETE FROM tool_observations WHERE created_at < datetime('now', ?)", (cutoff,)
            ).rowcount
        deleted["checkpoints"] = conn.execute(
            """
            DELETE FROM agent_job_checkpoints
            WHERE id IN (
                SELECT older.id
                FROM agent_job_checkpoints AS older
                WHERE (
                    SELECT COUNT(*) FROM agent_job_checkpoints AS newer
                    WHERE newer.job_id = older.job_id AND newer.step >= older.step
                ) > ?
            )
            """,
            (checkpoints_per_job,),
        ).rowcount
    with _connect() as conn:
        conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
    return deleted


init_runtime_db()
