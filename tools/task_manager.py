"""Compatibility task API backed by the durable runtime job store."""
from __future__ import annotations

import json
from enum import Enum

from .runtime import (
    create_job,
    get_job,
    list_jobs,
    load_checkpoint,
    save_checkpoint as _save_checkpoint,
    cancel_job,
    complete_job,
    fail_job,
)


class TaskState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


def init_task_manager_db() -> None:
    from .runtime import init_runtime_db
    init_runtime_db()


def create_task(name: str, description: str = "", priority: int = 0, dependencies: list | None = None, state: dict | None = None) -> str:
    """Create a durable generic task and return its job ID."""
    return create_job(
        "generic",
        str(name),
        payload={"description": description, "dependencies": dependencies or []},
        priority=priority,
    )


def save_checkpoint(task_id: str, state: dict) -> bool:
    """Persist the latest task checkpoint."""
    return _save_checkpoint(task_id, state)


def get_task_state(task_id: str) -> dict:
    """Return the latest durable task state."""
    return load_checkpoint(task_id)


def update_task_status(task_id: str, status: TaskState, error: str = None) -> bool:
    """Update a generic task's status."""
    status = TaskState(status)
    if status == TaskState.COMPLETED:
        return complete_job(task_id, "")
    if status == TaskState.FAILED:
        return fail_job(task_id, error or "Task failed.", retry=False)
    if status == TaskState.CANCELLED:
        return cancel_job(task_id)
    job = get_job(task_id)
    if not job:
        return False
    from .runtime import _connect, utc_now
    with _connect() as conn:
        cur = conn.execute("UPDATE agent_jobs SET status=?, updated_at=?, error=? WHERE id=?", (status.value, utc_now(), error, task_id))
    return cur.rowcount == 1


def log_task(message: str, task_id: str, level: str = "info") -> None:
    """Record a task message as a monitor event."""
    from .runtime import record_monitor_event
    record_monitor_event(f"task_{level}", str(message), {"task_id": task_id})


def list_tasks(status: str = None) -> str:
    """List durable tasks using the legacy task-manager response shape."""
    rows = list_jobs(status=status or "", limit=100)
    return json.dumps(rows, ensure_ascii=False, indent=2) if rows else f"No {status or 'any'} tasks"


def get_task_logs(task_id: str, limit: int = 20) -> str:
    """Return monitor events associated with a task."""
    from .runtime import list_monitor_events
    events = [e for e in list_monitor_events(limit=100) if e.get("details", {}).get("task_id") == task_id]
    return json.dumps(events[:max(1, min(int(limit), 100))][::-1], ensure_ascii=False, indent=2) if events else "No logs for this task"


def resume_interrupted_task(task_id: str) -> dict:
    """Return the latest checkpoint state for a task."""
    return load_checkpoint(task_id)


def get_task_info(task_id: str) -> str:
    """Return detailed durable task information."""
    job = get_job(task_id)
    return json.dumps(job, ensure_ascii=False, indent=2) if job else f"Error: task {task_id} not found"


def delete_task(task_id: str) -> str:
    """Delete a task from the durable runtime store."""
    from .runtime import _connect
    with _connect() as conn:
        conn.execute("DELETE FROM agent_jobs WHERE id=?", (task_id,))
    return f"Task {task_id[:8]} deleted"


init_task_manager_db()
