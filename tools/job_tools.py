"""Agent-visible durable job management tools."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from al_agent.compute.machine import MachineProgramError, validate_program

from .runtime import (
    cancel_job,
    create_job,
    find_active_job_by_idempotency,
    get_job,
    list_jobs,
)


def enqueue_research(topic: str = "", priority: int = 0) -> str:
    """Queue a durable deep-research job that continues across browser sessions and restarts."""
    topic = str(topic).strip()
    if not topic:
        return "Error: Missing required 'topic' parameter."
    job_id = create_job(
        "research",
        f"Research: {topic}",
        payload={"topic": topic},
        priority=max(-10, min(int(priority), 10)),
        max_attempts=3,
    )
    return json.dumps({"job_id": job_id, "status": "pending", "topic": topic}, indent=2)


def get_research_status(job_id: str = "") -> str:
    """Get detailed status, checkpoint state, attempts, errors, and result path for a research job."""
    if not str(job_id).strip():
        return "Error: Missing required 'job_id' parameter."
    job = get_job(job_id)
    if not job:
        return f"Error: job '{job_id}' not found."
    return json.dumps(job, ensure_ascii=False, indent=2)


def list_background_jobs(status: str = "", limit: int = 25) -> str:
    """List durable background jobs; optionally filter by pending, running, completed, failed, or cancelled."""
    allowed = {"", "pending", "running", "completed", "failed", "cancelled"}
    if status not in allowed:
        return f"Error: status must be one of {', '.join(sorted(allowed - {''}))}."
    return json.dumps(list_jobs(status=status, limit=limit), ensure_ascii=False, indent=2)


def cancel_background_job(job_id: str = "") -> str:
    """Cancel a pending or running durable background job."""
    if not str(job_id).strip():
        return "Error: Missing required 'job_id' parameter."
    return "Job cancelled." if cancel_job(job_id) else "Error: job could not be cancelled; it may not exist or may already be finished."



def _compute_idempotency_payload(
    program: dict[str, Any],
    input_text: str,
    quantum: int,
    max_steps: int,
    max_tape_cells: int,
    max_wall_time_seconds: float,
) -> str:
    encoded = json.dumps(
        {
            "program": program,
            "input_text": input_text,
            "quantum": quantum,
            "max_steps": max_steps,
            "max_tape_cells": max_tape_cells,
            "max_wall_time_seconds": max_wall_time_seconds,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def start_computation(
    program: dict[str, Any] | None = None,
    input_text: str = "",
    quantum: int = 0,
    max_steps: int = 0,
    max_tape_cells: int = 0,
    max_wall_time_seconds: float = 0,
    idempotency_key: str = "",
) -> str:
    """Start a durable deterministic computation that resumes until HALT or cancellation.

    ``quantum`` limits one worker slice only. The max_* values are optional
    resource policies; zero leaves that dimension unbounded by harness policy.
    """
    try:
        normalized = validate_program(program or {})
    except (MachineProgramError, TypeError, ValueError) as exc:
        return f"Error: invalid computation program: {exc}"
    try:
        quantum = int(quantum)
        max_steps = int(max_steps)
        max_tape_cells = int(max_tape_cells)
        max_wall_time_seconds = float(max_wall_time_seconds)
    except (TypeError, ValueError):
        return "Error: quantum and resource limits must be numeric."
    if quantum < 0 or max_steps < 0 or max_tape_cells < 0 or max_wall_time_seconds < 0:
        return "Error: quantum and resource limits cannot be negative."
    if quantum > 100000:
        return "Error: quantum cannot exceed 100000 transitions per worker slice."

    key = str(idempotency_key or "").strip() or _compute_idempotency_payload(
        normalized,
        str(input_text),
        quantum,
        max_steps,
        max_tape_cells,
        max_wall_time_seconds,
    )
    existing = find_active_job_by_idempotency("durable_compute", key)
    if existing:
        return json.dumps(
            {
                "job_id": existing["id"],
                "status": existing["status"],
                "job_type": "durable_compute",
                "deduplicated": True,
            },
            ensure_ascii=False,
        )

    payload = {
        "program": normalized,
        "input_text": str(input_text),
        "idempotency_key": key,
        "quantum": quantum,
        "max_steps": max_steps,
        "max_tape_cells": max_tape_cells,
        "max_wall_time_seconds": max_wall_time_seconds,
    }
    job_id = create_job(
        "durable_compute",
        f"Durable computation: {normalized['initial_state']}",
        payload=payload,
        priority=0,
        max_attempts=3,
    )
    return json.dumps(
        {
            "job_id": job_id,
            "status": "pending",
            "job_type": "durable_compute",
            "deduplicated": False,
            "unbounded": not any((max_steps, max_tape_cells, max_wall_time_seconds)),
        },
        ensure_ascii=False,
    )


def get_computation_status(job_id: str = "", tape_start: int | None = None, tape_cells: int = 32) -> str:
    """Inspect durable computation progress and a bounded sparse tape window."""
    if not str(job_id).strip():
        return "Error: Missing required 'job_id' parameter."
    job = get_job(str(job_id))
    if not job:
        return f"Error: job '{job_id}' not found."
    if job.get("job_type") != "durable_compute":
        return f"Error: job '{job_id}' is not a durable computation."

    state = job.get("state") or {}
    payload = job.get("payload") or {}
    tape = state.get("tape") if isinstance(state.get("tape"), dict) else {}
    head = int(state.get("head", 0) or 0)
    cells = max(1, min(int(tape_cells), 256))
    start = int(tape_start) if tape_start is not None else head - (cells // 2)
    stop = start + cells
    window = {
        str(index): tape[str(index)]
        for index in range(start, stop)
        if str(index) in tape
    }
    result_value: Any = job.get("result")
    if isinstance(result_value, str) and result_value.strip().startswith("{"):
        try:
            result_value = json.loads(result_value)
        except json.JSONDecodeError:
            pass
    response = {
        "job_id": job["id"],
        "runtime_status": job.get("status"),
        "machine_status": state.get("status") or ("not_started" if not state else "unknown"),
        "machine_state": state.get("machine_state"),
        "steps": int(state.get("steps", 0) or 0),
        "yield_count": int(state.get("yield_count", 0) or 0),
        "checkpoint_generation": int(state.get("checkpoint_generation", 0) or 0),
        "head": head,
        "tape_cells": int(state.get("tape_cells", len(tape)) or 0),
        "tape_window": {"start": start, "end_exclusive": stop, "cells": window},
        "policy": {
            "quantum": int(payload.get("quantum", 0) or 0),
            "max_steps": int(payload.get("max_steps", 0) or 0),
            "max_tape_cells": int(payload.get("max_tape_cells", 0) or 0),
            "max_wall_time_seconds": float(payload.get("max_wall_time_seconds", 0) or 0),
        },
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
        "completed_at": job.get("completed_at"),
        "result": result_value,
        "error": job.get("error"),
    }
    return json.dumps(response, ensure_ascii=False, indent=2)


def cancel_computation(job_id: str = "") -> str:
    """Cancel a pending or running durable deterministic computation."""
    if not str(job_id).strip():
        return "Error: Missing required 'job_id' parameter."
    job = get_job(str(job_id))
    if not job:
        return f"Error: job '{job_id}' not found."
    if job.get("job_type") != "durable_compute":
        return f"Error: job '{job_id}' is not a durable computation."
    if not cancel_job(str(job_id)):
        return "Error: computation could not be cancelled; it may already be finished."
    return json.dumps({"job_id": str(job_id), "status": "cancelled", "job_type": "durable_compute"})
