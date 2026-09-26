"""Agent-visible durable job management tools."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from al_agent.compute.inputs import describe_input_file, load_input_file
from al_agent.compute.machine import MachineProgramError, normalize_initial_tape, validate_program

from .runtime import (
    cancel_job,
    create_job,
    find_active_job_by_idempotency,
    get_compute_tape_cell_count,
    get_compute_tape_window,
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
    initial_tape: dict[str, str],
    initial_head: int,
    input_file: dict[str, str] | None,
    quantum: int,
    max_steps: int,
    max_tape_cells: int,
    max_wall_time_seconds: float,
    max_recovery_failures: int,
) -> str:
    encoded = json.dumps(
        {
            "program": program,
            "input_text": input_text,
            "initial_tape": initial_tape,
            "initial_head": initial_head,
            "input_file": input_file,
            "quantum": quantum,
            "max_steps": max_steps,
            "max_tape_cells": max_tape_cells,
            "max_wall_time_seconds": max_wall_time_seconds,
            "max_recovery_failures": max_recovery_failures,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def start_computation(
    program: dict[str, Any] | None = None,
    input_text: str = "",
    initial_tape: dict[str, str] | None = None,
    initial_head: int = 0,
    input_file: str = "",
    input_file_format: Literal["auto", "text", "tape_json"] = "auto",
    quantum: int = 0,
    max_steps: int = 0,
    max_tape_cells: int = 0,
    max_wall_time_seconds: float = 0,
    max_recovery_failures: int = 10,
    idempotency_key: str = "",
) -> str:
    """Simulate a durable universal Turing machine with potentially unbounded steps on a sparse bidirectional tape; resumes until HALT or cancellation.

    ``input_text`` seeds cells from address zero, ``initial_tape`` can overlay
    arbitrary integer addresses, and ``input_file`` may reference a hash-pinned
    workspace text/JSON source. ``quantum`` limits one worker slice only. The
    resource max_* values are optional policies; zero leaves that dimension
    unbounded. ``max_recovery_failures`` is a separate infrastructure-failure
    budget; zero explicitly requests unlimited recovery.
    """
    try:
        normalized = validate_program(program or {})
        canonical_initial_tape = normalize_initial_tape(normalized, "", initial_tape or {})
        initial_head = int(initial_head)
        input_descriptor = describe_input_file(input_file, input_file_format) if str(input_file).strip() else None
        # JSON tape sources are validated at queue time so malformed input is a
        # user-facing tool error rather than a delayed worker failure.
        if input_descriptor and input_descriptor["format"] == "tape_json":
            _text, file_tape = load_input_file(input_descriptor)
            normalize_initial_tape(normalized, "", file_tape)
    except (MachineProgramError, TypeError, ValueError) as exc:
        return f"Error: invalid computation setup: {exc}"
    try:
        quantum = int(quantum)
        max_steps = int(max_steps)
        max_tape_cells = int(max_tape_cells)
        max_wall_time_seconds = float(max_wall_time_seconds)
        max_recovery_failures = int(max_recovery_failures)
    except (TypeError, ValueError):
        return "Error: quantum, resource limits, and recovery limits must be numeric."
    if (
        quantum < 0
        or max_steps < 0
        or max_tape_cells < 0
        or max_wall_time_seconds < 0
        or max_recovery_failures < 0
    ):
        return "Error: quantum and resource limits cannot be negative."
    if quantum > 100000:
        return "Error: quantum cannot exceed 100000 transitions per worker slice."

    key = str(idempotency_key or "").strip() or _compute_idempotency_payload(
        normalized,
        str(input_text),
        canonical_initial_tape,
        initial_head,
        input_descriptor,
        quantum,
        max_steps,
        max_tape_cells,
        max_wall_time_seconds,
        max_recovery_failures,
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
        "initial_tape": canonical_initial_tape,
        "initial_head": initial_head,
        "input_file": input_descriptor,
        "idempotency_key": key,
        "quantum": quantum,
        "max_steps": max_steps,
        "max_tape_cells": max_tape_cells,
        "max_wall_time_seconds": max_wall_time_seconds,
        "max_recovery_failures": max_recovery_failures,
    }
    job_id = create_job(
        "durable_compute",
        f"Durable computation: {normalized['initial_state']}",
        payload=payload,
        priority=0,
        max_attempts=3,
        max_recovery_failures=max_recovery_failures,
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
    head = int(state.get("head", 0) or 0)
    cells = max(1, min(int(tape_cells), 256))
    start = int(tape_start) if tape_start is not None else head - (cells // 2)
    stop = start + cells
    window = get_compute_tape_window(job["id"], start, stop)
    # Compatibility fallback for jobs that have not yet been claimed/migrated.
    if not window and isinstance(state.get("tape"), dict):
        inline = state["tape"]
        window = {str(index): inline[str(index)] for index in range(start, stop) if str(index) in inline}
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
        "tape_cells": int(state.get("tape_cells", get_compute_tape_cell_count(job["id"])) or 0),
        "tape_window": {"start": start, "end_exclusive": stop, "cells": window},
        "policy": {
            "quantum": int(payload.get("quantum", 0) or 0),
            "max_steps": int(payload.get("max_steps", 0) or 0),
            "max_tape_cells": int(payload.get("max_tape_cells", 0) or 0),
            "max_wall_time_seconds": float(payload.get("max_wall_time_seconds", 0) or 0),
            "max_recovery_failures": int(job.get("max_recovery_failures", 0) or 0),
        },
        "recovery_failures": int(job.get("recovery_failures", 0) or 0),
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
