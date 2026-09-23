"""Durable deterministic computation job provider.

Each claim executes one bounded quantum, atomically checkpoints progress, and
then either completes, fails, or returns the job to the queue.  There is no
mandatory total step/yield limit; optional per-job policies can impose one when
a caller explicitly requests bounded resource consumption.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from al_agent.compute.inputs import load_input_file
from al_agent.compute.machine import MachineProgramError, initialize_state, run_quantum, validate_program
from tools.runtime import (
    checkpoint_and_complete_job,
    checkpoint_and_defer_job,
    checkpoint_and_fail_job,
    get_compute_tape_window,
    get_job,
    load_checkpoint,
    migrate_inline_compute_tape,
    replace_compute_tape,
)

from ..config import DURABLE_COMPUTE_QUANTUM, DURABLE_COMPUTE_YIELD_DELAY_SECONDS
from ..types import JobHandler


def _positive_policy(payload: dict[str, Any], name: str) -> int | float | None:
    value = payload.get(name)
    if value in (None, "", 0, 0.0):
        return None
    try:
        number = float(value) if name == "max_wall_time_seconds" else int(value)
    except (TypeError, ValueError):
        raise MachineProgramError(f"{name} must be a positive number or zero for unbounded")
    if number <= 0:
        return None
    return number


def _age_seconds(created_at: str) -> float:
    try:
        created = datetime.fromisoformat(str(created_at))
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - created.astimezone(timezone.utc)).total_seconds())
    except (TypeError, ValueError):
        return 0.0


def _terminal_summary(state: dict[str, Any]) -> str:
    return json.dumps(
        {
            "status": state.get("status"),
            "machine_state": state.get("machine_state"),
            "steps": int(state.get("steps", 0)),
            "yield_count": int(state.get("yield_count", 0)),
            "checkpoint_generation": int(state.get("checkpoint_generation", 0)),
            "head": int(state.get("head", 0)),
            "tape_cells": int(state.get("tape_cells", 0)),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _tape_delta(before: dict[str, str], after: dict[str, str]) -> dict[int, str | None]:
    """Return only cells changed during one hydrated execution window."""
    updates: dict[int, str | None] = {}
    for key in set(before) | set(after):
        old = before.get(key)
        new = after.get(key)
        if old == new:
            continue
        updates[int(key)] = new
    return updates


def run_durable_compute_job(job_id: str, worker_id: str) -> None:
    """Run exactly one cooperative compute quantum for ``job_id``."""
    del worker_id  # Job ownership is enforced by the runtime claim/update state.
    job = get_job(job_id)
    if not job:
        raise RuntimeError(f"Durable compute job '{job_id}' disappeared")
    payload = job.get("payload") or {}
    program = validate_program(payload.get("program") or {})
    state = load_checkpoint(job_id)
    if not state:
        input_text = str(payload.get("input_text") or "")
        initial_tape = dict(payload.get("initial_tape") or {})
        descriptor = payload.get("input_file")
        if descriptor:
            file_text, file_tape = load_input_file(descriptor)
            if str(descriptor.get("format") or "") == "text":
                input_text = file_text
            if file_tape:
                # File-provided tape is the base; explicit initial_tape values
                # are an intentional overlay supplied directly in the tool call.
                merged = dict(file_tape)
                merged.update(initial_tape)
                initial_tape = merged
        state = initialize_state(
            program,
            input_text,
            initial_tape=initial_tape,
            initial_head=int(payload.get("initial_head", 0) or 0),
        )
        initial_tape = dict(state.pop("tape", {}))
        state["tape_cells"] = replace_compute_tape(job_id, initial_tape)
    else:
        # One-time compatibility migration for jobs created before tape cells
        # moved out of JSON checkpoints into their own sparse SQLite table.
        state = migrate_inline_compute_tape(job_id, state)

    max_steps = _positive_policy(payload, "max_steps")
    max_tape_cells = _positive_policy(payload, "max_tape_cells")
    max_wall_time = _positive_policy(payload, "max_wall_time_seconds")

    if max_wall_time is not None and _age_seconds(job.get("created_at") or "") >= float(max_wall_time):
        state = dict(state)
        state["checkpoint_generation"] = int(state.get("checkpoint_generation", 0)) + 1
        state["status"] = "failed"
        state["last_error"] = "configured max_wall_time_seconds reached before HALT"
        checkpoint_and_fail_job(job_id, state, state["last_error"], step=state["checkpoint_generation"])
        return
    if max_steps is not None and int(state.get("steps", 0)) >= int(max_steps):
        state = dict(state)
        state["checkpoint_generation"] = int(state.get("checkpoint_generation", 0)) + 1
        state["status"] = "failed"
        state["last_error"] = "configured max_steps reached before HALT"
        checkpoint_and_fail_job(job_id, state, state["last_error"], step=state["checkpoint_generation"])
        return
    if max_tape_cells is not None and int(state.get("tape_cells", 0)) > int(max_tape_cells):
        state = dict(state)
        state["checkpoint_generation"] = int(state.get("checkpoint_generation", 0)) + 1
        state["status"] = "failed"
        state["last_error"] = "configured max_tape_cells exceeded by initial/resumed tape before execution"
        checkpoint_and_fail_job(job_id, state, state["last_error"], step=state["checkpoint_generation"])
        return

    requested_quantum = int(payload.get("quantum") or DURABLE_COMPUTE_QUANTUM)
    quantum = max(1, min(requested_quantum, 100000))
    if max_steps is not None:
        quantum = min(quantum, max(1, int(max_steps) - int(state.get("steps", 0))))

    # In one quantum the head can move at most ``quantum`` addresses from its
    # starting point, and writes happen before movement. Hydrating this bounded
    # range therefore supplies every tape cell the machine can observe in the
    # current slice without reading or serializing the full historical tape.
    head = int(state.get("head", 0) or 0)
    window_start = head - quantum
    window_end = head + quantum + 1
    hydrated_tape = get_compute_tape_window(job_id, window_start, window_end)
    hydrated_state = dict(state)
    hydrated_state["tape"] = hydrated_tape
    result = run_quantum(program, hydrated_state, quantum=quantum)
    final_tape = dict(result.state.get("tape") or {})
    tape_updates = _tape_delta(hydrated_tape, final_tape)
    durable_state = dict(result.state)
    durable_state.pop("tape", None)
    result = type(result)(
        status=result.status,
        state=durable_state,
        transitions_executed=result.transitions_executed,
        error=result.error,
    )
    generation = int(result.state.get("checkpoint_generation", 0))

    if max_tape_cells is not None and int(result.state.get("tape_cells", 0)) > int(max_tape_cells):
        limited = dict(result.state)
        limited["status"] = "failed"
        limited["last_error"] = "configured max_tape_cells reached before HALT"
        checkpoint_and_fail_job(
            job_id, limited, limited["last_error"], step=generation, tape_updates=tape_updates
        )
        return
    if max_wall_time is not None and _age_seconds(job.get("created_at") or "") >= float(max_wall_time):
        limited = dict(result.state)
        limited["status"] = "failed"
        limited["last_error"] = "configured max_wall_time_seconds reached before HALT"
        checkpoint_and_fail_job(
            job_id, limited, limited["last_error"], step=generation, tape_updates=tape_updates
        )
        return

    if result.status == "halted":
        checkpoint_and_complete_job(
            job_id,
            result.state,
            _terminal_summary(result.state),
            step=generation,
            tape_updates=tape_updates,
        )
        return
    if result.status == "failed":
        checkpoint_and_fail_job(
            job_id,
            result.state,
            result.error or "deterministic machine failed",
            step=generation,
            tape_updates=tape_updates,
        )
        return

    # A normal yield is healthy progress. It intentionally does not consume a
    # retry attempt and does not count toward any implicit total runtime bound.
    checkpoint_and_defer_job(
        job_id,
        result.state,
        step=generation,
        delay_seconds=DURABLE_COMPUTE_YIELD_DELAY_SECONDS,
        tape_updates=tape_updates,
    )


JOB_HANDLER = JobHandler("durable_compute", run_durable_compute_job)
