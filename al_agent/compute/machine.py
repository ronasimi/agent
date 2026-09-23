"""Small deterministic universal-machine core for durable computations.

This module deliberately has no dependency on the agent loop, SQLite, workers,
or model clients.  A caller gives it a JSON-serializable program and checkpoint,
and :func:`run_quantum` performs a bounded number of exact transitions.  The
caller is responsible for persisting the returned state and scheduling another
quantum when the result is ``yielded``.

The machine uses a sparse, bidirectional tape. Missing cells contain the
program's blank symbol and blank writes remove cells from the checkpoint.  No
architectural tape bound or total-step bound is imposed here; physical storage
and optional runtime policy remain external concerns.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

PROGRAM_VERSION = 1
CHECKPOINT_VERSION = 1
VALID_MOVES = {"L", "R", "N"}


class MachineProgramError(ValueError):
    """Raised when a durable-compute program or checkpoint is invalid."""


@dataclass(frozen=True)
class QuantumResult:
    """Outcome of one bounded deterministic execution slice."""

    status: Literal["halted", "yielded", "failed"]
    state: dict[str, Any]
    transitions_executed: int
    error: str | None = None


def _nonempty_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise MachineProgramError(f"{field} must be a non-empty string")
    return value


def validate_program(program: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize one deterministic machine program.

    Program format::

        {
          "version": 1,
          "initial_state": "q0",
          "blank": "_",
          "halt_states": ["HALT"],
          "transitions": {
            "q0": {
              "1": {"write": "1", "move": "R", "next": "q0"},
              "_": {"write": "_", "move": "N", "next": "HALT"}
            }
          }
        }

    Transition/state names and symbols are strings.  Symbols are not restricted
    to one character so callers may use compact token alphabets when desired.
    """
    if not isinstance(program, dict):
        raise MachineProgramError("program must be an object")
    version = int(program.get("version", PROGRAM_VERSION))
    if version != PROGRAM_VERSION:
        raise MachineProgramError(f"unsupported program version {version}")
    initial_state = _nonempty_text(program.get("initial_state"), "initial_state")
    blank = _nonempty_text(program.get("blank", "_"), "blank")
    halt_states_raw = program.get("halt_states", ["HALT"])
    if not isinstance(halt_states_raw, list) or not halt_states_raw:
        raise MachineProgramError("halt_states must be a non-empty list")
    halt_states = []
    for item in halt_states_raw:
        state_name = _nonempty_text(item, "halt state")
        if state_name not in halt_states:
            halt_states.append(state_name)

    transitions_raw = program.get("transitions")
    if not isinstance(transitions_raw, dict):
        raise MachineProgramError("transitions must be an object")
    transitions: dict[str, dict[str, dict[str, str]]] = {}
    for raw_state, raw_rules in transitions_raw.items():
        state = _nonempty_text(raw_state, "transition state")
        if state in halt_states:
            raise MachineProgramError(f"halt state '{state}' cannot define outgoing transitions")
        if not isinstance(raw_rules, dict):
            raise MachineProgramError(f"transitions for '{state}' must be an object")
        rules: dict[str, dict[str, str]] = {}
        for raw_symbol, raw_action in raw_rules.items():
            symbol = _nonempty_text(raw_symbol, "read symbol")
            if not isinstance(raw_action, dict):
                raise MachineProgramError(f"transition {state}/{symbol} must be an object")
            write = _nonempty_text(raw_action.get("write"), f"transition {state}/{symbol}.write")
            move = str(raw_action.get("move", "")).upper()
            if move not in VALID_MOVES:
                raise MachineProgramError(f"transition {state}/{symbol}.move must be L, R, or N")
            next_state = _nonempty_text(raw_action.get("next"), f"transition {state}/{symbol}.next")
            rules[symbol] = {"write": write, "move": move, "next": next_state}
        transitions[state] = rules

    if initial_state not in halt_states and initial_state not in transitions:
        raise MachineProgramError(f"initial state '{initial_state}' has no transition table")

    # Catch misspelled/unreachable transition targets before a long-running
    # durable job is queued.  A target is valid only when it has its own
    # transition table or is explicitly terminal.
    defined_states = set(transitions) | set(halt_states)
    for state, rules in transitions.items():
        for symbol, action in rules.items():
            target = action["next"]
            if target not in defined_states:
                raise MachineProgramError(
                    f"transition {state}/{symbol} references undefined state {target!r}"
                )

    return {
        "version": PROGRAM_VERSION,
        "initial_state": initial_state,
        "blank": blank,
        "halt_states": halt_states,
        "transitions": transitions,
    }


def normalize_initial_tape(
    program: dict[str, Any],
    input_text: str = "",
    initial_tape: Mapping[int | str, str] | None = None,
) -> dict[str, str]:
    """Build the canonical sparse initial tape.

    ``input_text`` is written character-by-character from address zero.  An
    optional ``initial_tape`` map is then applied as an overlay, so callers can
    initialize arbitrary positive or negative addresses and use multi-character
    symbols.  Blank symbols are omitted from the sparse representation.
    """
    normalized = validate_program(program)
    tape = {str(index): symbol for index, symbol in enumerate(str(input_text)) if symbol != normalized["blank"]}
    if initial_tape is not None:
        if not isinstance(initial_tape, Mapping):
            raise MachineProgramError("initial_tape must be an object mapping integer addresses to symbols")
        for raw_address, raw_symbol in initial_tape.items():
            try:
                address = int(raw_address)
            except (TypeError, ValueError) as exc:
                raise MachineProgramError(f"invalid initial tape address {raw_address!r}") from exc
            symbol = _nonempty_text(raw_symbol, f"initial_tape[{raw_address}]")
            key = str(address)
            if symbol == normalized["blank"]:
                tape.pop(key, None)
            else:
                tape[key] = symbol
    return {key: tape[key] for key in sorted(tape, key=int)}


def initialize_state(
    program: dict[str, Any],
    input_text: str = "",
    initial_tape: Mapping[int | str, str] | None = None,
    initial_head: int = 0,
) -> dict[str, Any]:
    """Return a fresh JSON-serializable checkpoint for ``program`` and input."""
    normalized = validate_program(program)
    tape = normalize_initial_tape(normalized, input_text, initial_tape)
    try:
        head = int(initial_head)
    except (TypeError, ValueError) as exc:
        raise MachineProgramError("initial_head must be an integer") from exc
    status = "halted" if normalized["initial_state"] in normalized["halt_states"] else "running"
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "machine_state": normalized["initial_state"],
        "head": head,
        "steps": 0,
        "yield_count": 0,
        "checkpoint_generation": 0,
        "status": status,
        "tape": tape,
        "tape_cells": len(tape),
        "last_quantum_transitions": 0,
        "last_error": None,
    }


def _normalize_checkpoint(state: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(state, dict):
        raise MachineProgramError("checkpoint must be an object")
    version = int(state.get("checkpoint_version", 0))
    if version != CHECKPOINT_VERSION:
        raise MachineProgramError(f"unsupported checkpoint version {version}")
    tape_raw = state.get("tape", {})
    if not isinstance(tape_raw, dict):
        raise MachineProgramError("checkpoint tape must be an object")
    tape: dict[int, str] = {}
    for key, value in tape_raw.items():
        try:
            index = int(key)
        except (TypeError, ValueError) as exc:
            raise MachineProgramError(f"invalid tape address {key!r}") from exc
        tape[index] = _nonempty_text(value, f"tape[{key}]")
    machine_state = _nonempty_text(state.get("machine_state"), "checkpoint machine_state")
    try:
        head = int(state.get("head", 0))
        steps = int(state.get("steps", 0))
        yield_count = int(state.get("yield_count", 0))
        checkpoint_generation = int(state.get("checkpoint_generation", 0))
    except (TypeError, ValueError) as exc:
        raise MachineProgramError("checkpoint counters must be integers") from exc
    if steps < 0 or yield_count < 0 or checkpoint_generation < 0:
        raise MachineProgramError("checkpoint counters cannot be negative")
    try:
        tape_cells = int(state.get("tape_cells", len(tape)))
    except (TypeError, ValueError) as exc:
        raise MachineProgramError("checkpoint tape_cells must be an integer") from exc
    if tape_cells < 0:
        raise MachineProgramError("checkpoint tape_cells cannot be negative")
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "machine_state": machine_state,
        "head": head,
        "steps": steps,
        "yield_count": yield_count,
        "checkpoint_generation": checkpoint_generation,
        "status": str(state.get("status") or "running"),
        "tape": tape,
        # In the pure in-memory machine this equals len(tape).  Durable workers
        # may hydrate only the scheduler-reachable tape window and preserve the
        # total populated-cell count separately.
        "tape_cells": tape_cells,
        "last_error": state.get("last_error"),
    }


def _serialize_checkpoint(state: dict[str, Any], *, transitions: int, status: str, error: str | None = None) -> dict[str, Any]:
    tape: dict[int, str] = state["tape"]
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "machine_state": state["machine_state"],
        "head": int(state["head"]),
        "steps": int(state["steps"]),
        "yield_count": int(state["yield_count"]),
        "checkpoint_generation": int(state.get("checkpoint_generation", 0)) + 1,
        "status": status,
        "tape": {str(key): tape[key] for key in sorted(tape)},
        "tape_cells": int(state.get("tape_cells", len(tape))),
        "last_quantum_transitions": int(transitions),
        "last_error": error,
    }


def run_quantum(program: dict[str, Any], state: dict[str, Any], quantum: int = 10_000) -> QuantumResult:
    """Execute at most ``quantum`` transitions and return a resumable result.

    ``quantum`` bounds one scheduler slice, not the total computation.  A
    ``yielded`` result is healthy progress and can be checkpointed/requeued an
    arbitrary number of times by the durable worker.
    """
    normalized = validate_program(program)
    work = _normalize_checkpoint(state)
    quantum = int(quantum)
    if quantum < 1:
        raise MachineProgramError("quantum must be at least 1")

    halt_states = set(normalized["halt_states"])
    blank = normalized["blank"]
    transitions = normalized["transitions"]
    executed = 0

    if work["machine_state"] in halt_states:
        checkpoint = _serialize_checkpoint(work, transitions=0, status="halted")
        return QuantumResult("halted", checkpoint, 0)

    try:
        for _ in range(quantum):
            current = work["machine_state"]
            if current in halt_states:
                break
            symbol = work["tape"].get(work["head"], blank)
            action = transitions.get(current, {}).get(symbol)
            if action is None:
                raise MachineProgramError(f"no transition for state {current!r} reading {symbol!r}")

            address = work["head"]
            was_populated = address in work["tape"]
            if action["write"] == blank:
                work["tape"].pop(work["head"], None)
            else:
                work["tape"][work["head"]] = action["write"]
            is_populated = action["write"] != blank
            if was_populated != is_populated:
                work["tape_cells"] += 1 if is_populated else -1

            if action["move"] == "L":
                work["head"] -= 1
            elif action["move"] == "R":
                work["head"] += 1
            work["machine_state"] = action["next"]
            work["steps"] += 1
            executed += 1

            if work["machine_state"] in halt_states:
                checkpoint = _serialize_checkpoint(work, transitions=executed, status="halted")
                return QuantumResult("halted", checkpoint, executed)

        work["yield_count"] += 1
        checkpoint = _serialize_checkpoint(work, transitions=executed, status="yielded")
        return QuantumResult("yielded", checkpoint, executed)
    except MachineProgramError as exc:
        checkpoint = _serialize_checkpoint(work, transitions=executed, status="failed", error=str(exc))
        return QuantumResult("failed", checkpoint, executed, str(exc))
