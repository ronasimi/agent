from __future__ import annotations

import pytest

from al_agent.compute.machine import MachineProgramError, initialize_state, run_quantum, validate_program


def scan_to_end_program():
    return {
        "version": 1,
        "initial_state": "scan",
        "blank": "_",
        "halt_states": ["HALT"],
        "transitions": {
            "scan": {
                "1": {"write": "1", "move": "R", "next": "scan"},
                "0": {"write": "0", "move": "R", "next": "scan"},
                "_": {"write": "_", "move": "N", "next": "HALT"},
            }
        },
    }


def test_machine_halts_with_exact_transition_count():
    program = scan_to_end_program()
    state = initialize_state(program, "101")
    result = run_quantum(program, state, quantum=100)
    assert result.status == "halted"
    assert result.transitions_executed == 4
    assert result.state["steps"] == 4
    assert result.state["head"] == 3
    assert result.state["tape"] == {"0": "1", "1": "0", "2": "1"}


def test_machine_yields_across_multiple_quanta_before_halting():
    program = scan_to_end_program()
    state = initialize_state(program, "111111")
    statuses = []
    while True:
        result = run_quantum(program, state, quantum=2)
        statuses.append(result.status)
        state = result.state
        if result.status == "halted":
            break
    assert statuses == ["yielded", "yielded", "yielded", "halted"]
    assert state["steps"] == 7
    assert state["yield_count"] == 3


def test_non_halting_machine_remains_resumable_without_global_step_cap():
    program = {
        "initial_state": "right",
        "halt_states": ["HALT"],
        "transitions": {
            "right": {"_": {"write": "1", "move": "R", "next": "right"}}
        },
    }
    state = initialize_state(program)
    for _ in range(100):
        result = run_quantum(program, state, quantum=3)
        assert result.status == "yielded"
        state = result.state
    assert state["steps"] == 300
    assert state["yield_count"] == 100
    assert state["tape_cells"] == 300


def test_machine_supports_negative_tape_addresses_and_branching():
    program = {
        "initial_state": "branch",
        "halt_states": ["HALT"],
        "transitions": {
            "branch": {
                "A": {"write": "X", "move": "L", "next": "left"},
                "B": {"write": "Y", "move": "R", "next": "HALT"},
            },
            "left": {"_": {"write": "L", "move": "N", "next": "HALT"}},
        },
    }
    a = run_quantum(program, initialize_state(program, "A"), quantum=10)
    b = run_quantum(program, initialize_state(program, "B"), quantum=10)
    assert a.state["tape"] == {"-1": "L", "0": "X"}
    assert b.state["tape"] == {"0": "Y"}
    assert a.state["steps"] == 2
    assert b.state["steps"] == 1


def test_invalid_program_and_corrupt_checkpoint_are_rejected_cleanly():
    with pytest.raises(MachineProgramError):
        validate_program({"initial_state": "q0", "transitions": {"q0": {"_": {"write": "_", "move": "UP", "next": "HALT"}}}})

    program = scan_to_end_program()
    # Invalid structural checkpoints are storage/programmer errors and are
    # rejected before any transition executes rather than silently resetting.
    with pytest.raises(MachineProgramError):
        run_quantum(program, {"checkpoint_version": 999}, quantum=1)
