"""Deterministic, checkpoint-friendly computation primitives.

The interactive LLM loop is intentionally bounded.  This package contains the
small deterministic execution substrate used by durable background computation
jobs when a task needs an arbitrary number of state transitions.
"""

from .machine import (
    CHECKPOINT_VERSION,
    PROGRAM_VERSION,
    MachineProgramError,
    QuantumResult,
    initialize_state,
    run_quantum,
    validate_program,
)

__all__ = [
    "CHECKPOINT_VERSION",
    "PROGRAM_VERSION",
    "MachineProgramError",
    "QuantumResult",
    "initialize_state",
    "run_quantum",
    "validate_program",
]
