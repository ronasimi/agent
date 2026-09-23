#!/usr/bin/env python3
"""Benchmark the deterministic durable-compute execution substrate.

This benchmark does not call Ollama. It measures exact machine transitions per
second for representative cooperative quantum sizes and the SQLite checkpoint
write cost for a small machine state. Run it on the deployment host when tuning
``worker.durable_compute_quantum``.
"""
from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

from al_agent.compute.machine import initialize_state, run_quantum
from tools import runtime

DEFAULT_QUANTA = (1_000, 10_000, 100_000)


def oscillator_program() -> dict:
    """Return a non-halting two-state machine with a one-cell working set."""
    return {
        "initial_state": "blank",
        "blank": "_",
        "halt_states": ["HALT"],
        "transitions": {
            "blank": {"_": {"write": "1", "move": "N", "next": "one"}},
            "one": {"1": {"write": "_", "move": "N", "next": "blank"}},
        },
    }


def benchmark_quantum(quantum: int, runs: int = 5) -> dict:
    """Measure pure deterministic transition throughput for one quantum size."""
    program = oscillator_program()
    samples = []
    transitions = []
    # Warm interpreter/import paths once before measuring.
    run_quantum(program, initialize_state(program), quantum=max(1, min(100, quantum)))
    for _ in range(max(1, int(runs))):
        state = initialize_state(program)
        start = time.perf_counter()
        result = run_quantum(program, state, quantum=int(quantum))
        elapsed = time.perf_counter() - start
        samples.append(elapsed)
        transitions.append(result.transitions_executed)
    median_s = statistics.median(samples)
    median_transitions = statistics.median(transitions)
    return {
        "quantum": int(quantum),
        "runs": len(samples),
        "median_ms": round(median_s * 1000, 3),
        "median_transitions_per_second": round(median_transitions / median_s, 1) if median_s else None,
        "min_ms": round(min(samples) * 1000, 3),
        "max_ms": round(max(samples) * 1000, 3),
    }


def benchmark_checkpoint(runs: int = 20) -> dict:
    """Measure a representative SQLite checkpoint write using an isolated DB."""
    old_db = runtime.DB_PATH
    samples = []
    try:
        with tempfile.TemporaryDirectory() as td:
            runtime.DB_PATH = str(Path(td) / "benchmark.db")
            runtime.init_runtime_db()
            job_id = runtime.create_job("durable_compute", "benchmark", {"benchmark": True})
            state = initialize_state(oscillator_program())
            for step in range(1, max(1, int(runs)) + 1):
                state = run_quantum(oscillator_program(), state, quantum=100).state
                start = time.perf_counter()
                runtime.save_checkpoint(job_id, state, step=step)
                samples.append(time.perf_counter() - start)
    finally:
        runtime.DB_PATH = old_db
    median_s = statistics.median(samples)
    return {
        "runs": len(samples),
        "median_ms": round(median_s * 1000, 3),
        "min_ms": round(min(samples) * 1000, 3),
        "max_ms": round(max(samples) * 1000, 3),
    }


def run_benchmark(quanta: tuple[int, ...] = DEFAULT_QUANTA, runs: int = 5, checkpoint_runs: int = 20) -> dict:
    """Return the complete benchmark report as a JSON-serializable object."""
    return {
        "quantum_results": [benchmark_quantum(q, runs=runs) for q in quanta],
        "checkpoint_write": benchmark_checkpoint(runs=checkpoint_runs),
        "guidance": (
            "Choose a quantum large enough that checkpoint overhead is small relative to compute time, "
            "but small enough that cancellation and queue fairness remain responsive. The quantum bounds "
            "one slice only and must not be treated as a total computation limit."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5, help="samples per quantum")
    parser.add_argument("--checkpoint-runs", type=int, default=20, help="SQLite checkpoint samples")
    parser.add_argument(
        "--quanta",
        default=",".join(str(q) for q in DEFAULT_QUANTA),
        help="comma-separated transition quanta",
    )
    args = parser.parse_args()
    quanta = tuple(int(value.strip()) for value in str(args.quanta).split(",") if value.strip())
    if not quanta or any(value < 1 for value in quanta):
        raise SystemExit("--quanta must contain positive integers")
    report = run_benchmark(quanta=quanta, runs=max(1, args.runs), checkpoint_runs=max(1, args.checkpoint_runs))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
