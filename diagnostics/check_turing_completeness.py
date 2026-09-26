#!/usr/bin/env python3
"""Verify the harness's practical Turing-completeness contract.

This is an architectural contract check, not a claim of literally infinite
physical memory.  The harness is considered practically Turing complete when it
exposes a standard Turing-machine substrate with persistent sparse tape,
conditional state transitions, resumable unbounded iteration, and an active
worker/tool path that can execute that substrate independently of the bounded
foreground LLM loop.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def check_contract() -> list[str]:
    errors: list[str] = []

    try:
        from al_agent.background.job_providers.p15_compute import JOB_HANDLER
        from al_agent.compute.machine import initialize_state, run_quantum
        from al_agent.deterministic_router import DeterministicToolRouter
        from tools.builtin_manifest import BUILTIN_MANIFEST
        from tools.job_tools import start_computation
        from tools.providers import BUILTINS
        from tools.runtime import _join_compute_tape_address, _split_compute_tape_address
    except Exception as exc:  # pragma: no cover - reported as a contract failure
        return [f"Turing-completeness imports failed: {exc}"]

    required_tools = {
        ("job_tools", "start_computation"),
        ("job_tools", "get_computation_status"),
        ("job_tools", "cancel_computation"),
    }
    missing_tools = sorted(required_tools - set(BUILTINS))
    if missing_tools:
        errors.append(f"durable compute tools are not registered: {missing_tools}")

    if getattr(JOB_HANDLER, "name", "") != "durable_compute":
        errors.append("durable_compute background handler is not registered")

    signature = inspect.signature(start_computation)
    for name in ("max_steps", "max_tape_cells", "max_wall_time_seconds"):
        parameter = signature.parameters.get(name)
        if parameter is None or parameter.default not in (0, 0.0):
            errors.append(f"{name} must default to zero (no harness-level global bound)")

    # Iteration contract: one finite quantum must yield cleanly and resume from
    # its exact checkpoint without a total-step ceiling in the machine core.
    forever = {
        "version": 1,
        "initial_state": "run",
        "blank": "_",
        "halt_states": ["HALT"],
        "transitions": {
            "run": {
                "_": {"write": "1", "move": "R", "next": "run"},
                "1": {"write": "1", "move": "R", "next": "run"},
            }
        },
    }
    try:
        state = initialize_state(forever)
        first = run_quantum(forever, state, quantum=7)
        second = run_quantum(forever, first.state, quantum=11)
        if first.status != "yielded" or second.status != "yielded":
            errors.append("non-halting machine did not cooperatively yield")
        if int(second.state.get("steps", -1)) != 18:
            errors.append("durable machine did not resume exact transition count")
        if int(second.state.get("head", -1)) != 18:
            errors.append("durable machine did not resume exact tape-head position")
    except Exception as exc:
        errors.append(f"resumable iteration check failed: {exc}")

    # Conditional branching contract: execution path changes according to the
    # symbol read from tape, which is the canonical state-transition primitive.
    branch = {
        "version": 1,
        "initial_state": "branch",
        "blank": "_",
        "halt_states": ["HALT"],
        "transitions": {
            "branch": {
                "0": {"write": "A", "move": "N", "next": "HALT"},
                "1": {"write": "B", "move": "N", "next": "HALT"},
            }
        },
    }
    try:
        zero = run_quantum(branch, initialize_state(branch, input_text="0"), quantum=1)
        one = run_quantum(branch, initialize_state(branch, input_text="1"), quantum=1)
        if zero.state.get("tape", {}).get("0") != "A":
            errors.append("conditional branch for symbol 0 failed")
        if one.state.get("tape", {}).get("0") != "B":
            errors.append("conditional branch for symbol 1 failed")
    except Exception as exc:
        errors.append(f"conditional branching check failed: {exc}")

    # Tape/address contract: Python integer heads plus TEXT page indexes allow
    # addresses far beyond SQLite INT64, in both directions.
    try:
        far = 10**80
        for address in (far, -far):
            page, offset = _split_compute_tape_address(address)
            if _join_compute_tape_address(page, offset) != address:
                errors.append(f"arbitrary-precision tape address round trip failed: {address}")
    except Exception as exc:
        errors.append(f"arbitrary-precision tape check failed: {exc}")

    # Agent visibility contract: explicit Turing-machine requests must discover
    # the durable universal-compute tool without another routing model.
    try:
        schemas = [item["schema"] for item in BUILTIN_MANIFEST]
        decision = DeterministicToolRouter().decide(
            "simulate a Turing machine on a sparse tape until HALT", schemas
        )
        if "start_computation" not in set(decision.selected):
            errors.append("deterministic router does not activate start_computation for Turing-machine intent")
    except Exception as exc:
        errors.append(f"durable-compute routing check failed: {exc}")

    # Primary WebUI deployment must bring up the worker; otherwise the machine
    # can be queued but cannot make progress in the normal product entrypoint.
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    webui = compose.split("\n  webui:\n", 1)[-1] if "\n  webui:\n" in compose else ""
    if not webui or "\n      worker:\n" not in webui.split("\nvolumes:", 1)[0]:
        errors.append("webui service does not depend on the durable-compute worker")

    return errors


def main() -> int:
    errors = check_contract()
    if errors:
        print("Turing-completeness contract failed:")
        for error in errors:
            print(" -", error)
        return 1
    print("Turing-completeness contract passed (practical finite-resource model)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
