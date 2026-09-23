# Refactor notes

> **Historical status:** This is a point-in-time engineering/review record and is intentionally preserved as written. Model roles, tool counts, test totals, limits, and runtime behavior may have changed since this revision. For the current harness use `CURRENT_STATE.md`, `README.md`, and `ARCHITECTURE.md`.

## 2026-09-23 durable compute refactor

A deterministic resumable compute layer was added without relaxing the foreground agent's iteration/model-call limits or the shell/Python subprocess timeout. The new `al_agent.compute.machine` core executes versioned sparse-tape programs in bounded quanta. `durable_compute` background jobs atomically checkpoint continuation state and defer themselves, so healthy yields do not consume retry attempts and computations may resume for an arbitrary number of claims until `HALT`, cancellation, failure, or an explicitly requested resource policy.

The background provider contract was moved to `al_agent.background.types.JobHandler` so provider discovery no longer owns the type it imports from plugins. Model-facing start/status/cancel tools are idempotency-protected and expose bounded state only. The Web UI jobs panel displays compute progress and cancellation, the soak runner exercises the new mutators only against isolated state, and a model-independent benchmark measures transition and checkpoint throughput.

The generated shell/Python schemas now match the existing 120-second runtime clamp. Recovery, non-halting continuation, cancellation races, checkpoint corruption, branching, exact symbol mutation, negative tape positions, idempotent starts, routing/validator behavior, Web UI contracts, and benchmark execution all have focused regression coverage. See `DURABLE_COMPUTE.md` for the execution contract.


This refactor preserves public tool names while moving implementation behind focused modules and making the Web UI the only user interface.

Key compatibility facades:
- Web runtime -> `al_agent/runtime.py` + `webui/*`
- `worker.py` -> `al_agent/background/*`
- `tools/__init__.py` -> `tools/catalog.py` + memory exports
- `tools/primitive_ops.py` -> `tools/primitive_modules/*`

Extension seams are documented in `ARCHITECTURE.md`.  Existing recipes remain valid because registered public tool names did not change.
