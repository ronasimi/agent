# Durable deterministic computation

> **Current-state note (2026-09-23):** This is the operator/developer guide for the resumable deterministic compute subsystem. The foreground LLM loop remains deliberately bounded; arbitrary-length iteration lives below it in checkpointed worker jobs.

## Why this exists

The interactive agent has hard iteration, model-call, and wall-clock limits so a small local model cannot spin indefinitely. Those safeguards remain in place. When a task genuinely requires an arbitrary number of deterministic state transitions, the harness can instead create a `durable_compute` background job.

A durable computation executes one bounded **quantum**, commits lightweight machine metadata plus the sparse tape cells changed by that quantum in the same SQLite transaction as the queue transition, and yields back to the worker. If the machine has not reached `HALT`, the job is requeued and may resume for another quantum. There is no mandatory total step or yield ceiling unless the caller explicitly supplies one of the optional resource policies.

```text
bounded foreground LLM turn
          │
          └── start_computation
                    │
                    ▼
              durable job
                    │
             load checkpoint
                    │
             run N transitions
                    │
          ┌─────────┴─────────┐
        HALT                not HALT
          │                    │
     checkpoint +          checkpoint +
       complete               defer
                               │
                               └── requeue
```

The LLM is not called between compute quanta. This keeps the universal execution substrate deterministic and avoids consuming model tokens for bookkeeping.

## Program format

Version 1 uses a sparse bidirectional tape and explicit transition table:

```json
{
  "version": 1,
  "initial_state": "scan",
  "blank": "_",
  "halt_states": ["HALT"],
  "transitions": {
    "scan": {
      "1": {"write": "1", "move": "R", "next": "scan"},
      "_": {"write": "_", "move": "N", "next": "HALT"}
    }
  }
}
```

`move` is `L`, `R`, or `N`. Symbols and state names are non-empty strings. A halt state cannot define outgoing transitions. Missing tape addresses read as the program's blank symbol; writing the blank symbol removes that sparse tape cell.

The checkpoint format is versioned independently from the program. It persists the current machine state, head position, exact transition count, yield count, checkpoint generation, populated-cell count, and bounded observability fields. Tape cells are stored separately in `durable_compute_tape(job_id, address, symbol)` rather than copied into every checkpoint. Unknown checkpoint versions are rejected rather than silently reset.

Every quantum hydrates only the address range the head can possibly reach during that slice (`head - quantum` through `head + quantum`). The deterministic core records exact symbol changes, and the runtime upserts/deletes only those changed cells. This makes normal checkpoint work proportional to the cells changed in the current quantum rather than to the total historical tape size. Older inline-tape checkpoints are migrated automatically on first resume.

Program validation is strict: every transition target must either have a transition table of its own or be listed in `halt_states`. Misspelled target states are rejected before a background job is queued instead of becoming delayed runtime failures.

## Agent-visible tools

- `start_computation(program, input_text="", initial_tape=None, initial_head=0, input_file="", input_file_format="auto", quantum=0, max_steps=0, max_tape_cells=0, max_wall_time_seconds=0, max_recovery_failures=10, idempotency_key="")`
- `get_computation_status(job_id, tape_start=None, tape_cells=32)`
- `cancel_computation(job_id)`

`quantum=0` uses `worker.durable_compute_quantum`. It limits only one scheduling slice. The `max_*` values are optional policies; zero means that dimension has no harness-imposed global limit. Active creation retries are deduplicated by an idempotency fingerprint unless the caller supplies an explicit key.

Initial tape data can come from three sources:

- `input_text` writes one character per cell starting at address `0`;
- `initial_tape` is a sparse address-to-symbol map and may use negative addresses or multi-character symbols;
- `input_file` points to a workspace-local text file or JSON sparse-tape object. `input_file_format=auto` treats `.json` as `tape_json` and other files as text.

File inputs are SHA-256 pinned when the job is queued. A worker verifies that digest before first execution, so editing the referenced file later cannot silently change a queued computation. Explicit `initial_tape` entries overlay file-provided tape entries. `initial_head` selects the starting address independently of the input source.

`get_computation_status` does not dump the entire tape. It returns progress counters plus a bounded sparse tape window centered on the head unless `tape_start` is provided.

## Worker and crash semantics

`al_agent/background/job_providers/p15_compute.py` executes exactly one quantum per job claim. Healthy yields call `checkpoint_and_defer_job`, which performs the checkpoint insert and queue-state update in one SQLite transaction and returns the claim attempt so yields do not consume retry budget.

The durable checkpoint includes a monotonic `checkpoint_generation`. Tape deltas, checkpoint metadata, and the pending/completed/failed queue transition are committed atomically. If a worker dies before that transaction commits, the previous checkpoint+tape state remains authoritative and the pure deterministic quantum can be replayed safely. If the transaction commits first, the job is already pending with the new state and the next worker resumes from that generation.

The worker's normal `max_job_runtime_seconds` watchdog still bounds a single claimed execution context. It is **not** a total lifetime limit for the durable computation because each normal quantum returns and is later reclaimed.

Infrastructure recovery has a separate counter from ordinary handler attempts. A stale heartbeat or worker watchdog restart returns the consumed claim attempt, increments `recovery_failures`, and requeues the job until `max_recovery_failures` is reached. Healthy compute progress resets the consecutive recovery count. `start_computation` defaults this policy to `10`; setting it to `0` explicitly requests unlimited infrastructure recovery. Deterministic machine/program failures still use their normal terminal/failure path rather than being mislabeled as infrastructure recovery.

Cancellation is terminal. Atomic checkpoint helpers update only jobs still in `running`, so a late worker result cannot resurrect a job that was cancelled concurrently.

## Resource policy

The default mode has no harness-level total step, tape-cell, or wall-time limit. Operators can opt into any combination of:

- `max_steps`
- `max_tape_cells`
- `max_wall_time_seconds`

These are explicit job policies, not hidden runtime bounds. Physical CPU, RAM, disk, SQLite limits, process termination, and host failure still exist as they do for every real computer. The subsystem is therefore described as **practically Turing complete under the conventional finite-resource computer abstraction**, not as literally possessing infinite physical memory.

The foreground LLM loop, recipes, and generic shell/Python tools remain bounded. Do not remove their limits to obtain universality; durable compute is the intended escape hatch.

## Web UI observability

The Jobs panel exposes, for `durable_compute` jobs:

- runtime status and machine state;
- exact transition count;
- yield count;
- populated sparse tape-cell count;
- consecutive infrastructure recovery count and configured maximum when non-zero;
- cancellation for pending/running jobs.

The panel refreshes while visible. `list_jobs()` returns only a bounded progress summary, never the full tape. Detailed bounded inspection remains available through `get_computation_status`.

## Configuration

```yaml
worker:
  max_job_runtime_seconds: 7200
  durable_compute_quantum: 10000
  durable_compute_yield_delay_seconds: 1
```

A short yield delay improves queue fairness so an old, non-halting computation does not immediately reclaim the worker ahead of unrelated jobs. `durable_compute_quantum` is clamped to 100,000 transitions per slice.

## Testing

Focused tests cover exact halting, strict transition-target validation, branching, negative tape addresses, arbitrary sparse initial tape maps, workspace-file input/hash pinning, quantum-partition equivalence, multi-quantum resumption, non-halting execution, sparse transactional tape persistence, stale-worker/infrastructure recovery budgets, cancellation races, explicit resource policies, idempotent creation, routing/validator behavior, and Web UI progress output.

```bash
python -m pytest -q \
  tests/test_durable_compute_machine.py \
  tests/test_durable_compute_worker.py \
  tests/test_durable_compute_tools.py \
  tests/test_runtime.py
```

Benchmark deterministic transition throughput and SQLite checkpoint cost with:

```bash
python scripts/benchmark_durable_compute.py
```

The benchmark tests 1K, 10K, and 100K transition quanta by default and measures a lightweight metadata checkpoint plus a sparse tape delta. Tune for the deployment host: the goal is to make checkpoint overhead small relative to useful work while keeping cancellation latency and queue fairness responsive.
