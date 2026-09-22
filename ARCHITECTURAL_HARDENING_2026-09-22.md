# Architectural Hardening — 2026-09-22

This follow-up verifies and implements the externally supplied audit findings against the repository as shipped. The supplied list contains **13** concrete items despite describing itself as 12.

## Verification outcome

| Finding | Verification | Implementation |
|---|---|---|
| Runtime facade global override race | Confirmed | Replaced cross-module mutation with request-local dependency overrides passed into `turn_engine` and explicit finalizer dependencies. |
| Validator fenced/preamble JSON parsing | Partially confirmed | Existing raw-decode already handled ordinary preambles and callers failed closed, so severity was overstated. Parser now safely tries whole JSON, fenced blocks, then bounded object starts. |
| Stream tool-call dict iteration | Confirmed | A single mapping is wrapped as one tool call before list normalization. |
| HTTP response tarpit/total duration | Confirmed as a total-deadline gap | Requests already enforces connect/read inactivity timeouts; `fetch_bytes` now also enforces a monotonic total deadline across redirects/body chunks. |
| Orphan timeout thread resource exhaustion | Confirmed in `tools/executor.py`, not `execute_python` | `execute_python` was already process-group-safe. Timeout-decorated registered tools now run in a killable child interpreter; orphan-thread accounting was removed. |
| PSI `ValueError`/`IndexError` | Confirmed in `tools/primitive_modules/system.py` | Malformed PSI rows are converted to structured error rows. |
| Background job blocks maintenance/heartbeat | Confirmed | Job execution is supervised while heartbeat, monitor, and maintenance continue. A hard runtime ceiling requeues/fails then exits the worker so Compose kills a wedged thread and restarts cleanly. |
| Recovery recipe stage-ID collision | Confirmed | Collision fallback now loops until an unused ID is found. |
| Same-model validator `num_ctx` thrash | Confirmed | Fast and final loop-validator contexts are forced to main `num_ctx` when `MODEL == FAST_MODEL`. |
| Compaction client has no transport timeout | Confirmed | Added configurable compaction client timeout. |
| Host filesystem traversal via `../../` | Confirmed in primitive `filesystem_usage` | Resolved target is checked with `commonpath` against resolved `/host`. |
| Main Ollama client lacks explicit timeout | Confirmed | Added configurable main transport timeout. |
| Descendant-kill test uses blind sleep | Confirmed as test flake risk | Test now captures descendant PID and waits deterministically with `psutil.wait_procs`. |

## Validation

- Targeted architectural/grounding/stress tests: **82 passed**.
- Complete deterministic suite: **420 passed, 1 skipped**.
- `scripts/check_architecture.py`: passed.
- builtin tool manifest: current at **223 tools**.
- Python compilation: passed.
- Ruff was not available in the validation environment, so no Ruff result is claimed.

Temporary `ollama` and `ddgs` import stubs were supplied only through an external `PYTHONPATH` during tests because those packages are absent from the validation sandbox. They are not included in this repository.
