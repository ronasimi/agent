# Structured Plan Throughput Hardening — 2026-09-24

## Incident

A 77-step structured-plan stress test remained active after roughly 15 minutes and had completed only the first nine requirements. The scheduler was progressing, but throughput was dominated by repeated main-model calls rather than tool execution.

## Root causes

1. Deterministic requirement preflight ran only for the initial scheduler step. After advancing to a new atomic requirement, the harness returned directly to the main model even when the active requirement ledger already named safe, read-only tools such as `cpu_info`, `memory_info`, or `filesystem_snapshot`.
2. After required tools completed successfully, the scheduler waited for another main-model response before marking the step terminal. Those intermediate mini-reports were discarded in favor of verified evidence, yet some consumed the full 384-token tool-turn allowance.
3. Selection-only requirements were already represented by a deterministic capability digest, but still spent a main-model call to reformat that metadata.
4. `tool_health` returned the complete 237-tool registry for aggregate health/count checks. The resulting ~37 KB observation triggered middle-truncation recovery reads even though the step needed only counts.

## Changes

- Re-run deterministic grounding/explicit-requirement preflight at every scheduler-step boundary before acquiring the Ollama inference lock.
- Advance typed scheduler requirements directly from verified evidence once their requirement/fact ledgers are terminal.
- Advance a step immediately after successful tool execution closes it; do not request an intermediate prose report.
- Resolve selection-only steps directly from the deterministic capability digest.
- Limit structured-plan tool-selection generations to 128 tokens when model routing is genuinely required.
- Add `tool_health(summary_only=True)` with aggregate registered/healthy/degraded/unavailable/readonly/mutating/missing-dependency counts and no full tool row list.
- Use summary mode automatically for deterministic scheduler tool-registry requirements.
- Continue using the main model for genuinely semantic/untyped requirements and the final whole-plan synthesis.

## Expected behavior

For the first nine requirements of the reported stress suite (runtime identity, tool registry, tool selection, host snapshot, CPU, memory, storage, pressure/load, and temperature), the scheduler now completes the individual requirements without intermediate main-model calls. A regression test verifies that only the final synthesis reaches the main model for that nine-step subset.

## Validation

- Focused scheduler/requirements/grounding tests: 77 passed.
- Full offline suite: 678 passed, 3 skipped.
- Builtin tool manifest consistency check: passed (237 tools).
- Python compileall: passed.
