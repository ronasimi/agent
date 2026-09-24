# Scheduler Timeout and Evidence Hardening — 2026-09-24

## Incident

A 77-step structured stress test progressed through several requirements and then stopped with the working state marked `blocked` while step 7 remained `PENDING`. The two immediately preceding main-model `tool_selection` requests timed out.

The incident exposed three coupled defects rather than a simple timeout configuration problem.

## Root causes

1. **Cross-step live-tail growth** — completed assistant/tool protocol from earlier scheduler steps remained in `turn_tail`. Even though future task text was isolated, prior tool transactions were repeatedly prefetched. On the affected local 4B model, prompt evaluation grew from roughly 2.2k tokens to more than 4k tokens and approached the 60-second transport idle timeout.
2. **Whole-plan termination on step-local inference failure** — the generic no-progress path called `complete_turn(blocked=True)` after two model inference failures. Structured-plan semantics require independent requirements, so a bounded failure in one atomic step must not terminate all remaining steps.
3. **Insufficient evidence contracts** — `filesystem_snapshot`, `process_snapshot`, `pressure_snapshot`, and `service_health` were all classified as generic `host_state`. A narrow filesystem observation could therefore satisfy a broad Host Snapshot grounding request. Several common scheduler steps also lacked explicit task requirements, allowing model prose to be accepted without the requested tool evidence.

A fourth contributing issue was that constraint extraction ran across the entire original structured suite. Step-local instructions containing phrases such as `do not`, `without`, or `read-only` were promoted to global constraints, unnecessarily increasing every model prompt and leaking later-step wording into the active execution context.

## Changes

### Atomic scheduler prompt boundary

When one scheduled step closes, the live `turn_tail` is cleared. Durable evidence remains in working state and observation storage, while the next prompt contains only the new active scheduler control note and any metadata-only capability digest.

Read-only repeat/no-progress signatures are also reset per step. Mutating-call signatures remain turn-global for safety.

On the first model transport/inference failure within a structured step, the retry is rebuilt from the compact durable state rather than replaying the expensive failed prompt.

### Step-local failure recovery

After the configured bounded main-model retry count is exhausted:

- the current scheduler step is marked `FAIL` with the model error;
- the scheduler advances to the next independent requirement;
- model no-progress counters are reset;
- the plan continues;
- only a failure during final synthesis or a genuinely fatal whole-runtime condition uses the whole-turn blocked path.

The same scheduler-local handling now applies to repeated invalid/empty model output, policy-leak suppression exhaustion, grounding exhaustion, truncated-observation recovery exhaustion, and browser outcome-verification exhaustion. These errors can invalidate one independent requirement without silently cancelling the rest of a long plan.

### Stronger completion/evidence gate

A fact-bearing scheduler step can reach `PASS` only when both:

- its explicit task requirement ledger is terminal without failed/blocked requirements; and
- the grounding ledger reports the required fact type as grounded.

Broad `host_state` grounding now requires `host_snapshot`. Narrow host diagnostics no longer satisfy the broad fact contract by tool name alone.

Operational scheduler steps with an empty explicit ledger cannot be accepted from prose alone when native tools are exposed. At least one successful active-step tool observation is required as a backstop for parser/catalog gaps.

### Requirement parsing additions

Explicit rules were added for:

- runtime environment / OS / kernel identity → `environment_summary`;
- uptime → `uptime`;
- hostname → `hostname`;
- CPU identity/topology → `cpu_info`;
- memory/swap counters → `memory_info`;
- tool registry/dependency health → `tool_health`;
- `host snapshot` phrasing → `host_snapshot`.

This prevents the early stress-test steps from passing on unsupported model summaries.

### Structured-plan constraint isolation

For a structured numbered suite, global constraints are extracted only from the preamble before the first numbered requirement. Step-local constraints stay attached to their scheduler step instead of becoming global prompt material.

### Working-state prompt compaction

In scheduler mode, `verified_observations` are no longer duplicated inside the canonical system-state JSON. Their excerpts are delivered through the separately bounded evidence block. The durable state still retains all configured observations for audit/recovery.

The evidence block is now **active-step scoped**: only observations matching the current requirement's tools/fact types are rendered. Explicit cross-check/reuse steps may receive at most two recent observations as context. Selection-only steps receive no observation digest. This prevents unrelated evidence from earlier requirements from rebuilding the same 4k-token prompt that caused the original timeout.

### Evidence-backed scheduler results

Operational scheduler PASS results no longer persist the model's prose as the durable factual result. Instead, the scheduler stores a compact representation of the explicit requirement ledger and its concrete evidence references/previews. Model-authored summaries remain appropriate for non-executing selection/explanation steps, but runtime facts cannot become authoritative merely because the model wrote a plausible table.

This directly prevents the incident's fabricated kernel/uptime/tool-count values from contaminating final synthesis.

CPU steps that explicitly request architecture now also require `environment_summary`, because `cpu_info` may expose CPU model/topology without a machine-architecture field.

## Regression coverage

Tests now cover:

- step-local timeout failure followed by scheduler continuation;
- prior tool-call/result protocol disappearing at a scheduler boundary;
- structured-plan global constraints excluding future step-local instructions;
- scheduler canonical state not duplicating observation metadata;
- broad host state rejecting filesystem/process/pressure-only observations;
- runtime identity, tool registry, host snapshot, CPU, and memory requirement derivation;
- existing prompt isolation and one-tool-call continuation behavior;
- active-step evidence filtering;
- 20-step provider-prompt boundedness;
- zero-tool protocol failure isolation;
- evidence-backed operational scheduler results that reject fabricated model claims;
- CPU architecture requiring environment evidence.

Validation: `673 passed, 3 skipped` in the complete test suite. The skipped tests require live external runtime integration.
