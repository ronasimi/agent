# Prompt-to-Tool Argument Isolation — 2026-09-24

## Problem

A structured-plan audit step that asked the agent to *identify* the appropriate primitive without executing it was re-parsed as live work. Fact-like examples such as `current weather`, `network routes`, and `repository status` created grounding requirements and executable tool schemas. Separately, whole-prompt lexical target extraction could interpret prose such as `DNS requests` or `Network Path: Use ...` as tool targets.

This allowed surrounding prompt/control text to contaminate tool selection, requirement scopes, and potentially model-emitted selector arguments.

## Fixes

- Added explicit detection for non-executing tool-selection/capability-comparison requests.
- Selection-only steps now create no live fact frames, fact-grounding requirements, or executable requirement ledger entries.
- Selection-only steps receive a compact text-only capability candidate index instead of native executable schemas.
- Structured-plan mode no longer reparses the complete original prompt into a parallel model-facing requirement ledger; the scheduler is authoritative and only the active step ledger is rendered.
- Explicit numbered plan extraction removes following `PHASE` headings from the preceding atomic task.
- DNS/network-path/URL target extraction now rejects descriptive prose and strips Markdown punctuation.
- Added a pre-execution guard that suppresses tool calls when selector-like arguments contain copied structured prompt/control text.

## Validation

- Focused routing/grounding/protocol/scheduler tests: 81 passed.
- Full offline suite: 655 passed, 8 live tests deselected.

## Scheduler deterministic-finalization regression

The initial isolation fix exposed a separate completion bug: in structured-plan mode the global requirement ledger is intentionally empty, but the deterministic fact fast path still used that global ledger as its whole-turn completion predicate. A pre-grounded fact such as `current_time` could therefore emit a final answer and call `complete_turn()` while the active scheduler step still had another requirement such as `hostname` pending.

The correction adds two independent guards:

- Whole-turn deterministic fast paths are disabled while a structured scheduler is active; scheduled steps must flow through the normal active-step completion/advance path.
- `WorkingStateStore.complete_turn(blocked=False)` refuses to mark a turn complete until every scheduler step is terminal. Explicit blocked/error termination remains allowed.

Regression coverage reproduces a two-step plan whose first step needs both `current_time` and `hostname`: time is pre-grounded, hostname is still executed exactly once, the scheduler advances, and the final synthesis is emitted only after all steps complete.

Updated validation after this correction:

- Focused scheduler/grounding/completion suite: 69 passed.
- Full offline suite: 658 passed, 8 live tests deselected.
