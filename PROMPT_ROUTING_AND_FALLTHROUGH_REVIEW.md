# Prompt Routing and Fall-through Review — 2026-09-20

> **Historical status:** This is a point-in-time engineering/review record and is intentionally preserved as written. Model roles, tool counts, test totals, limits, and runtime behavior may have changed since this revision. For the current harness use `CURRENT_STATE.md`, `README.md`, and `ARCHITECTURE.md`.


## Scope

This review traced prompt classification, task-frame continuity, schema selection, pre-generation grounding, repeated-call suppression, validator recovery, recipe capture, and terminal tool failures. It focused on nearby prompts that should take different paths through the harness.

## Corrected failure paths

| Observed path | Failure | Corrected behavior |
| --- | --- | --- |
| `latest headlines in London ON` | Results could drift to London, UK or generic world news. | `London ON` is canonicalized to `London, Ontario, Canada`; search receives explicit query/location scope; locality ranking and grounding reject mismatched evidence. |
| `latest local headlines` after a scoped news turn | The follow-up could lose the prior city. | Same-intent deictic news follow-ups inherit the prior entity; a new topical request starts a fresh frame. |
| `refactor the weather validator` and similar implementation prompts | Lexical overlap could expose or trigger live-fact tools. | Deterministic intent classification suppresses weather, time, and news fact requirements and prunes their schemas unless a tool is explicitly requested. |
| Pre-grounded time/host/network/repository turns | The model could call an already-completed read-only tool again. | Recovery calls are recorded, satisfied schemas are pruned before generation, and exact completed repeats are suppressed. |
| Reminder scheduling without a user systemd bus | The loop retried, including with degraded arguments. | Backend-unavailable errors are terminal for the turn, partial units are cleaned up, and the schema is blocked from retry. |
| `yes, save it as report.md` | The phrase could be captured as a named recipe save. | Named recipe capture requires explicit `save recipe as ...`; the UI recipe action sends `save recipe`. |
| One-call or built-in weather workflows | The harness could suggest low-value or duplicate recipes. | Recipe suggestions require at least two stages and omit the built-in structured weather workflow. |

## Prompt-pair regressions

The tests now assert that each pair below follows a different route:

- `What is the weather in London ON?` → live weather; `refactor the weather validator` → implementation work.
- `What time is it?` → current-time grounding; `fix the current time tool` → implementation work.
- `latest local headlines` → scoped news continuity; `debug the headlines formatter` → implementation work.
- `latest local headlines` after London, Ontario news → inherit London; `latest AI news` → start a topical frame.
- `save recipe as morning brief` → recipe persistence; `save it as report.md` → artifact persistence.

These decisions occur before the main model sees tool schemas. The LLM still handles language generation and planning, but it is not asked to resolve deterministic collisions that the harness can settle safely.

## Fall-through invariants

1. A successful pre-generation fact call is registered as completed evidence before model generation.
2. A satisfied requirement cannot be fulfilled again by an identical read-only call unless the user explicitly asks to recheck.
3. Empty scoped-news results are no progress, not successful grounding.
4. Evidence for the wrong news location cannot satisfy the current task frame.
5. Terminal backend failures stop that tool path for the turn instead of mutating arguments and retrying.
6. Ephemeral recovery remains read-only, allowlisted, validated, and single-use.

## Validation

- Full test suite: **288 passed, 2 skipped**.
- Generated built-in manifest: **216 tools**, current.
- Architecture check: passed.
- Python bytecode compilation: passed.
- Offline control-loop simulations: single execution of each pre-grounded read-only fact tool, with no loop exceptions.
