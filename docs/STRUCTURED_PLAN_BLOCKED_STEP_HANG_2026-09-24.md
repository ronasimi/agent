# Structured-plan blocked-step hang hardening — 2026-09-24

## Incident

A 77-step stress plan reached requirement 25 (Semantic Observation) and then stopped
making scheduler progress. `search_semantic_memory` returned `no_progress_result`
three times. The fast validator correctly returned `blocked/tool_unavailable`, but the
active scheduler requirement remained `PENDING`. The main model was subsequently
called repeatedly to restate the same blocker.

## Root causes

1. **Ambiguous semantic routing.** Browser/accessibility wording such as
   "interactive elements semantically" could select `search_semantic_memory` rather
   than the live browser semantic-state primitive.
2. **Missing validator-to-scheduler terminal transition.** A terminal validator
   result was recorded in diagnostic history but did not atomically close the
   active structured-plan requirement.
3. **Legacy no-tool recovery could still globalize a step-local blocker.** A
   blocked/finish recovery report reaching the no-tool branch could use ordinary
   whole-turn completion logic rather than scheduler advancement.

## Corrections

- Browser semantic/accessibility language now routes to `browser_step`.
- `search_semantic_memory` lexical selection now requires an explicit
  memory/memories/remember/recall concept; the bare word "semantic" is insufficient.
- Browser-session and semantic-observation requirements compile explicitly to
  `browser_step`.
- In a structured plan, a terminal stalled-step validator decision now immediately:
  1. marks any matching requirement blocked,
  2. commits the active scheduler step as terminal (`FAIL` with a `BLOCKED` reason),
  3. advances to the next independent requirement, and
  4. rebuilds the prompt at the new scheduler boundary.
- The no-tool validator recovery branch has the same step-local terminalization as
  a second guard.
- A validator `finish` may PASS only when the ordinary requirement/evidence gate is
  already satisfied; otherwise it is treated as a terminal step failure rather
  than unsupported success.

## Invariant

Once the validator returns a terminal `blocked` decision for a structured-plan
requirement, the main model must never receive another inference request for that
same requirement.

## Regression coverage

- semantic interactive-element requests expose `browser_step` and not
  `search_semantic_memory`;
- explicit semantic-memory requests still expose `search_semantic_memory`;
- browser session/semantic observation compile to `browser_step` requirements;
- three no-progress tool attempts followed by validator `blocked` terminate exactly
  that step, advance the scheduler, and do not produce a fourth blocked-step model
  call.

## Validation

- focused scheduler/browser/requirement/validator tests: 93 passed
- complete offline suite: 683 passed, 3 skipped
- Python compileall: passed
