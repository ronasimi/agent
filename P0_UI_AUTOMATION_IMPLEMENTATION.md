# P0 UI Automation Implementation

Implemented the complete P0 browser/UI automation foundation identified in the UI/task-understanding review.

## Changes

- Added a persistent Playwright Chromium runtime on a dedicated asyncio thread.
- Browser contexts are automatically scoped to the harness `conversation_id`, preserving cookies, SPA state, scroll state, form contents, and tabs within a conversation.
- Added one small typed `browser_step` tool supporting `observe`, `navigate`, `click`, `type`, `select`, `scroll`, `key`, `back`, and `verify`.
- Added semantic UI snapshots with persistent element refs, roles, accessible names, values, enabled/checked/selected/expanded state, viewport status, and bounding boxes.
- Added state-version validation. DOM/UI changes between calls advance the browser version; interactive actions using an older version return `STALE_OBSERVATION` rather than guessing.
- Added delta-first observations keyed by stable element refs, with automatic full-state fallback for initial observations or high-churn rerenders.
- Added normalized structured browser failures and integrated them with `classify_tool_outcome` / `StepFailureTracker`.
- Added machine-verifiable UI completion predicates through `browser_step(op="verify")`.
- Added a hard turn-engine outcome gate: after successful interactive browser actions, a final answer cannot claim task completion until a verify step passes explicit non-empty end-state checks.
- Refactored `take_web_screenshot` to reuse the persistent browser runtime rather than launching and closing Chromium for every screenshot.
- Registered browser UI intent selection and regenerated the builtin manifest to 236 tools.

## Verification

- `python -m compileall -q .` passes.
- P0/browser + loop-validator + registry targeted tests: **53 passed, 1 deselected**. The deselected pre-existing registry test imports the unavailable `ollama` dependency in this inspection environment.
- Direct Playwright smoke test passed against an in-memory dynamic form:
  - semantic refs emitted;
  - typing changed the UI/version;
  - old version rejected with `STALE_OBSERVATION`;
  - corrected click succeeded;
  - explicit `text_present` completion verification passed.
- Full pytest collection cannot run in this inspection environment because optional/runtime dependencies `ollama` and `ddgs` are absent. These are environment dependency errors, not P0 test failures.
