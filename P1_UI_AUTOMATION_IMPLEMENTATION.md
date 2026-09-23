# P1 UI Automation Implementation

This update layers the P1 UI/task-understanding and latency work onto the P0 persistent browser implementation.

## Implemented P1 changes

### 9. Explicit UI task-completion verification
- `browser_step(op="verify", checks=[...])` evaluates end-state predicates against live browser state.
- Interactive browser mutations invalidate prior completion proof.
- The turn engine keeps the P0 final-answer gate: interactive UI tasks cannot be finalized until machine verification succeeds.

### 10. Separate process and outcome scoring
- Browser responses expose separate `scores.process` and `scores.outcome` values.
- Durable trajectory rows store `process_success` independently from `outcome_success`.
- A valid action can therefore be distinguished from successful completion of the user objective.

### 11. UI-specific requirement types
The requirement ledger now supports verifier-owned UI requirements and does not allow ordinary click/type results to satisfy them. Supported checks include:
- `url_equals`, `url_contains`, `url_matches`
- `title_contains`, `title_matches`, `page_title_matches`
- `text_present`, `text_absent`
- `element_visible`, `element_not_visible`
- `element_value`, `element_value_equals`
- `element_checked`, `element_disabled`, `element_expanded`, `element_selected`
- `tab_open`
- `download_exists`

### 12. State-based stabilization instead of fixed sleeps
- Browser actions wait for DOM revision, URL/scroll state, document readiness, and pending-request quiescence.
- The normal UI path no longer pays an unconditional 1.5-second delay.
- Stabilization is bounded by a timeout and reported in browser metrics.

### 13. Reuse of the already-loaded page
- `browse_url`, `page_metadata`, and `page_links` reuse the current persistent browser DOM when their URL matches the active page.
- This preserves SPA/form state and avoids redundant network fetches.
- Existing fetch-based behavior remains as a fallback when no matching live page exists.

### 14. Semantic observations by default; screenshots on demand
- Normal UI reasoning stays semantic/delta-first.
- `browser_step(..., screenshot=True)` attaches a viewport screenshot only when pixels are useful.
- Compatibility screenshot tooling now reuses the persistent browser session.

### 15. Coordinate action fallback
- Semantic `ref` targeting remains the preferred click path.
- `browser_step(op="click", x=..., y=...)` is available as a visual/coordinate fallback when semantic grounding is insufficient.

### 16. Action + post-action observation fusion
Each interactive `browser_step` performs the complete transition in one tool round trip:
1. validate state/version and target,
2. execute action,
3. wait for UI stabilization,
4. capture canonical state,
5. compute the model-facing projection/delta,
6. return scores/metrics/result.

### 17. Causal browser actions remain serialized
- Browser mutation calls are deliberately excluded from read-only parallel tool batching.
- Existing parallel execution remains available for genuinely independent read-only work.

### 18. Inference lock release during external/browser I/O
- The conversation turn lock remains held for ordering.
- The global Ollama inference mutex can be released while Playwright/network-facing browser work runs.
- The inference mutex is reacquired only before the next model invocation.
- Model residency is not intentionally evicted by this path.

### 19. Per-stage browser latency telemetry
Browser results report bounded timing metrics including:
- browser/session acquisition and creation,
- pre/post semantic snapshot work,
- browser action/navigation,
- UI stabilization,
- projection and diff generation,
- screenshot capture when requested,
- total browser-step time,
- semantic snapshot cache hits/misses.

The turn engine emits browser metric events and stores the latest browser metrics in monitor state.

### 20. UI token-efficiency metrics
Browser steps estimate and report:
- full semantic snapshot tokens,
- actual observation/delta tokens,
- tokens saved by delta delivery,
- delta savings percentage,
- total/exposed/pruned candidate counts.

### 21. Browser-session semantic caching
- Each browser session caches the canonical semantic tree.
- Lightweight revision/URL/viewport probes determine whether a rescan is necessary.
- Tab/download state can update without forcing a full DOM semantic rescan.

### 22. Task-aware candidate pruning and expansion
- Model-facing observations rank controls by viewport presence, semantic role, task-text relevance, status/dialog significance, and pinned references.
- Default observations expose a bounded candidate set rather than the entire interactive DOM.
- `max_candidates` and `include_offscreen` provide an explicit expansion path when needed.

### 23. Durable `BrowserStateStore`
`tools/browser_state.py` persists canonical browser state by conversation, including:
- state version,
- projected/canonical UI state,
- machine-verification requirements,
- latest browser metrics,
- state timestamps.

The store uses the harness database path and fails open for browser execution if diagnostics storage is temporarily unavailable.

### 24. Complete browser trajectory recording
Each browser step records a compact trajectory entry with:
- operation and sanitized action,
- resulting state version,
- process success,
- outcome success when independently known,
- structured result/error,
- stage timings,
- token-efficiency metrics.

Sensitive typed values (for example password/secret-like fields) are redacted from the durable audit trail.

## Additional integration changes
- Browser tool schema now exposes coordinate fallback, candidate-expansion, and optional-screenshot controls.
- The builtin manifest remains at 236 tools and was regenerated after schema changes.
- Soak-test classification treats `browser_step` as a non-isolatable mutator/slow tool.
- Browser metric and score events are emitted from the turn engine for observability.

## Validation
- Full pytest suite: **573 passed, 1 skipped**.
- Python `compileall`: passed.
- Builtin manifest consistency: **236 tools, current**.
- Architecture checker: passed.
- Direct Playwright smoke test covered observe -> type -> stale-state rejection -> coordinate click -> outcome verification -> optional screenshot.
- Smoke-test no-change delta reduced the example observation from roughly 241 estimated tokens to 40 estimated tokens (~83% reduction).

The full test suite was executed in this inspection environment with temporary external import shims for unavailable `ollama` and `ddgs` Python packages. Those shims were not added to this repository.
