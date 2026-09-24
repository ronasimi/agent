# P2 UI Automation Implementation

This release implements the remaining P2 items from the UI/task-understanding and performance backlog on top of the P0/P1 browser architecture.

## 29. BrowserGym-compatible benchmarking

- Added `tools/browser_benchmark.py` with an optional `BrowserGymRunner` and compact `BrowserGymAdapter`.
- The normal agent runtime does not import BrowserGym or Gymnasium at startup.
- `diagnostics/requirements-benchmark.txt` keeps benchmark dependencies separate from the low-spec production image.
- Added `diagnostics/benchmarks/benchmark_browsergym.py` and a dependency probe mode.
- The adapter preserves BrowserGym's structured accessibility/DOM observations and translates the harness's compact browser actions into high-level action strings.

## 30. Navigation/action safety boundaries

`browser_step` now classifies each operation as:

- `read_only`: observation, verification, navigation/back, scrolling, tab listing/switching, download waits.
- `reversible`: typing, selection, ordinary clicks, tab creation/close, non-submit keys.
- `consequential`: submit/confirm/purchase/payment/send/publish/delete/transfer/book/apply-style controls, including Enter on a focused consequential control.

The classification is computed by the harness from semantic UI state rather than accepted from model-provided labels.

## 31. Just-in-time pre-submit verification

Consequential actions are blocked unless they include machine-verifiable `checks`. Immediately before execution the harness:

1. refreshes canonical state;
2. rejects UI drift as stale;
3. evaluates the supplied predicates;
4. executes the action only if all checks pass.

This is separate from the existing final-outcome `verify` operation, which is still required to prove task completion after interaction.

## 32. Tab and popup lifecycle

`browser_step` adds:

- `new_tab`
- `list_tabs`
- `switch_tab`
- `close_tab`

Context-level page events record popups. When an action opens exactly one new page, deterministic recovery registers it, switches to it, and returns the resulting state without an extra planning round-trip.

## 33. Download lifecycle

Browser contexts accept downloads. Downloads are captured, bounded in session state, saved below:

`$AGENT_WORKSPACE/browser_ui/<conversation>/downloads/`

Each record includes ID, filename, source URL, state, path and size. `wait_download` provides a bounded wait, and action+observation fusion waits briefly when an action has just started a download.

## 34. Authentication-state detection

Semantic observations include a conservative auth state:

- `signed_in`
- `login_required`
- `session_expired`
- `mfa_required`
- `unknown`

Classification uses current route, password/MFA controls and visible account/login semantics. It is evidence-backed and deliberately avoids guessing identity.

## 35. DOM mutation/event hooks

A bounded in-page mutation journal records:

- mutation sequence;
- affected semantic ref;
- attribute changes;
- topology changes;
- added/removed interactive-node counts;
- input/change/history/scroll events.

For non-topology changes affecting a small number of known refs, the harness patches cached semantic rows instead of rescanning the entire interactive DOM. Topology/URL/viewport uncertainty falls back to a full semantic scan.

## 36. Viewport-aware pruning

Candidate ranking now uses viewport membership and distance. Visible controls are preferred, nearby off-screen controls remain available to avoid needless expansion after small scrolls, and distant controls require stronger task relevance unless `include_offscreen` is requested.

## 37. Hierarchical observations

Interactive elements are assigned to semantic regions:

- Dialog
- Navigation
- Header
- Main
- Form
- Sidebar
- Footer
- Page

Model-facing observations include compact per-region ref groups while retaining stable element refs.

## 38. Deterministic recovery

Mechanical failures are handled before another LLM round-trip when safe:

- one-version stale state with unchanged non-consequential target -> auto-rebase;
- detached semantic ref -> uniquely reacquire by stable role/name identity;
- hidden/obscured target -> scroll into view and retry;
- new popup -> register and switch;
- newly-started download -> bounded wait for completion.

Recovery events are emitted in browser results, monitor state and trajectory records.

## 39. Benchmark budgets

`BenchmarkBudget` enforces per-task ceilings for:

- model calls;
- browser actions;
- wall time;
- prompt tokens;
- output tokens;
- recovery attempts.

Limits can be supplied in code or via `AGENT_UI_BUDGET_*` environment variables. Budget violations terminate benchmark loops deterministically and are persisted with the run.

## 40. Regression dashboard

Benchmark results persist in SQLite. The new `/api/browser-benchmarks` endpoint returns:

- recent/previous success rates;
- p50/p95 task latency;
- average steps;
- average prompt/output tokens;
- live browser process success;
- live independently verified outcome success;
- optional BrowserGym availability;
- recent benchmark runs/errors.

The Web UI exposes these under **UI benchmarks**. `diagnostics/browser_regression_dashboard.py` provides the same data as JSON for CLI/CI use.

## Additional P2 hardening

- P2 safety/recovery diagnostics survive bounded trajectory serialization.
- Browser safety and deterministic recovery events are exposed through the existing frontend event stream and monitor state.
- BrowserGym benchmark imports remain lazy and optional, preserving production startup and memory behavior.
- The BrowserGym adapter is aligned with the current high-level API: pixel-delta `scroll(dx, dy)`, global `keyboard_press(key)`, element-scoped `press(bid, key)`, and index-safe tab close via focus + close when required.
- Benchmark budgets are checked immediately after model/tool budget consumption, so an over-budget action is never executed.
- Popup adoption has a post-stabilization recovery pass to catch Playwright page events that arrive just after the click returns, without adding a fixed delay to every click.
- Context-level request guarding remains active for popups and new tabs.
- Existing P0/P1 APIs remain backward compatible; `browser_step` is still the only model-facing interactive browser primitive.

## Validation

Validation performed for this release:

- full pytest suite: **584 passed, 1 skipped**;
- Python `compileall`: passed;
- architecture checker: passed;
- generated builtin manifest: current, **236 tools**;
- JavaScript syntax check for `webui/static/app.js`: passed;
- direct Playwright smoke passed for:
  - mutation-driven incremental typing;
  - consequential-action pre-submit blocking;
  - verified submit execution;
  - popup detection and automatic switching;
  - tab enumeration/switching;
  - download capture, persistence and bounded completion wait.

The smoke fixture used a local/in-memory test page; production SSRF protections were not relaxed in repository code.
