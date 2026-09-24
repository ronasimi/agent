# Optimization Patch Review — 2026-09-24

This review compares the external optimization patch suite against the current
`agent-master-evidence-scope-hardened` codebase after the structured-plan
scheduler fixes. Recommendations were applied only where they improve the
current architecture without duplicating or weakening existing safeguards.

## Decision summary

| Recommendation | Decision | Current implementation / change |
|---|---|---|
| 1. Prevent element-ref collisions on hard reload | **Applied** | Ephemeral ref counter persists in `sessionStorage` with a reload gap; unload persistence added. |
| 2. Expand fact-type matching | **Applied selectively** | Existing implementation-aware classifier retained. Added `conditions outside`, `current clock time`/`time in …`, compound-news detection, Bitcoin/Ethereum aliases, and explicit ticker quote detection. Broad ambiguous terms such as generic `trending`, `cost`, or `how much` without an instrument were not adopted. |
| 3. Task-aware element pruning | **Already implemented / patch form rejected** | `project_snapshot()` already scores task relevance, bounds candidates, keeps pinned refs, and filters distant off-screen controls. Canonical snapshots remain complete so verification/recovery is not weakened. |
| 4. Element-text caching | **Not applied as proposed** | The harness already has a global mutation journal, cache-hit path, and `_incremental_snapshot()` that reserializes only affected refs. Per-element MutationObservers would duplicate observers and add memory/CPU overhead. |
| 5. Persistent stable refs | **Applied as recovery-only layer** | Snapshots now include a semantic `stable_ref`; bounded ephemeral→stable history and unique-match recovery reacquire controls after same-origin reload/framework re-render. Stable hashes are never trusted when ambiguous and never replace the primary `eN` action ref. |
| 6. Observation token accounting | **Already present; corrected/enhanced** | Metrics now separately account for canonical snapshot, model projection, and final delta. Added projection savings and full/projected element counts. |
| 7. Observation tiers | **Not applied** | Existing `project_snapshot()` + `semantic_diff()` already provide candidate-level pruning and delta-first detail reduction while retaining needed state fields. An additional heuristic tier could silently remove verification state and was not justified by current evidence. |
| 8. Regression tests | **Applied** | Added `tests/test_browser_optimization_review.py` for ref persistence/recovery, accounting, fact phrases, and market aliases. |
| Extract `turn_engine.py` subsystems | **Deferred** | The evaluation labels this a post-release maintainability refactor rather than a correctness fix. It is intentionally kept separate from this optimization patch to avoid a large behavioral rewrite. |

## Important implementation choices

### Canonical state remains complete

The patch suite proposes pruning elements during the JavaScript snapshot itself.
The current harness instead preserves a full canonical state and prunes only the
model-facing projection. This is intentional: verification checks, stale-ref
recovery, safety classification, and deterministic state deltas all benefit from
a complete canonical observation.

### Stable refs are hints, not authority

Semantic hashes can collide when a page contains repeated controls such as many
"Add to cart" buttons. Recovery therefore accepts a `stable_ref` only when it
maps to exactly one current element. Otherwise the existing role/name recovery
path must also be unambiguous or recovery declines.

### Fact matching remains capability-aware

The external regex proposal would classify broad words such as `price`, `cost`,
`trending`, and `how much` as live fact requests. That causes false positives in
code/document discussions and can create grounding requirements the harness
cannot satisfy. The implemented expansion keeps the existing implementation
request guard and only adds phrases with clear supported semantics.

## Validation

- Python byte-compilation: **PASS**
- Browser init/snapshot JavaScript syntax via Node: **PASS**
- Focused browser/grounding/market/task tests: **83 passed**
- Offline suite (`-k 'not live_'`): **648 passed, 7 deselected**
- Full offline collection required test-only `ollama` and `ddgs` import stubs in
  the sandbox because those runtime dependencies are not installed here. The
  stubs are outside the repository and are not included in the packaged result.
