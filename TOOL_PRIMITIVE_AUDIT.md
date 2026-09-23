# Tool and Primitive Reliability Audit

> **Current-state update (2026-09-23):** This audit has been refreshed to the current tree. See `CURRENT_STATE.md` for model/deployment details and `ARCHITECTURE.md` for extension boundaries.

This repository-wide audit covers the generated builtin manifest and the shared execution, working-state, grounding, recipe/pipeline, requirement-ledger, and loop-validation paths used by the harness.

## Inventory

- 232 registered builtin tools
- 232 unique registered tool names
- No duplicate top-level primitive function definitions after cleanup
- Generated builtin manifest matches the source registry

## Reliability invariants implemented

1. **Grounding uses full-result proof, not clipped previews.** Working-state ingestion derives compact validator-only metadata from the complete tool result before `evidence_preview` is bounded. This includes returned market instruments, page scope, geocode candidates, weather coordinates, time-zone proof, and relevant source linkage.
2. **Returned data, not requested arguments, proves coverage.** Multi-instrument market requirements now require numeric quote rows for every requested asset. A call that asked for Brent and WTI but returned only Brent cannot satisfy the gate or requirement ledger.
3. **Volatile facts are turn-scoped by default.** Current time, market, news, live web, host, network, and repository state use current-turn observations unless the user explicitly requests reuse of previous evidence.
4. **Location-dependent facts are linked structurally.** Geocoding candidates can prove the relationship between a requested place and a forecast coordinate/time zone without relying on city names surviving text truncation.
5. **Recipe provenance requires successful execution.** Optional failures and skipped stages no longer count as successful source tools or satisfy requirements. Learned recipe traces contain only successful, non-skipped stages.
6. **Negative diagnostics are valid observations.** A successful network probe that establishes DNS failure, connection refusal, or unreachability is not misclassified as a harness/tool execution failure.
7. **Deterministic structured primitives fail closed on malformed output.** Grounding-sensitive JSON tools must return their expected top-level structure. Arbitrary non-empty text cannot satisfy host/network/repository/time/weather/market requirements.
8. **Empty results are classified by tool semantics.** Legitimate empty diagnostics such as no journal entries, no mDNS services, no official packages, no neighbors, or no local subnets are preserved as observations. `web_search`/geocode misses remain recoverable no-progress outcomes, while `news_search` performs its own bounded daily→weekly fallback and then treats a valid `[]` as a completed retrieval miss with no factual-news grounding.
9. **Executed arguments are persisted.** Working state records schema-normalized arguments actually sent to a tool rather than the raw model proposal.
10. **Recovery avoids blind duplicate retries.** Initial deterministic grounding recovery runs once and recomputes evidence after each recovery family instead of repeating the same provider call without a changed condition.
11. **Explicit invalid time zones fail instead of silently changing scope.** `current_time` rejects an invalid requested IANA zone rather than falling back to the host/configured zone.
12. **Shadowed primitives were removed.** Duplicate top-level network/process function definitions were deleted so the registered implementation is unambiguous and future fixes cannot land in dead code.
13. **Explicit tool errors are classified before structured-success validation.** JSON-returning primitives that emit `Error: ...` are now recorded as provider/tool failures rather than mislabeled `malformed_structured_result`, preventing argument-churn retries after a clear terminal error.
14. **Local-news recovery is bounded inside the primitive.** `news_search` no longer duplicates an already-scoped location in its query; DDGS `No results found` is normalized to an empty result, and a sparse daily result gets exactly one locality-heavy weekly fallback.
15. **Ambiguous-city news is scope checked for explicit conflicts.** A result explicitly referring to another city-country pairing (for example `London, England`) cannot satisfy `London, Ontario, Canada` merely because `Ontario` appears elsewhere in the story.
16. **News-only retrieval misses terminate deterministically.** After the bounded provider lookup returns no qualifying rows or a provider error, the harness reports that retrieval state directly instead of giving the main model repeated opportunities to mutate and retry the same search.
17. **Persistent requirements are separate from prompt rendering.** Working state retains up to 96 requirement entries while the model-facing requirement block is capped at 24, so large structured tasks remain auditable without injecting the full ledger on every inference.
18. **Discovery provenance is explicit.** Capability checks record whether a tool was already exposed or was discovered through `tool_search`; repeated workflow phases can attach evidence to a specific requirement key.
19. **Observation recovery is metadata-driven and bounded.** Preview `…[clipped]…` text is never treated as middle truncation. Genuine structured middle truncations are recovered with `read_observation` before evidence audits, and failed/non-progressing recovery becomes terminally unresolved instead of consuming the model-call budget.
20. **Recipe learning generalizes successful traces conservatively.** Task-defining literals can become shared parameters and derived strings can become `$template` references; operational constants and secret-like values remain fixed/excluded unless the objective explicitly requires otherwise. Fast-model naming hints are advisory and deterministically validated.
21. **Recipe retrieval does not depend on embeddings.** Recipe candidates are found locally with SQLite FTS5 and token-overlap scoring. `nomic-embed-text` is used only by optional semantic-memory paths.

## Validation

Current repository baseline after the latest routing, recipe-generalization, truncation-recovery, ledger/provenance, and WebUI artifact-filter changes:

- `pytest`: **520 passed, 1 skipped**
- `compileall`: passed
- WebUI JavaScript `node --check`: passed
- builtin manifest check: current (**232 tools**)
- duplicate registered tool names: none

Historical test totals in the dated engineering reports describe those earlier revisions and are intentionally preserved.

## News search scope and provider fallback hardening (2026-09-21)

Additional regressions discovered after the bounded-news change were corrected:

- A fresh generic request such as `latest headlines` no longer inherits the previous turn's local-news city.
- `default_location` is used for news only when the current request is explicitly local/deictic; it no longer silently localizes general news.
- Common topical phrases such as `AI headlines`, `Business news`, and `Technology news` are not classified as geographic locations from capitalization alone.
- General news queries are normalized to compact provider queries such as `latest news`; topical requests retain their topic.
- `news_search` now has one bounded provider fallback: DDGS first, then Google News RSS. A local daily miss broadens only the RSS fallback to one week.
- General-news no-result/provider-error messages no longer claim the request was local.
- Recovery remains inside the primitive, so the model cannot enter a query-rewording loop.

Regression coverage includes prior-local -> generic-news scope reset, topical-vs-location parsing, DDGS empty/error behavior, RSS parsing/fallback, and deterministic rendering.
