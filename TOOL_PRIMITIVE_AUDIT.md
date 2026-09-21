# Tool and Primitive Reliability Audit

This repository-wide audit covers the generated builtin manifest and the shared execution, working-state, grounding, recipe/pipeline, requirement-ledger, and loop-validation paths used by the harness.

## Inventory

- 223 registered builtin tools
- 45 tool/provider modules
- 223 unique registered tool names
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
8. **Empty results are classified by tool semantics.** Legitimate empty diagnostics such as no journal entries, no mDNS services, no official packages, no neighbors, or no local subnets are preserved as observations, while search/geocode misses remain recoverable no-progress outcomes.
9. **Executed arguments are persisted.** Working state records schema-normalized arguments actually sent to a tool rather than the raw model proposal.
10. **Recovery avoids blind duplicate retries.** Initial deterministic grounding recovery runs once and recomputes evidence after each recovery family instead of repeating the same provider call without a changed condition.
11. **Explicit invalid time zones fail instead of silently changing scope.** `current_time` rejects an invalid requested IANA zone rather than falling back to the host/configured zone.
12. **Shadowed primitives were removed.** Duplicate top-level network/process function definitions were deleted so the registered implementation is unambiguous and future fixes cannot land in dead code.

## Validation

The audit added regressions for clipped market/page evidence, partial quote coverage, wrong encyclopedic subjects, current-time place/time-zone linkage, weather coordinate linkage, failed/skipped recipe provenance, malformed structured output, valid negative network diagnostics, and invalid IANA zones.

Final validation in the audit environment:

- `pytest`: 374 passed, 1 skipped
- `compileall`: passed
- builtin manifest check: current (223 tools)
- duplicate registered tool names: none
- duplicate top-level primitive functions: none

The audit environment does not provide the real `ollama` and `ddgs` Python packages. Minimal import stubs were used only to allow repository tests that do not exercise those external services to collect; those stubs are **not** included in this repository.
