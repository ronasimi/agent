# Al Agent runtime audit — 26 September 2026

## Outcome

The corrected repository preserves the single-model design and repairs evidence gating, history retention, retrieval, routing continuity, cancellation, parser handling and WebUI recovery. The final suite reports **704 passed, 0 failed, 5 skipped** (709 collected; 10.42 seconds). All five requested scenarios pass through the production engine with scripted Qwen XML and fixture provider responses. Architecture, durable-computation contract, manifest, Python compilation, critical lint and JavaScript syntax checks pass.

One existing diagnostic remains **failed**: the configured 250,000-token whole-repository source-size gate. The baseline already measures 769,738 estimated source tokens. Its limit was not raised and source/history was not deleted to pass it. The packaged measurement is in `audit-evidence/source-size-after.json`.

Measured deterministic routing median fell from **4.3963 ms to 1.7159 ms** (61.0%) across 240 calls per revision. Prompt assembly median increased from **0.5496 ms to 1.1532 ms** with the stricter shared estimator. These are local Python measurements, not Ollama or end-to-end speedups. No Ollama server was reachable at the configured address; live prefill, model quality, actual token counts, TTFT and GPU residency remain unmeasured.

Both supplied source archives have SHA-256 `abea1308fc74476283e857e3fb3690dcd5d56557c6314b959699ce5f37e6b7c7`. The package contains the complete original source/configuration/assets, the corrections, regression tests, this report and supporting evidence. Generated caches, local Git bookkeeping and runtime databases are excluded.

## Architecture traced

The WebUI enters `webui/chat.py:_run_turn`, delegates through the runtime facade to `al_agent/turn_engine.py:handle_user_turn`, takes the conversation lock, refreshes history, resolves profile queries, constructs context and runs the deterministic catalog prefilter. A turn-local `ToolSession` supplies the selected schemas plus discovery controls. `agent_loop.run_loop` acquires the global inference lock for each model request, parses the response and invokes the canonical registered-tool executor. Tool observations, working state, raw history and State Tape are persisted before terminal cleanup.

| Component | Assessment after correction |
|---|---|
| Model | `agent-main:4b`, `num_ctx=16384`, `keep_alive=-1`, Qwen XML, `think` wired, thinking off by default. **The alias is not a verified parameter count**: configuration points `model_source` at `hf.co/empero-ai/Qwen3.8-9B-Distill-GGUF:Q4_K_M`. No weights or deployed template were inspected. |
| Router / registry | 240 catalog entries after adding `gmail_inbox_counts` and `search_observations`; no routing LLM. At most eight initial candidates, 16 active tools, 20,000 active-schema characters. Catalog availability and per-turn schema activation are distinct. |
| Prompt | Stable policy leads the system message; bounded tape/profile data follow it. Four natural conversation turns follow the system message, then the current request. Active schemas travel in Ollama's tools field. This deliberately retains the existing prompt order rather than rewriting it to match an illustrative diagram. |
| Memory | Recent user/assistant content normally capped at 1,400/900 characters, with exact-evidence handles added separately. Six recent tape entries, three unresolved entries and a 2,400-character rolling summary are prompt bounds, not database retention limits. Full objectives, archived tape rows and raw observations remain stored. |
| Evidence / working state | The production engine now invokes the existing fact validator, with added Gmail/profile/calculation/file/temperature/history contracts. Required facts are updated after observations and checked before completion. Source/type/scope checks are deterministic; they do not prove every sentence of a generated answer. |
| Scheduler | The replacement foreground loop is intentionally model-directed and bounded. Legacy structured scheduler/requirement APIs remain in the repository but are **not a foreground planner**. The fact ledger is now connected; no claim is made that all legacy scheduler functions participate in normal chat. |
| Execution | Argument/schema checks and lifecycle hooks precede execution. Timeout-decorated tools use killable subprocesses. Existing workspace/path and public-URL checks remain; explicit local shell/Python/network capabilities retain their documented broad powers. |
| Durable computation | Registered `start_computation`, status and cancellation tools hand work to the background worker. Sparse mutable tape, branching transitions, checkpoint/resume and repeated cooperative quanta remain intact. Zero global step limit means no artificial total-step ceiling; finite hardware, cancellation and configured resource limits still apply. |
| Persistence / locking | SQLite WAL, indexes and per-conversation scoping remain. Same-conversation turns serialize across processes; the global inference lock excludes tool I/O. Tape rollup now commits its summary, watermark and archive flags atomically. UI history caches now expire after 250 ms across external writes. |
| WebUI | Plain/thinking streams, structured buffering, activity grouping, history/sidebar, artifacts, files drawer, jobs and diagnostic report paths were inspected and covered by existing tests. Reconnect now polls turn status, retains cancellation and reloads persisted history after completion without resending the request. |

## Fixes and regression coverage

Test names below are in `tests/test_runtime_audit.py` unless another file is named. Existing tests also cover the unchanged guards and streaming behavior.

| Priority / subsystem | Root cause → correction | Regression evidence |
|---|---|---|
| P0 completion | Production used a narrow discovery/routed-tool blocker instead of the fact validator → connect per-domain grounding and the persistent fact ledger; completion cannot clear pending facts. | `test_production_turn_rejects_unsupported_final_without_discovery`, recovery/multidomain tests, `test_working_state_cannot_complete_with_pending_facts` |
| P0 control-plane confusion | Catalog payloads could resemble facts → exclude discovery controls regardless of claimed fact metadata; reject downstream syntax as capability queries. | `test_control_metadata_never_satisfies_domain_facts`, `test_catalog_rejects_downstream_queries` |
| P0 factual display | A rejected answer could appear transiently while streaming → suppress factual prose until its required evidence exists; preserve thinking/tool events and stream grounded final prose. | Rejection test uses Qwen XML mode; `test_grounded_qwen_fact_answer_still_streams` |
| P0 active evidence | Old active results were shortened after two iterations without dependency knowledge → remove that destructive rule; keep bounded result previews and durable handles; refuse safely at the hard ceiling. | `test_active_results_not_shortened_after_two_iterations`, soft/hard budget tests |
| P0 persistence | Rollup deleted tape records and could cut provenance/IDs; similar partial work could resolve a larger failed task → retain archived rows, reserve whole handles, keep full objectives, use exact normalized retry matching, transact rollup. | Tape handle, rollup, unresolved-objective and partial-task tests; existing `test_state_tape.py` |
| P0 isolation / cleanup | Observation diff omitted conversation scope; deleting a conversation left tape data → scope both reads and remove associated tape rows on explicit deletion. | Observation diff and conversation-deletion tests |
| P0 side effects | Exception-based uncertain mutation protection missed returned `outcome_unknown` packets → track both forms and block the identical mutation during that turn. | `test_unknown_mutation_return_blocks_repeat`; existing exception-path tests |
| P1 history | A 100-row history query could consist almost entirely of tool activity → select the last four complete user turns directly; strip historical schemas/XML/raw tool JSON; preserve identities across repeated projection. | Four-turn/360-tool-row regression, source-identity regression; `test_context_tiers.py`, `test_state_tape.py` |
| P1 exact recall | Compaction removed visible handles and no tool searched the observation archive → attach handles to the one recent conversational copy, retain a tape fallback on soft eviction, add scoped/paginated `search_observations`. | Search/isolation, rehydration and soft-eviction tests; scenario C |
| P1 historical grounding | CPU recall was classified as a fresh weather/host request; the old routed-tool blocker then rejected valid rehydration → distinguish historical facts, preserve original provenance through pagination and let the production evidence validator be authoritative. | Historical domain/page tests; scenario C recovers original data after seven unrelated turns |
| P1 routing | Short chains lost their topic; assistant prose could bias hints; explicit switches inherited stale affinity → use bounded prior user intent and nearest tape objective/capability, retain the original topic across chains, reject explicit new domains. | `test_explicit_topic_switch_ignores_gmail_affinity`, `test_deterministic_router.py`, scenarios B/D |
| P1 profile | Explicit profile phrasings missed the deterministic path; broad matches swallowed compound requests → recognize direct reads/location, exclude compound external work and record verified profile observations. | Profile query/compound/context regressions; scenario A; existing profile tests |
| P1 Gmail counts | Search results expose an estimate, not an exact inbox total → add read-only INBOX label counts; reject missing counters instead of assuming zero; label search estimates and require inbox/unread query scope. | Gmail count/scope tests; scenario B |
| P1 budget | Telemetry and admission used different estimates → share conservative wire/lexical estimation plus template overhead; keep 8,192 soft and 13,824 hard estimated-token bounds. | Soft compaction, hard refusal, active transaction and telemetry tests; scenario E |
| P1 parser | Parameter regex stripped significant whitespace; incomplete/conflicting XML/native calls could be accepted; retry output could linger → preserve argument body whitespace, reject incomplete/mismatched calls and reset invalid visible output. | Whitespace/mismatch/incomplete regressions; existing split-marker, multi-call, thinking/prose tests |
| P1 stream cancellation | Waiting for the first chunk blocked cancellation; producer queue was unbounded; closing an executing generator could mask the real timeout → poll cancellation, bound the queue, let the pump close its iterator and preserve the original failure. | Prefill cancellation regression; existing first-chunk/idle timeout and stream tests |
| P1 reconnect | A failed WebSocket send removed the Stop handle while the worker continued → drain offline until completion, expose status, recover history and use HTTP cancellation. | Backend disconnect/cancel test and executable Node recovery test |
| P1 stale history | Process-local history cache never expired after another process wrote rows → add the same short TTL used by context caching. | `test_history_cache_expires_after_external_writer` |
| P2 performance | Re-tokenizing unchanged schema text and activation-order-dependent schemas → bounded tokenization memoization and stable schema ordering. | Ordering regression, existing router suite and paired benchmark |
| P2 metadata / scope | Clipped results lost fact metadata; compound web requests omitted their web requirement; sentence punctuation broke explicit URL matching → classify full raw results, retain independent web requirements and normalize terminal punctuation. | Full-result regression and scenario E |
| P2 diagnostics | Trace credentials could be written verbatim → recursive common-secret/Bearer redaction and mode 0600; record assembly, queue, model wall, visible latency, tool execution and parser postprocessing metrics. | Trace redaction/permissions and telemetry tests |
| P2 diagnostic validity | Process fixture embedded an invalid Python newline; a JSON simulator used the Qwen default and reported success without dispatch → escape the fixture, explicitly run the legacy JSON simulator and assert execution/answers. | Soak fixture test (platform inspection skip explained below), simulator regression; unavailable process I/O returns an honest structured failure |

The stale unknown-model capability test now tests both advertised capability values. Five executable file modes lost by initial ZIP extraction were restored from the original archive metadata; this was a packaging repair, not an original application defect.

## Scenarios A–E

These are orchestration simulations, not live-provider or model-quality evaluations. The real registry, router, XML parser, schema validator, fact gate, database and turn cleanup run; model responses and provider payloads are controlled fixtures.

| Scenario | Verified result |
|---|---|
| A — profile | Three specified questions consult the saved profile, produce three profile observations and make zero model calls. |
| B — Gmail | All four specified turns route to Gmail; the elliptical chain retains inbox intent; each final answer has a Gmail observation. Separate adversarial tests reject discovery-only “0 emails.” |
| C — history | CPU observation survives seven unrelated calculation turns. Search finds its handle; two reads recover the earlier temperature and exact critical threshold. No fresh sensor read substitutes for history. |
| D — switch | Gmail discussion followed by current CPU temperature selects host capability without Gmail contextual affinity. |
| E — compound | Time, geocoding/weather, calculation, host, web documentation and profile retrieval complete. Estimated input by model call: **2,920 → 3,219 → 3,417 → 3,622 → 3,876 → 4,041 → 4,225 → 4,414 → 4,692**. Active evidence remains available. Separate stress tests exercise soft eviction and hard refusal. |

## Latency and prefill

Same Python runtime, serial before/after runs, 240 calls over six repeated routing requests. Imports/catalog initialization are outside the timer; the first routed call is included. Assembly measures recent-history projection, policy/schema copy, wire conversion, telemetry and budget admission. It excludes database hydration, filesystem work and turn-level ledger writes. Full inputs and distributions are in `audit-evidence/benchmark-{before,after}.json`.

| Measurement | Revision | Median | p95 | Min | Max |
|---|---|---:|---:|---:|---:|
| Routing, ms | Before | 4.3963 | 5.2040 | 3.8985 | 7.5910 |
| Routing, ms | After | 1.7159 | 2.9926 | 1.5421 | 4.9761 |
| Prompt assembly, ms | Before | 0.5496 | 0.7832 | 0.4336 | 1.2350 |
| Prompt assembly, ms | After | 1.1532 | 2.0909 | 0.8268 | 3.5492 |
| Estimated input tokens | Before | 1,480.5 | 1,945 | 1,275 | 1,945 |
| Estimated input tokens | After | 2,464 | 3,058 | 2,104 | 3,058 |
| Estimated schema tokens | Before | 586 | 1,047 | 385 | 1,047 |
| Estimated schema tokens | After | 636.5 | 1,047 | 385 | 1,047 |
| Serialized message/schema bytes | Before | 6,413 | 8,408 | 5,545 | 8,408 |
| Serialized message/schema bytes | After | 6,647.5 | 8,429 | 5,566 | 8,429 |

The input-estimate increase primarily reflects a changed estimator; it is **not measured tokenizer growth**. Wire bytes are directly comparable and rise modestly, including the new Gmail count schema. Routing improves; assembly does not. No inference latency improvement is claimed.

| Runtime metric | Availability and interpretation |
|---|---|
| Routing / prompt assembly | Measured above; production events/traces also record them. |
| Queue latency | Instrumented separately for conversation/inference acquisition; no deployment contention benchmark. |
| Ollama prefill | Preserve `prompt_eval_count` and `prompt_eval_duration` from actual completion metadata; unavailable here. |
| Harness TTFT | First received model token relative to request start, including queue wait; unavailable for a live model here. |
| Visible TTFT | First emitted prose delta, after protocol/progress/evidence gates. Thinking is a separate stream. This measures backend emission, not browser paint/network delay. |
| Generation | Ollama `eval_count`/`eval_duration` and model wall duration retained; no live measurement. |
| Tool execution | Per-result execution time instrumented; scenario providers are fixtures, so their timings are not service benchmarks. |
| Postprocessing | XML/action decoding and pending-content flush instrumented; excludes full finalization/database/UI rendering costs. |

**Prefix reuse assessment (inference, not measured KV reuse):** static policy precedes volatile context; active schemas are consistently ordered. Warmup uses the resident model, matching context options and a tiny valid chat turn. It primes discovery schemas, not every possible routed schema set. Native template placement of tools and changing active sets can invalidate a prefix before message text. `keep_alive=-1` requests residency but does not prove it under memory pressure. Deployment traces and the actual Ollama template are needed before changing context size or claiming prefill gains.

The 16k window and 8k soft target are reasonable provisional settings for the observed 2.9k–4.7k compound simulation. This is insufficient evidence to tune hardware/model context. Arbitrarily large active turns still exceed the soft target; they fail safely at the estimated hard boundary. Truncated results retain exact retrieval handles. The harness does not guess when a dependent result has been consumed.

## Test and diagnostic record

| Check | Result |
|---|---|
| Initial baseline suite | 638 passed, 3 failed, 4 skipped (645 tests). Failures: executable extraction mode, stale capability expectation, malformed process fixture. |
| Final full suite | **704 passed, 0 failed, 5 skipped**, two dependency deprecation warnings; 709 tests, 10.42 seconds. No test selection exclusions. |
| Requested simulations | 5/5 pass; included in the full suite. |
| Existing JSON simulator | Executes calculation after discovery and after argument repair; 1/3/4 model calls, rather than the prior false one-call successes. |
| Architecture / universal computation | Both pass. Durable benchmark: about 2.61–2.73 million deterministic transitions/s median; checkpoint median 0.457 ms, 10 samples. This is local substrate throughput, not an LLM measurement. |
| Manifest | Current, 240 tools. |
| Compilation / JS syntax / critical lint | Pass. Ruff selection: `E9,F63,F7,F82`; this is not a claim that all repository style rules pass. |
| Whole-source size gate | **Fails before and after**; 769,738 baseline estimated tokens versus the unchanged 250,000 limit. Packaged result included. |

Skipped: three opt-in live Ollama conformance tests (server unavailable); one existing sandbox validation test (UID remapping prohibited); one same-UID process-inspection integration test (this environment does not expose the live child PID to psutil). The fixture now asserts that the child is alive before the platform skip. Its escaping defect is fixed. No failed test was silently dropped. WeasyPrint/HarfBuzz and Starlette/httpx emit dependency deprecation warnings; they did not affect these results. Runtime versions are recorded in `audit-evidence/environment.json`.

Reproduce from the repository root after installing its requirements and the test client's `httpx` dependency:

```bash
python -m pytest --capture=sys -q -o junit_family=xunit1
python diagnostics/check_architecture.py
python diagnostics/check_turing_completeness.py
PYTHONPATH=. python scripts/generate_builtin_manifest.py --check
python diagnostics/simulate_turns.py
python diagnostics/benchmarks/benchmark_runtime_audit.py --runs 240
python diagnostics/benchmarks/benchmark_harness.py --max-source-tokens 250000
```

The final command is expected to fail at the current source size. On the deployment host, additionally run `RUN_OLLAMA_LIVE_TESTS=1 python -m pytest tests/test_ollama_conformance_live.py -q` and collect real prompt/eval metrics. No live-provider access or deployment was performed in this audit.

## Answers to the 16 architecture questions

| # | Answer |
|---:|---|
| 1 | Availability and activation are separate in catalog/ToolSession. The old completion policy effectively confused routed schemas with required evidence; production now uses fact provenance instead. |
| 2 | A model can still propose an unsupported absence claim. For recognized retrieval domains the gate rejects it, directs discovery/retrieval and bounds repeated failure. Direct profile questions consult storage deterministically. |
| 3 | Discovery controls cannot satisfy the supported domain contracts, even with forged-looking payload fields. |
| 4 | Assistant-only tape records remain explicitly unverified; routing uses user intent. The fact gate does not accept prose as a current or exact historical measurement. General conversational text remains visible for continuity. |
| 5 | Yes. Whole handles survive projection/rollup; observation search recovers older handles after bounded summaries evict them. Reads/diffs remain conversation-scoped. |
| 6 | Four-turn continuity is preserved independently of tool-row count. Presentation/details can still be omitted by intentional per-message bounds; exact raw history and evidence remain retrievable. |
| 7 | Explicit topic changes now suppress stale affinity; the Gmail/CPU switch and follow-up chain pass. Ambiguous natural-language references remain heuristic, not universally resolved. |
| 8 | Yes, active evidence can exceed the 8k soft target. It cannot be admitted above the estimated 13,824-token ceiling; it is not silently truncated to fit. |
| 9 | Provisionally yes for these fixtures. Actual tokenizer, model/template and hardware measurements are still needed. The model alias does not verify a 4B weight size. |
| 10 | Yes as a conservative starting target; this audit does not establish an optimal threshold. |
| 11 | Only selected schemas are exposed and their order is stable. Sets still change with tasks; descriptions and candidate relevance are not globally minimal. KV reuse is unmeasured. |
| 12 | Profile answers avoid inference. Correct initial routing avoids discovery calls where confidence permits. Invalid protocol/evidence may still require bounded recovery calls; counts depend on actual model behavior. |
| 13 | Routing, compaction, profile lookup, evidence validation and durable computation are deterministic. No extra LLM was added to perform them. Foreground planning remains with the resident model. |
| 14 | Cross-process rollup, stale history and disconnect/cancellation defects were repaired. File locks serialize turns/inference; 250 ms caches allow brief eventual consistency. No proof of all possible multi-process interleavings is claimed. |
| 15 | Yes. Mutable tape, universal transition rules, durable checkpoints and unbounded repeated quanta remain connected to registered tools and worker handlers. Foreground limits are unchanged. |
| 16 | No, not every legacy subsystem is active in foreground chat. Router, ToolSession, tape, profile, fact ledger, executor and persistence are connected; the former structured planner/scheduler is not. Durable computation runs through its explicit background path. |

## Remaining risks

- Live Ollama/provider behavior, tokenizer accuracy, transport cancellation under real prefill, GPU residency and browser paint latency were not validated. The daemon stream pump may remain inside a blocked transport read until that read unwinds; foreground cancellation returns promptly, but immediate GPU abort is not guaranteed.
- Grounding covers recognized domains and checks source/type/scope, not full semantic entailment. Unknown phrasings, unsupported domains and inaccurate synthesis from valid evidence still need deployment evaluation. Retrieved prompt injection is mitigated by data treatment and tool guards, not mathematically eliminated; arbitrary execution capabilities are powerful by design.
- The 8k/13.8k budgets use estimates, not the deployed tokenizer. Very large active results may require explicit paging or a split task. Summaries, unresolved prompt entries and conversational snippets are bounded; exact records remain stored and require explicit retrieval when omitted.
- Raw history, observations and archived tape grow durably; no retention policy was imposed. New observation search scans matching conversation content on demand. Long-lived large databases need operational sizing and eventually indexed archive search.
- Credential-pattern redaction is not a general personal-data scrubber. Traces and bug reports can contain conversation content; existing pre-audit trace files were not retroactively rewritten. Mutation replay protection covers a turn, not cross-process/crash-safe idempotency at every external provider.
- The source-size optimization gate remains unusable at its current whole-repository limit. Revisit what that gate measures in a separate scoped change; deleting modules or relaxing the limit solely for this audit would hide the problem.

API contract references used: [Gmail label counters](https://developers.google.com/workspace/gmail/api/reference/rest/v1/users.labels), [Gmail message-list estimate](https://developers.google.com/workspace/gmail/api/reference/rest/v1/users.messages/list), [Ollama chat API](https://docs.ollama.com/api/chat). These support the counter and telemetry interpretation; all repository findings and performance values come from the supplied source and local execution.

## Conventional commit message

```text
fix(runtime): enforce grounded completion and preserve durable history

Wire domain evidence gates into the production turn engine, preserve exact
observation provenance through compaction, and repair contextual routing.
Harden streaming cancellation, uncertain mutations and WebUI reconnects.
Add exact Gmail inbox counts, archive search, regression scenarios and
repeatable runtime benchmarks with documented live-inference limitations.
```
