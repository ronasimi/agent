# Latency and correctness review

> **Historical status:** This is a point-in-time engineering/review record and is intentionally preserved as written. Model roles, tool counts, test totals, limits, and runtime behavior may have changed since this revision. For the current harness use `CURRENT_STATE.md`, `README.md`, and `ARCHITECTURE.md`.


This round was driven by simulated turns rather than reading alone.
`scripts/simulate_turns.py` replaces the Ollama transport with a scripted client
whose behavior is a callable over the real request, so the fake model can only
call tools that were actually supplied. Roughly twenty turns were run across
plain chat, single-tool, multi-check diagnostics, weather (the hard grounding
gate), mutating batches, malformed calls, empty responses, repeated failure,
transport retry, and thinking mode.

No unhandled exception was found; the turn state machine is sound. Everything
below is a selection, loop-bounding, or prompt-layout problem.

## Time to first token

On a local server, TTFT is dominated by prompt prefill, and Ollama/llama.cpp can
only skip prefill for a prompt prefix that is byte-identical to the previous
request. Chat templates render tool schemas and the system prompt at the top of
that prompt, so anything that changes early costs a full re-prefill.

| Problem | Effect measured in simulation | Change |
| --- | --- | --- |
| Working state and evidence digest were emitted at message index 1, before the history | Reusable prefix collapsed to **0.1%** whenever the state changed; one multi-check turn re-prefilled 11,374 of 11,382 chars on iteration 2 | `build_active_messages(volatile_last=True)` emits them after the stable history (`context.volatile_blocks_last`) |
| Satisfied requirement schemas were pruned, and the set re-sorted pending-first, on every iteration | Both rewrite the top of the prompt: reuse fell to 0.1% on half the iterations of a four-check turn even though the tool *set* was unchanged | `minimize_schema_churn` prunes and reorders only in an iteration that must change the set anyway |
| Compaction reused the **main** model with `num_ctx: 4096` against the interactive `16384` | Ollama keys a loaded runner by context size, so every background compaction unloaded the warm interactive model and the next user turn paid a full load plus full prefill | `COMPACTION_OPTIONS` keeps `main_options.num_ctx` when the compaction model is the interactive model |
| The former terminal warm-up blocked startup and generated a full reply to the prompt `warmup`, caching a prefix no turn would reuse | First browser turn still paid model-load cost | `warm_model_async` reliably preloads weights from the Web UI; optional system-prefix priming is disabled by default and must be justified by `prompt_eval_cached_count` using `scripts/benchmark_warmup.py` |
| Idempotent harness control notes were re-appended every iteration | A stalled weather turn grew from 9.5K to 14.6K chars re-sending the same note | `append_control_note` appends a given note once per turn |
| `select_tool_schemas` had no relevance floor | "Explain what an agent harness does" selected 12 schemas including `execute_shell`/`execute_python` (matched on the word "agent" in their descriptions); "what is 17*3?" selected `temperature_sensors` and `notify_desktop` (matched on "through") and missed `calculate` | A tool qualifies on a name-token match or an aggregate score of 6; generic execution and `reload_tools` need their distinctive trigger term |
| The fallback finalizer was non-streaming | The user waited for the entire safety-limit summary | Streamed, emitting `assistant_delta` |

Measured reuse across consecutive requests of one four-check turn, before and
after: `0.1% / 75.1% / 0.1%` → `76.6% / 78.1% / 75.8% / 75.4%`. The two
remaining low-reuse points in the corpus are legitimate: the fallback finalizer
builds a deliberately different prompt, and the weather recovery genuinely has
to add `run_recipe` to the tool set.

One idea was measured and **rejected**: parallelizing the pre-inference recipe
preflight, memory search, conversation summary and tool selection. Together they
take 3–5 ms, which is noise next to prefill. A thread pool there would have been
complexity for nothing.

## Logic and schema fixes

- **The grounding gate could consume the whole iteration budget.** Discarding a
  candidate answer produces no new evidence, and the discard path had no counter
  of its own — `StepFailureTracker` never saw it because no tool calls were being
  made. An ungrounded weather turn burned 12 inferences. Now bounded by
  `grounding.max_candidate_discards` (3), which takes the same turn to 3
  requests and reports the exhausted budget as a validator event.
- **"network status" was not a requirement.** The `network_state` rule matched
  `interfaces|routes|health|state` but not `status`, so "check host memory, disk
  usage, network status and the git repo status" derived only two requirements
  and finalized claiming all four checks were done with `network_snapshot` never
  supplied or called. The simulated turn now runs all four.
- **`_finalize_after_limit` sent local bookkeeping on the wire.** It called
  `OLLAMA.chat` directly instead of through `ollama_wire_messages`, so
  `tool_call_id` was included. Harmless only because pydantic defaults to
  `extra='ignore'`.
- **A missing optional dependency broke all background job discovery.**
  `al_agent/background/research.py` imported `tools.pdf_generator` at module
  scope, so an absent `weasyprint` took down `discover_job_handlers` entirely,
  while the tool registry degraded gracefully for the same dependency. The
  import is now lazy and its failure degrades only PDF rendering.

## Tests

`tests/test_latency_and_prefix.py` adds 18 tests: prefix ordering and its token
budgeting, the legacy ordering escape hatch, byte-stability of an unchanged tool
set, pruning when the set changes anyway, the selection relevance floor, the
generic-execution rule, the network-status requirement, warm-up behavior and its
failure path, compaction context alignment, wire-format filtering, and two
turn-level tests covering the grounding budget and control-note de-duplication.

Full suite: 228 passing.

## Structured weather recovery follow-up

A live Web UI test exposed an important distinction between the grounding gate
and evidence acquisition: the gate correctly rejected a weather answer with no
weather observation, but the deterministic recovery path still depended on a
search engine plus scraping the first result. It also emitted a visible
`missing_evidence` validator event *before* trying recovery, making normal
retrieval look like a failure.

Changes in this follow-up:

- Added `geocode_location` and `weather_forecast` as narrow, read-only builtin
  primitives. They use Open-Meteo's keyless geocoding and forecast endpoints and
  return bounded structured JSON.
- Upgraded builtin recipe `weather.current_forecast` to version 2:
  `geocode_location -> weather_forecast -> compose_object`.
- Kept `web_search -> browse_url` as an independent fallback if the structured
  provider is unavailable.
- Removed weather from the generic tool-completion ledger. Weather completion is
  now owned by the stronger fact-grounding gate, so either structured weather or
  a linked search+browse fallback can satisfy the turn without contradictory
  hard-coded tool requirements.
- Fixed saved-location resolution. OOBE location data is now included directly
  in rendered profile context, and weather recovery can read the canonical
  stored `user_location` even when semantic-memory retrieval misses it. Mixed
  profile text + JSON memory context is handled correctly.
- A request such as `What is the weather for the next week?` uses an 8-day
  provider horizon (today plus seven future days), while explicit shorter
  periods are bounded appropriately.
- Pre-generation lack of weather evidence is treated as a normal retrieval
  trigger. The Web UI receives `tool_start` instead of an alarming
  `missing_evidence` validator card. `missing_evidence` is emitted only if the
  recovery itself fails or a later candidate tries to finalize ungrounded.
- The grounding validator recognizes both structured weather recipe provenance
  (`weather_forecast`) and the existing linked web-search/browse provenance.

Regression coverage includes mixed profile/memory location recovery, structured
weather recipe provenance, tool selection, provider composition, and an
end-to-end mocked next-week weather recovery.

Full deterministic suite after this follow-up: **244 passed, 1 skipped**.

## 2026-09-19: structured weather/news finalization fixes

A live harness run exposed two weak-model failure modes that deterministic unit tests had not yet covered:

1. A plain weekly weather request was grounded successfully, but the model reformatted the provider's parallel daily arrays into a malformed table, invented unsupported fields (for example humidity), and extended beyond the requested seven-day horizon.
2. A current-headlines request was not classified as a fact-retrieval task, and the model printed a JSON `Tool call:` block in prose rather than using Ollama's native tool-call channel.

The harness now:

- deterministically renders simple weather-display requests directly from structured provider fields, selecting only the requested dates;
- uses a dedicated `news_search` primitive backed by DDGS news metasearch for current headline metadata;
- classifies latest/recent/current news/headline requests under a hard `news` grounding type and retrieves evidence before model generation;
- deterministically renders simple headline-list requests from returned title/date/source/URL metadata;
- repairs a narrowly formatted, explicitly labelled textual tool-call envelope only for supplied read-only tools, never for mutating tools;
- adds regression tests for news grounding, headline rendering, weather-horizon enforcement, unsupported-field avoidance, and textual read-only tool-call repair.

This both improves reliability and removes a full LLM generation from common weather/headline queries, reducing perceived TTFT for those paths.

## 2026-09-22 follow-up

Additional latency/correctness hardening closes two same-runner transport gaps:

| Issue | Root cause | Fix |
|---|---|---|
| Fast validator could reload the main Ollama runner when `fast_model == model` | `tool_loop_validator.options.num_ctx` could override the already-aligned fast context with a smaller value | `state.py` now forces both `FAST_OPTIONS` and final `LOOP_VALIDATOR_OPTIONS.num_ctx` to `MAIN_OPTIONS.num_ctx` whenever the model identity is shared |
| Main/compaction requests could wait on an unbounded client transport | The main `Client` and inline compaction `Client` omitted explicit timeout values | Added configurable `model_transport.timeout_seconds` and `worker.compaction_timeout_seconds` bounds |

The read-only parallel tool-batch path remains unchanged; timeout-decorated registered tools now pay subprocess isolation only when a tool declares a harness timeout, avoiding a blanket latency penalty on the common tool path.
