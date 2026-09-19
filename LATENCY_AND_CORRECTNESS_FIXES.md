# Latency and correctness review

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
| CLI warm-up blocked startup and generated a full reply to the prompt `warmup`, caching a prefix no turn would reuse | First turn still paid the system-prompt prefill; the Web UI had no warm-up at all | `warm_model_async` loads weights and primes the real system prefix at `num_predict: 1`, on a daemon thread, from both frontends (`warmup.*`) |
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
