# Interactive Turn / TTFT Refactor — 2026-09-22

> **Historical status:** This is a point-in-time engineering/review record and is intentionally preserved as written. Model roles, tool counts, test totals, limits, and runtime behavior may have changed since this revision. For the current harness use `CURRENT_STATE.md`, `README.md`, and `ARCHITECTURE.md`.


## Execution path

1. `webui/chat.py` accepts a message, resolves its conversation, installs an event/cancellation context, and dispatches `runtime.handle_user_turn()` on a worker thread.
2. `al_agent/runtime.py` is a thin dependency-injection facade over `turn_engine.handle_user_turn()`.
3. `turn_engine.handle_user_turn()` takes a per-conversation turn lock, refreshes canonical history, persists the user message, derives task/fact frames and completion requirements, selects bounded tool schemas, and initializes working state.
4. Deterministic pre-grounding runs for current facts (time, weather, news, market, host/network/repository state, etc.) before a draft answer is generated. Successful simple fact requests can finalize here without Ollama.
5. Only if model inference is needed does the turn build the active prompt, acquire the global Ollama inference lock, evict an incompatible report-model residency if necessary, and start the streamed main-model request.
6. `model_protocol.consume_chat_stream()` normalizes streamed content/thinking/tool calls, measures first-token/first-visible times, applies the pre-release policy-leak guard, and returns one normalized capture.
7. The turn engine validates native tool calls, executes bounded batches, records observations/requirements, refreshes only the prompt state/tool schemas that changed, and iterates until finalization or a bounded recovery path.
8. Grounding gates reject factual finals whose observations do not satisfy requested fact types/scopes. Loop validators are used only for stalled/final-recovery cases.
9. The Ollama lock is released before post-turn compaction/bookkeeping; the per-conversation lock is released after those operations so same-chat history ordering remains deterministic.

## TTFT findings

### Global inference lock was acquired too early

Before this refactor, the cross-process Ollama lock covered history refresh, recipe lookup, profile/memory lookup, tool selection, working-state writes, deterministic web/weather/news retrieval, prompt assembly, model inference, tool execution, and post-turn work. A slow network preflight in one conversation therefore blocked unrelated interactive conversations from even entering Ollama.

The lock is now deferred until an actual model call is required. A separate per-conversation lock preserves history/state ordering while allowing unrelated turns to complete local/network preflight concurrently.

### Deterministic turns unnecessarily entered the model queue

Simple structured requests such as current time or a directly formatable grounded weather result previously paid the inference lock/model-residency cost even when no generation was ultimately performed. Deterministic fast paths now complete without acquiring the Ollama lock.

### Prompt construction was performed before deterministic retrieval

The active prompt and tool-schema token estimate were built before pre-grounding, then rebuilt after pre-grounding/schema pruning. Prompt assembly is now deferred until after deterministic fast paths. Model-backed turns build the initial prompt once from the post-grounding state.

### Working-state prompt rendering read SQLite twice

Canonical state and evidence rendering independently loaded the same working-state row. `render()` and `render_evidence()` now accept a shared state snapshot, so one prompt rebuild uses one state read.

### Grounding gates reread persisted state repeatedly

Within a serialized turn, newly verified evidence is already known by the orchestrator. Grounding checks now use a turn-local observation cache while still persisting every observation to working state for crash/restart and subsequent-turn recovery.

### Streaming protocol logic was embedded in the state machine

Chunk normalization, thinking events, tool-call accumulation, first-token timing, cancellation, guarded visible output, and policy-leak suppression were deeply nested inside `handle_user_turn()`. They now live in the pure `consume_chat_stream()` protocol helper and are unit-tested independently.

### Repeated deterministic finalization branches

Time, weather, news, market and encyclopedia fast paths repeated the same save/state/event logic. They now share one local `finish_deterministic()` path.

## Complexity change

A simple AST comparison of `handle_user_turn()` against the immediate pre-refactor baseline shows:

| Metric | Baseline | Refactored |
| --- | ---: | ---: |
| Function lines | 1,822 | 1,810 |
| Branch/control nodes | 485 | 458 |
| Call nodes | 962 | 909 |
| Total AST nodes | 12,062 | 11,624 |

This is not a performance benchmark; it is a structural complexity check. Real TTFT still depends mainly on model load state, prompt-prefill length, selected tool schemas, deterministic retrieval latency, and Ollama scheduling.

## New latency telemetry

Turn metrics now separate:

- `turn_queue_wait_ms`: waiting for same-conversation serialization.
- `turn_preparation_ms`: synchronous preparation/pre-grounding before requesting the model lock.
- `model_queue_wait_ms`: waiting specifically for the global Ollama slot.
- `ttft_ms`: Ollama request start to first model token/tool/thinking chunk.
- `answer_first_visible_ms`: user turn start to first visible answer content.

`queue_wait_ms` remains as the combined compatibility metric.

## Remaining intentional bottlenecks

- Current-fact pre-grounding remains synchronous because factual finalization is not allowed before qualifying evidence exists. Independent fact retrieval could be parallelized in a future change, but doing so safely requires explicit provider concurrency and cancellation budgets.
- Once the first model call begins, the global inference lock is still retained through the current model/tool loop. This avoids model-residency interleaving on memory-constrained hosts. Releasing it around long tool execution could improve multi-chat fairness, but should be benchmarked against Ollama runner churn before enabling it by default.
- Tool-capable generations intentionally delay visible prose because a streamed candidate may still resolve into native tool calls. Direct/no-tool turns retain true guarded token streaming.

## Validation

- Full pytest suite: 434 passed, 1 skipped.
- Focused new/refactored tests cover deterministic no-model-lock behavior, pre-grounding before model-lock acquisition, stream tool-call merging, leak suppression, cancellation, multi-fact grounding, and working-state rendering.
- Architecture checker passes when the validation environment supplies the same Ollama import dependency expected by the application.
- `git diff --check` passes.
## Follow-up: internal artifacts and simple multi-fact finalization

- WebUI artifact discovery now suppresses all `*.lock` files, including per-conversation and model-maintenance locks, and the client attachment parser refuses to render leaked lock paths inline.
- Simple fully-grounded combinations of current time, weather, headlines, and market quotes are rendered deterministically from structured observations instead of invoking the main model solely for formatting. Analytical requests continue through the model path.
- This specifically removes model queue/prefill/decode latency after successful pre-grounding for prompts such as `What are the current weather conditions, local news, and Brent crude price?`.
