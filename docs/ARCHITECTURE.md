# Architecture

This document describes the autonomous runtime with one resident all-purpose model. Tool-candidate retrieval is deterministic and performs no model inference.

## Model and protocol

`tools.config.load_config` normalizes task inference consumers to `agent.model` and `agent.main_options`. The default runtime alias is `agent-main:4b`. Compatibility constants such as `FAST_MODEL`, `REPORT_MODEL`, and `COMPACTION_MODEL` still refer to that same model. There is no separate routing runner.

The default protocol is `qwen_xml`. Tool definitions are sent through Ollama's native `tools` request field so the model template serializes them into its `<tools>` system block. The resident model makes the semantic tool decision, emits the tool call, consumes the observation, and produces the final answer.

## Module responsibilities

| Module | Responsibility |
|---|---|
| `al_agent/runtime.py` | Stable public entry point and request-local dependencies |
| `al_agent/turn_engine.py` | Conversation serialization, deterministic candidate selection, observations, trace and UI adapters |
| `al_agent/deterministic_router.py` | Non-generative catalog ranking and bounded schema preselection |
| `al_agent/agent_loop.py` | Bounded model/action/result iteration |
| `al_agent/tool_session.py` | Turn-local catalog discovery, activation, validation, and dispatch |
| `al_agent/model_protocol.py` | Stream consumption, split first-byte/idle deadlines, and provider wire normalization |
| `tools/state_tape.py` | Three-tier prompt compaction, turn collapse, unresolved state, and deterministic rolling summaries |
| `al_agent/events.py` | Frontend event context and bounded process locks |
| `tools/catalog.py` | Registration, duplicate detection, and registry snapshots |
| `tools/executor.py` | Lifecycle checks and isolated execution support |
| `al_agent/background/` | Durable job execution using the same configured model |
| `webui/chat.py` | WebSocket orchestration and cancellation slots |

## Tool discovery

A cheap lexical/metadata scorer ranks the registered catalog before the first model call. Unrelated schemas receive zero routing relevance. Persistent global/context outcome statistics may slightly reorder already-related candidates, but cannot manufacture relevance for an unrelated tool.

A high-confidence, well-separated result activates one schema. Ambiguous or multi-intent requests activate a bounded candidate set (eight by default). The resident 4B model receives only those schemas plus the stable `tool_search`/`load_tools` controls and remains authoritative about whether and what to execute.

`tool_search` uses the same deterministic scorer and performs zero Ollama calls. It returns compact metadata only; complete schemas appear only in the next native `tools` field. Each search replaces the prior active task-schema set, preventing prompt growth across a turn. `load_tools` remains an exact-name escape hatch.

Routing outcomes are stored in the main SQLite database. Infrastructure timeouts are recorded without penalizing a tool.


## State Tape prompt lifecycle

Prompt history uses three tiers instead of replaying the SQLite chat transcript verbatim. Tier 1 is the active turn: the current user request, current routed schemas, raw current-turn call/result transactions, and transient reasoning. When the turn closes, native schemas and raw tool protocol are discarded from model-facing context. Raw rows remain durable in SQLite for audit/search only.

Tier 2 keeps at most three recent conversational turns as user/final-assistant surface text plus a bounded State Tape of compact deterministic outcomes. Discovery calls are excluded. Large JSON/XML results are collapsed before persistence to the tape, while the full tool observation remains retrievable from the observation store. Blocked or scheduler-incomplete work is retained as compact unresolved state and is re-routed on a later turn rather than carrying its old schema.

Tier 3 deterministically rolls older resolved tape entries into `conversation_context.summary` and advances the prompt-facing history watermark. No LLM call is required for normal compaction. This keeps the leading system policy byte-stable, prevents historical schema/result replay, and makes prompt growth bounded by configuration rather than conversation age.

## Execution invariants

1. The complete model stream must arrive before any selected action executes. Incomplete responses, token-limit truncation, malformed XML envelopes, and text after the final `</tool_call>` are rejected.
2. Duplicate XML parameters, missing parameters, wrong types, and unexpected arguments are rejected. XML parameter text is coerced only when the authoritative schema requires it.
3. Tool-call batches are validated before execution. Calls run sequentially, and a failure skips later calls in that batch.
4. Each accepted call has an assistant call record and corresponding result. Interrupted history receives unknown-outcome placeholders rather than replaying the operation.
5. Tool exceptions and results go back to the same resident model. Repeated failures, cancellation, and exhausted budgets end explicitly.
6. An exception from a mutating tool prevents an identical invocation from being retried within the same turn.
7. Inference locks are released before tool I/O. Conversation locks preserve serialized history updates.
8. Turn-local schemas and conversation context are not shared across concurrent chats.

## Warmup, residency, and prefill controls

WebUI startup primes only `agent-main:4b`, using the stable system policy prefix and discovery-control schemas when configured. Turn-specific inventory text is no longer appended to the system message; native schemas live only in the active `tools` field. No router warmup, router residency polling, or secondary model load exists. With `keep_alive: -1`, foreground requests reuse the same runner until Ollama or the host unloads/restarts it.

Foreground streaming separates the prefill/first-response deadline from the post-start stream-idle deadline. Prompt traces record message count, message/schema characters, estimated input tokens, and historical tool-protocol counts before every model call, plus harness first-token/first-visible timings when available.

## Limits

A blocking tool remains subject to its own timeout or isolated executor; the turn deadline is checked at execution boundaries and while model chunks arrive. The default text-only main model cannot inspect pixels. Deterministic retrieval narrows candidates but does not replace the main model's semantic decision, so tool-selection quality still depends on the resident model once schemas are supplied.
