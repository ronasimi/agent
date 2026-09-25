# Architecture

This document describes the autonomous runtime with one main model and a tiny stateless System-1 routing model. Older dated design documents describe earlier versions.

## Model and protocol

tools.config.load_config normalizes task inference consumers to agent.model and agent.main_options. The main runtime alias is agent-main:4b. Compatibility constants named FAST_MODEL, REPORT_MODEL, or COMPACTION_MODEL still refer to that main model. Tool selection is the exception: a resident qwen2.5:0.5b System-1 router receives only a compact current-turn shortlist and never conversation history or full schemas.

The default protocol is `qwen_xml`. Tool definitions are sent through Ollama's native `tools` request field so the model's Jinja template serializes them into its `<tools>` system block. The model then emits invocations using its XML grammar:

```text
<tool_call>
<function=calculate>
<parameter=expression>
19 * 23
</parameter>
</function>
</tool_call>
```

The harness parses complete XML envelopes, validates and schema-coerces parameter text, executes the selected function, and places the result back on the model wire as a user message containing exact `<tool_response>...</tool_response>` tags. Normal answers remain ordinary assistant text. The configured runtime never instructs this model to emit JSON action envelopes.

## Module responsibilities

| Module | Responsibility |
|---|---|
| al_agent/runtime.py | Stable public entry point and request-local dependencies |
| al_agent/turn_engine.py | Conversation serialization, history, observations, trace and UI adapters |
| al_agent/agent_loop.py | Bounded model/action/result iteration |
| al_agent/tool_session.py | Turn-local catalog discovery, activation, validation, and dispatch |
| al_agent/model_protocol.py | Stream consumption and provider wire normalization |
| al_agent/events.py | Frontend event context and bounded process locks |
| tools/catalog.py | Registration, duplicate detection, and registry snapshots |
| tools/executor.py | Lifecycle checks and existing isolated execution support |
| tools/executor_worker.py | Child-process invocation with conversation context |
| al_agent/background/ | Durable job execution using the same configured model |
| webui/chat.py | WebSocket orchestration and cancellation slots |

## Tool discovery

A cheap lexical prefilter narrows the registered catalog to at most eight compact name/description candidates. The 0.5B System-1 router then chooses one candidate (or no tool) using a stateless Ollama `generate` call. Only the selected task schema is exposed to the 4B model alongside the stable `tool_search`/`load_tools` controls.

`tool_search` returns compact metadata only; complete schemas never appear in its observation. A confident search replaces the active task schema with one router-selected schema. A low-confidence or failed router decision clears task schemas and leaves the 4B model with discovery controls so it can explicitly search/load a capability. `load_tools` remains an explicit escape hatch when the main model needs a named tool from compact discovery results.

Persistent global/context routing outcomes are stored in the main SQLite database and calibrate router confidence with a bounded adjustment. Infrastructure timeouts do not penalize a tool. Router prompt/request/candidate state is cleared in the turn `finally` path, while the learned SQLite calibration persists across runs.

Builtin providers continue to register tools declaratively. Add an implementation and a TOOL_SPECS entry under tools/provider_groups. No modification of the action loop is needed. Custom tools still use the existing validation and registration mechanisms.

## Execution invariants

1. The complete model stream must arrive before any selected action executes. Incomplete responses, token-limit truncation, malformed XML envelopes, and text after the final `</tool_call>` are rejected.
2. Duplicate XML parameters, missing parameters, wrong types, and unexpected arguments are rejected. XML parameter text is coerced only when the authoritative tool schema requires an integer, number, boolean, array, or object; string parameters are preserved.
3. Qwen XML/native-parsed batches are validated before execution. Calls run sequentially, and a failure skips later calls in that batch. Dependencies across tool results should use separate model turns.
4. Each accepted call has an assistant call record and a corresponding result record. Interrupted history receives unknown-outcome placeholders rather than replaying the operation.
5. Tool exceptions and results go back to the same model. Repeated failures, cancellation, and exhausted budgets end with an explicit termination message.
6. An exception from a mutating tool prevents an identical invocation from being retried within the same turn. This is not durable exactly-once delivery; operators and tools must handle idempotency across turns and process crashes.
7. Inference locks are released before tool I/O. Conversation locks preserve serialized history updates.
8. Turn-local schemas and conversation context are not shared across concurrent chats.

Protocol branches, validation, authorization hooks, and resource checks remain deterministic. They enforce execution rules; they do not choose task intent, tools, or fallback workflows.

## History and observations

Storage keeps logical assistant/tool roles and local correlation IDs. At the Ollama boundary, tool result records are converted into exact `<tool_response>` user-message envelopes required by the Qwen template, preventing protocol-only messages from polluting saved chat history.

The loop reloads recent raw history after acquiring the conversation lock. It trims complete older turns when its estimated context budget is exceeded and refuses to drop the current request. Context accounting is an estimate, not a model-specific tokenizer guarantee. Full tool observations are stored separately; model-visible results are bounded and older results retain retrieval handles.

Working state remains an observation journal for existing diagnostics. Legacy requirement and recipe data structures remain available to their tools, but they no longer drive foreground selection or finalization. Memory and older history retrieval are model-selected tools; no separate embedding model is used.

## Background work and interface controls

Durable jobs retain their explicit checkpoint state machines, timeouts, evidence checks, and cancellation behavior. Choosing a research job does not cause any model swap. Report generation, compaction, and custom-tool generation reuse the all-purpose model. Background inference yields to foreground work at the existing arbitration boundaries.

Slash commands remain explicit user interface controls outside the natural-language loop. The websocket acknowledgment precedes workspace scans, and scans run off the event loop. Run slots are cleaned even if acknowledgment or preflight fails. Frontend event names and media rendering remain compatible.

## Limits

A blocking tool remains subject to its own timeout or isolated executor; the turn deadline is checked at execution boundaries and while model chunks arrive. Transport read timeout bounds stalled reads. Stop does not promise immediate preemption of every third-party operation.

The default text-only main model cannot inspect pixels. The 0.5B router and 4B main model can still choose an incorrect capability or produce an incorrect answer. The live smoke test and opt-in model tests must be run on the target Ollama deployment.
