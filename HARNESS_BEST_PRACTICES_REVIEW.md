# Small-Model Harness Review — 2026-09-19

This review focuses on a local Ollama deployment where a small main model must remain responsive, use tools reliably, recover from mistakes, and preserve enough state for long-running work. The goal is not to copy a large agent framework; it is to keep deterministic responsibilities in the harness and leave semantic decisions to the model.

## Executive summary

The harness is already substantially more robust than a minimal ReAct/tool loop. It has bounded per-turn tool exposure, typed primitives, reusable recipes, a requirement ledger, durable working state, a fast-model validator, deterministic grounding gates, context compaction, background jobs, and architecture/benchmark checks. Those choices are well suited to 2B–4B local models because they reduce the amount of policy the model must learn from prose on every turn.

The highest-value defects found in this review were at the Ollama protocol boundary, not in the high-level architecture:

1. streamed `tool_calls` were overwritten by the latest chunk instead of accumulated across all chunks;
2. tool results were sent with legacy `name` instead of Ollama-native `tool_name`;
3. local `tool_call_id` bookkeeping could leak into the native Ollama request even though the current Python `Message` type does not define that field;
4. a transient transport failure before the first streamed token consumed the normal failure path rather than receiving a safe bounded retry;
5. the self-optimization source-size gate was stale at 160k estimated tokens even though the unmodified repository was already about 232k, causing every optimizer baseline to fail before a candidate could be generated.

These issues are corrected in this revision. Protocol behavior is covered by regression tests, and the optimizer size ceiling now has modest headroom above the current source baseline.

## Comparison with lightweight/local-model harness patterns

### Hugging Face smolagents

smolagents emphasizes a compact multi-step agent with a bounded step count, optional planning intervals, final-answer checks, local model backends, and structured internal outputs. Its `ToolCallingAgent` is close to this project's execution style, while `CodeAgent` deliberately trades a larger execution surface for flexible code actions.

This harness already matches or exceeds the controls most useful for small local models: deterministic tool selection, explicit iteration limits, finalization checks, structured validator outputs, and typed tools. Keeping primitive execution in harness code rather than switching to generated Python is appropriate here because this agent also performs host/network operations and long-running automation, where a narrower side-effect surface is valuable.

### LangGraph / Deep Agents

LangGraph's strongest relevant pattern is explicit state plus explicit recovery paths: keep raw state, checkpoint transitions, distinguish retryable infrastructure failures from model-recoverable errors, and avoid burying control state inside formatted prompt text. Deep Agents adds provider/model-specific harness profiles, subagents, context offloading, and human approval for high-impact actions.

This project is already close on durable state, context offloading, subagent/background work, and validator-driven recovery. The new pre-first-chunk transport retry makes the infrastructure/model distinction clearer. Two capabilities remain useful future additions: model capability profiles (schema/tool-description overrides per model family) and a configurable approval tier for high-impact external mutations.

### PocketFlow

PocketFlow demonstrates the opposite end of the spectrum: tiny explicit graph/node primitives with retry behavior rather than a large framework. The relevant lesson is architectural, not an API to adopt: keep transitions simple and observable, and avoid making the model the workflow engine when deterministic code can own the transition.

This harness is necessarily heavier because it includes recipes, scheduling, storage, monitoring, browser/research work, and grounding. The current module split and architecture-size gate are therefore worth keeping. New orchestration logic should continue to live outside `turn_engine.py` when it can be expressed as a protocol, policy, recipe, or state helper.

### Ollama / llama.cpp-style local inference

For local models, the native inference protocol should be treated as a first-class compatibility layer. Ollama's current streaming guidance says to accumulate streamed fields, including every `tool_call`, and its Python examples send tool results with `tool_name`. Current `ollama-python` `Message` exposes `tool_name` but not `tool_call_id`. llama.cpp similarly invests in model/template-aware tool-call parsing and grammars because small models benefit from constrained, protocol-correct structure.

The new `al_agent.model_protocol` module centralizes these concerns so Ollama-specific wire behavior is no longer scattered through the turn state machine.

## Changes in this revision

### Native streamed tool-call accumulation

`turn_engine.py` now accumulates tool calls emitted over multiple streaming chunks instead of replacing prior calls. This fixes requests where the model emits more than one tool call in separate chunks.

### Correct tool-result messages

Tool observations are now persisted with Ollama-native `tool_name`. Old history that contains OpenAI-style `name` is upgraded when context is rebuilt, so existing sessions remain usable.

### Internal state vs. Ollama wire schema

The harness may still retain `tool_call_id` internally for transaction pairing/compaction, but `ollama_wire_messages()` strips local-only fields before a request is sent. This creates an explicit boundary between durable harness state and provider protocol state.

### Safe model transport retry

A bounded retry is allowed only if the Ollama request fails before yielding its first chunk. Once any thinking, content, or tool-call chunk has arrived, the request is never replayed; replaying at that point can duplicate user-visible output or side effects. Defaults are intentionally conservative: one retry with a 150 ms base delay and a 750 ms cap.

### Functional optimizer benchmark gate

The configured source-size ceiling is raised from 160k to 250k estimated tokens. The original repository measured about 231.5k, so the previous baseline gate could never pass. The new cap is intentionally close enough to remain a bloat guard while leaving room for a normal bounded optimization patch.

### Dependency and tests

The Ollama Python dependency is now `ollama>=0.5.2`, the release line that includes native `tool_name` support. Protocol behavior has dedicated unit tests independent of a running Ollama server.

## Existing design choices to keep

- Keep tool schemas small per iteration and expand them from requirements rather than exposing the full registry.
- Keep recipes as deterministic compositions of typed primitives; do not replace them with a larger monolithic tool API.
- Keep the grounding/fact-type gate deterministic and provenance-aware. A validator model should advise recovery, not be the only evidence check.
- Keep validator outputs schema-constrained and bounded. Small models are much more reliable when control decisions are parsed as typed data rather than free-form prose.
- Keep raw observations/state separate from presentation formatting and from the compact prompt projection.
- Keep transport/infrastructure retries outside the semantic iteration budget.
- Keep read-only parallelism narrow and mutation serialized so the model sees side effects before choosing another mutation.

## Recommended next steps

1. **Model profiles.** Add a small profile registry keyed by model/family with capability flags and optional tool-description/schema overrides. Validate profiles against `ollama.show()` at startup, cache the result, and warn rather than hard-fail when capabilities cannot be queried. This makes it easier to tune Qwen, Gemma, or llama-family behavior without branching the main loop.
2. **Risk-tiered confirmation.** Add `read`, `local_write`, `external_mutation`, and `privileged` risk metadata to tool definitions. Allow policies/recipes to require user confirmation only for the last two tiers. This provides a cleaner safety boundary without slowing ordinary host diagnostics.
3. **Live model conformance benchmark.** Add an optional benchmark that runs a fixed corpus of single-tool, multi-tool, malformed-argument, missing-evidence, and recovery tasks against both the main and fast models. Record tool-selection accuracy, valid-argument rate, unnecessary calls, recovery rate, TTFT, and total tokens. Keep it separate from deterministic CI so CI remains model-free.
4. **Continue shrinking `turn_engine.py`.** The file remains the largest coordination surface. As new behavior arrives, extract state transitions/recovery stages rather than adding more nested conditions.

## Validation performed

- `python -m compileall -q al_agent tools tests`
- targeted protocol/context regression tests
- complete project pytest suite using temporary import-only stubs for `ollama` and `ddgs` because this review environment could not install packages from the network; the stubs were outside the repository and are not included in the deliverable
- architecture-size/provider discovery check
- deterministic source-size benchmark

## Primary references

- Ollama streaming: https://docs.ollama.com/capabilities/streaming
- Ollama tool calling: https://docs.ollama.com/capabilities/tool-calling
- Ollama structured outputs: https://docs.ollama.com/capabilities/structured-outputs
- ollama-python tools example: https://github.com/ollama/ollama-python/blob/main/examples/tools.py
- ollama-python streamed tools example: https://github.com/ollama/ollama-python/blob/main/examples/gpt-oss-tools-stream.py
- ollama-python message types: https://github.com/ollama/ollama-python/blob/main/ollama/_types.py
- Hugging Face smolagents: https://huggingface.co/docs/smolagents/
- LangGraph: https://docs.langchain.com/oss/python/langgraph/
- Deep Agents: https://docs.langchain.com/oss/python/deepagents/
- PocketFlow: https://github.com/The-Pocket/PocketFlow
- llama.cpp tool calling: https://github.com/ggml-org/llama.cpp
