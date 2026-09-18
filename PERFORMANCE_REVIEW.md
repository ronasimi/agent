# Agent Harness Performance Review

## Implemented

- Cache the turn-level system prompt and memory lookup instead of rebuilding them on every tool-loop iteration.
- Replace the model-facing full tool inventory with a compact policy; native Ollama tool schemas remain authoritative.
- Deterministically select a bounded, relevant tool-schema subset per user turn, while retaining a general fallback set.
- Reserve prompt budget for native tool schemas so the message context does not silently consume the entire configured `num_ctx`.
- Default normal turns to `think: false`; `/think on` remains available for explicit reasoning-heavy turns.
- Reuse the main model for context compaction by default. This avoids unnecessary main/fast model eviction on Ollama servers limited to one resident model.
- Add an explicit interactive-inference activity lease in SQLite. The background worker defers research while the interactive frontend is actively using the model, and also respects the existing post-turn cooldown.
- Lower the background worker CPU scheduling priority with `nice(5)` where permitted.
- Reduce the retry delay after a transient Ollama error from 1.0s to 0.2s.
- Add optional per-request Ollama prompt/eval timing and prompt-cache counters to the interactive UI.

## Validation

The updated Python modules compile successfully.

The repository test suite passes with the available local dependency stubs:

`10 passed`

A mocked end-to-end tool-calling turn also passed, including:

- native tool schema selection
- streamed tool call parsing
- tool execution
- second model turn
- interaction-active state assertion/clear
- `think=False` propagation

The execution environment did not have a live Ollama server, so real model-token latency and GPU throughput were not benchmarked here. The harness now displays Ollama `prompt_eval_count`, `prompt_eval_cached_count`, `prompt_eval_duration`, `eval_count`, and `eval_duration` so those metrics can be measured on the target machine.

## Static prompt-footprint result

Original harness:

- 47 tool schemas
- ~13.1 KB native tool-schema JSON
- ~6.6 KB model-facing system/tool-summary text

Updated harness:

- ~1.9 KB model-facing system/tool-policy text
- Typical selected tool-schema payloads measured at ~3.7–4.6 KB for representative CPU/web/reminder/network/package queries
- Tool schema prompt budget is now explicitly subtracted from the conversation-context budget

The combination materially reduces first-pass prefill work while preserving native Ollama tool calling.

## Follow-up: absolute workspace paths

A live harness trace exposed a path-normalization bug in `tools/workspace.py`: absolute paths such as `/app/workspace/tools/__init__.py` were passed through `os.path.join(WORKSPACE_DIR, filename.lstrip('/'))`, producing `/app/workspace/app/workspace/tools/__init__.py` and causing avoidable tool-call retries.

The workspace tool now preserves genuine absolute paths, canonicalizes them, and then enforces the workspace boundary. Relative paths continue to resolve from `/app/workspace`. Attempts to access paths outside the workspace are rejected with an explicit error.

This is primarily a latency fix at the agent-loop level: eliminating an erroneous tool call also eliminates the subsequent model-generation round required to recover from that error.

### Trace-derived performance snapshot

From the supplied trace:

- Prompt processing: 2,078–3,303 tokens over 8.6–23.3 s, approximately 124–243 prompt tokens/s.
- Generation: 31–48 tokens over 3.1–5.2 s, approximately 9.0–9.3 output tokens/s.
- Three repeated file-path failures therefore cost multiple additional inference rounds, making correctness fixes in tool execution materially more valuable than micro-optimizing Python file I/O.

Ollama exposes `prompt_eval_cached_count` alongside prompt and generation timings on the final streamed response; the harness performance display already surfaces it when the server provides the field.
