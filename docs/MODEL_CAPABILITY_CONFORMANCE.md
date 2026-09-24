# Model Capability / Conformance Layer

Al Agent performs a small, safe conformance check when a configured Ollama model is first seen. The purpose is to adapt request construction to the actual model/chat template instead of assuming every model supports optional Ollama fields in the same way.

## What is checked

For the interactive model the probe records:

- plain chat completion
- whether `stream=True` produces a streaming transport
- whether the Boolean `think` parameter is accepted
- whether streamed reasoning appears in `message.thinking`
- whether tool schemas are accepted
- the observed tool-call mode: native `message.tool_calls`, Qwen-style XML, accepted-but-unverified, unsupported, or unknown
- whether an assistant tool invocation can be followed by a tool-result message and another model response

All probes use synthetic text and one harmless fake tool named `capability_probe_echo`. No real tool implementation is executed and no user conversation data is included.

## Cache and invalidation

Profiles are written to:

```text
/app/memory/model_capabilities.json
```

The cache key includes the probe schema version and the Ollama model digest when available. Rebuilding/replacing a model alias therefore causes a new probe automatically. If the digest is unavailable, the chat-template hash is used; the model name is the final fallback.

Transient failures that prevent even plain chat from completing are not persisted, so an unavailable Ollama server cannot permanently poison the model profile.

## Startup behavior

The WebUI remains non-blocking. Normal model warm-up happens on a daemon thread. A cached profile is loaded using metadata only. A cold/new model waits for the configured idle window before conformance generation begins, and recent foreground activity defers the probe.

The distinct fast model is probed only from its existing idle/prewarm path, after it is already resident. This avoids an extra model swap on constrained hardware.

Default configuration:

```yaml
agent:
  model_capabilities:
    enabled: true
    probe_main_on_startup: true
    probe_fast_when_warmed: true
    cache_path: "/app/memory/model_capabilities.json"
    idle_delay_seconds: 3
    force_probe: false
```

## Runtime adaptation

The profile is conservative:

- unknown/inconclusive results preserve historical behavior
- an explicitly unsupported `think` parameter is omitted
- empty `tools=[]` is never serialized; direct chat simply omits the optional field
- an explicitly rejected tool-schema or tool-result protocol blocks model-directed tool use before side effects can occur
- a backend that returns a completed response despite `stream=True` is normalized and remains functional, though live token/reasoning streaming is unavailable
- Qwen-style textual XML tool calls remain supported by the existing parser

Deterministic recipes and other harness-owned fast paths are independent of model-directed tool calling and can continue to work even when a model does not support tool schemas.

## Diagnostics

Active profiles are included in generated bug reports under the runtime snapshot. The exact model request remains available in `memory/model_calls.jsonl`.

To force a live conformance check on the target host:

```bash
RUN_OLLAMA_LIVE_TESTS=1 pytest -q tests/test_ollama_conformance_live.py
```
