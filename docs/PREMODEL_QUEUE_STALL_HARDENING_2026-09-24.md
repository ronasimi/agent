# Pre-model queue stall hardening — 2026-09-24

## Incident

A newly submitted 77-step stress-test turn was persisted to chat history but produced no model trace, no tool call, and no working-state scheduler initialization. The diagnostic report showed `status=idle`, `turn_id=0`, and an empty model-call trace list.

## Root cause

`turn_engine.handle_user_turn()` acquired the shared Ollama inference lock before calling `compile_structured_plan()`. The compiler itself first checks for an explicit numbered plan and can parse such suites deterministically without any model inference. Therefore an explicitly numbered request could block indefinitely on the inference mutex before the deterministic parser ran.

The shared lock helper also had no deadline, so a long-held or leaked inference slot could leave the WebUI turn pending forever unless the user explicitly cancelled it.

## Fix

- Structured-plan compilation now acquires the inference slot lazily through a `before_model_call` callback.
- Explicit numbered suites return before the callback is invoked and therefore do not touch the Ollama queue during plan setup.
- Fast-model compiler calls still acquire the same shared lock immediately before the actual Ollama request.
- Foreground inference-lock waits now have a configurable finite deadline (`model_transport.queue_timeout_seconds`, default 90 seconds).
- Queue wait/progress/timeout events expose the blocked phase to the WebUI.
- Diagnostic reports now capture the current interaction `active` / `waiting` monitor state and the effective inference queue timeout.

## Invariant

A deterministic preflight operation must never acquire a model/inference resource merely because a later fallback might need one. Resource acquisition belongs at the exact side-effect boundary.
