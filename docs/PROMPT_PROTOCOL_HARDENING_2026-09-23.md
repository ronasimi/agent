# Prompt Protocol Hardening — 2026-09-23

## Failure

The Qwen3.8 Ollama chat template rejected ordinary model calls with:

`Jinja Exception: System message must be at the beginning.`

The harness had been emitting a stable base system message, the current user/history,
and then a second system-role working-state block. That layout was introduced to
maximize KV-prefix reuse but is invalid for strict templates that allow a system
message only at the beginning.

## Remediation

- `build_active_messages()` now merges base policy plus working state (or rolling
  summary) into exactly one leading system message.
- The evidence digest remains untrusted user-role data after the selected history.
- `ollama_wire_messages()` canonicalizes all system-role blocks again at the
  provider boundary and validates the resulting invariant.
- Prompt/template errors are classified as deterministic even when Ollama returns
  HTTP 500, so preflight transport retry does not replay them.
- The main turn loop stops immediately with a prompt-protocol diagnostic if a
  provider still rejects the canonical request.
- Fallback finalization now uses the same canonical context builder directly,
  preventing working state from being dropped after the single-system change.
- `context.volatile_blocks_last` remains only as a deprecated compatibility key
  and defaults to `false`.

## Invariant

Every provider-bound chat request must satisfy:

1. zero or one system message;
2. when present, the system message is index `0`;
3. no later message has role `system`;
4. removing/merging system blocks must not reorder assistant/tool transactions.

## Regression coverage

Tests cover context assembly, wire canonicalization, tool-transaction preservation,
finalization, deterministic HTTP-500 template errors, and turn-level no-retry
behavior for prompt-protocol failures.
