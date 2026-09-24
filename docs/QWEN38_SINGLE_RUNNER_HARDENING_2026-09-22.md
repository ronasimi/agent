# Qwen3.8 single-runner hardening

> **Historical status:** This is a point-in-time engineering/review record and is intentionally preserved as written. Model roles, tool counts, test totals, limits, and runtime behavior may have changed since this revision. For the current harness use `CURRENT_STATE.md`, `README.md`, and `ARCHITECTURE.md`.


- Main, fast-validator, compaction, and configured vision role use the literal `agent-main:2b` identity.
- All same-model runtime paths use `num_ctx: 16384`; no 4K interactive/validator override remains.
- Main/fast/vision keep-alive paths retain the shared runner indefinitely where those roles are active.
- Tool-selection turns default to `think=false`, `num_predict<=384`, temperature 0.2.
- Validator turns use 16K context, temperature 0.1, `num_predict=128`.
- Interactive turns have 120s soft / 180s hard deadlines, six model calls, two validator calls, and two attempts per requirement.
- Explicit compound requirements use a narrow requirement-led schema and deterministic read-only execution where supported.
- Structured weather recovery geocodes once, owns the coordinates in the harness, and requests one forecast day for current-weather tasks.
- Deterministic non-retryable file/URL failures close their requirements immediately; exhausted requirements are frozen.
- Satisfied/blocked requirements are excluded from repeat work and deterministic partial finalization is used when the turn budget expires.
- Working-state evidence referenced by active requirements is retained preferentially to reduce evidence loss during repeated calls.
- `agent-main:2b` is text-only in this configuration, so `vision.supports_images` is false even though the role name aliases main.
