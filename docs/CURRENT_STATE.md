# Current state

The foreground harness uses one resident `agent-main:4b` model for conversation, tool decisions, execution planning, research, and background inference. The former 0.5B System-1 routing model has been removed.

- Main alias: `agent-main:4b`.
- No secondary routing model or routing-model Ollama client.
- Default protocol: Qwen XML-style tool calls (`qwen_xml`) with complete active schemas supplied only through Ollama's native `tools` field.
- Main context: 32,768 tokens; default output allowance: 2,048 tokens.
- Deterministic catalog prefilter: up to eight relevant candidate schemas by default.
- High-confidence matches activate one schema; ambiguous/multi-intent requests activate a bounded relevant set.
- `tool_search` performs no model inference and replaces, rather than accumulates, active task schemas.
- Persistent routing calibration remains in `/app/memory/knowledge.db` and can reorder only already-related candidates.
- Startup warmup primes only the main model; no router residency maintenance runs.
- Background consumers and legacy main-role aliases continue to use `agent-main:4b`.
- Existing SQLite data, tool implementations, UI, jobs, recipes, and integration controls are preserved.

## Validation

Run the offline test suite, deterministic routing benchmark, simulator, and the live Ollama conformance test on the target host. The bug-report runtime snapshot now reports `deterministic_catalog_prefilter`, `separate_model: false`, and `llm_calls: 0` for routing.
