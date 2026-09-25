# Current state

The foreground harness uses an autonomous action/result loop with `agent-main:4b` plus a resident stateless 0.5B System-1 router for tool selection.

- Main alias: `agent-main:4b`.
- Router model: `qwen2.5:0.5b` by default, configurable with `AGENT_ROUTER_MODEL`.
- Default protocol: Qwen XML-style tool calls (`qwen_xml`) with complete schemas supplied only through Ollama's native `tools` field.
- Main context: 32,768 tokens; default output allowance: 2,048 tokens.
- Router context: 2,048 tokens, deterministic decoding, four output tokens, up to eight compact candidates.
- Router input is stateless: current request + compact candidate metadata only; no conversation history or full schemas.
- Router prompt/request/candidate state is cleared at turn completion.
- Persistent routing calibration is stored in `/app/memory/knowledge.db` and survives restarts.
- Low-confidence/unavailable router decisions fall back to compact `tool_search`/`load_tools`; no task schema is guessed.
- Background consumers and legacy main-role aliases continue to use `agent-main:4b`.
- Text-only image-understanding limitation remains explicit.
- Existing SQLite data, tool implementations, UI, jobs, recipes, and integration controls are preserved.

## Validation

The Qwen XML protocol, System-1 router, persistent calibration, action-loop, argument-coercion, warmup, and hybrid-streaming focused tests pass offline. SDK-backed integration tests require the `ollama` Python package and live-model tests remain opt-in. Run `diagnostics/validate_ollama.py` on the target host before relying on the deployment.

See `SYSTEM1_ROUTER_2026-09-25.md` for the router contract. Historical design documents are not descriptions of the current routing path.
