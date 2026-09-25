# Current state

The foreground harness now uses an autonomous action/result loop and one all-purpose model.

- Default alias: agent-main:9b.
- Default source: hf.co/empero-ai/Qwen3.8-9B-Distill-GGUF:Q4_K_M.
- Default protocol: Qwen3.8 XML-style tool calls (`qwen_xml`) with schemas supplied through Ollama’s native `tools` field.
- Context: 32,768 tokens; default output allowance: 2,048 tokens.
- Model-selected discovery, task tools, arguments, order, and finalization.
- No prompt keyword router, forced recipe/fact fallback, model escalation, or vision sidecar.
- Background consumers and legacy role aliases use the same model and context.
- Text-only image-understanding limitation remains explicit.
- Existing SQLite data, tool implementations, UI, jobs, recipes, and integration controls are preserved.

## Validation

The Qwen XML protocol, action-loop, argument-coercion, and warmup unit set passes offline. SDK-backed integration tests require the `ollama` Python package and live-model tests remain opt-in. Run `diagnostics/validate_ollama.py` on the target host before relying on the deployment.

See AUTONOMOUS_REFACTOR.md for specific fixes and migration notes. Historical design documents are not descriptions of the current router.
