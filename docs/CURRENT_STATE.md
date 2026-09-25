# Current state

The foreground harness now uses an autonomous action/result loop and one all-purpose model.

- Default alias: agent-main:4b.
- Default source: hf.co/empero-ai/Qwen3.8-4B-Distill-GGUF:Q4_K_M.
- Default protocol: schema-constrained JSON; native calls are an explicit configuration option.
- Context: 16,384 tokens; default output allowance: 2,048 tokens.
- Model-selected discovery, task tools, arguments, order, and finalization.
- No prompt keyword router, forced recipe/fact fallback, model escalation, or vision sidecar.
- Background consumers and legacy role aliases use the same model and context.
- Text-only image-understanding limitation remains explicit.
- Existing SQLite data, tool implementations, UI, jobs, recipes, and integration controls are preserved.

## Validation

587 offline tests passed; four skipped: three require a live configured Ollama server, and one requires UID remapping unavailable in the test environment. Two dependency deprecation warnings remain.

Python compilation, production correctness lint, focused core lint, JavaScript syntax, shell syntax, and the three-scenario offline simulator passed. The actual Ollama Python SDK was tested with mocked HTTP transport for both protocols.

No live Ollama/model-quality evaluation or Docker image build was performed. Run diagnostics/validate_ollama.py on the target host before relying on the deployment.

See AUTONOMOUS_REFACTOR.md for specific fixes and migration notes. Historical design documents are not descriptions of the current router.
