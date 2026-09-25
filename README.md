# Al Agent

A local assistant for Ollama with autonomous tool selection and one all-purpose model. The default is your original distilled 4B model, exposed as `agent-main:4b`.

## Features

- Browser chat, saved conversations, files, and generated artifacts.
- Model-selected tools, arguments, execution order, and final answers.
- Schema-constrained JSON actions by default; optional native tool calling.
- Searchable tool catalog with bounded, dynamically loaded schemas.
- Conversation-scoped memory and tool observations.
- Research, reminders, browser automation, and durable background jobs.
- One model for chat, tool decisions, research, maintenance, and custom-tool generation.
- Strict argument validation, finite execution budgets, and cancellation.
- Optional read-only Google integrations and local encrypted credential storage.

## Prerequisites

- Linux and Docker with the Compose plugin.
- Ollama running on the host, normally at http://127.0.0.1:11434.
- Enough RAM or VRAM for the distilled 4B model with a 16,384-token context.
- Python 3.11 or newer for development outside Docker.

## Quick start

Extract the repository and open a terminal in its root directory.

```bash
./scripts/create_ollama_aliases.sh
docker compose up -d --build
```

The alias script pulls `hf.co/empero-ai/Qwen3.8-4B-Distill-GGUF:Q4_K_M` and creates `agent-main:4b`. It creates one alias. To select an existing deployment's model, set AGENT_MODEL_SOURCE and AGENT_MODEL when running the script; keep the same AGENT_MODEL value when starting Compose.

Open http://127.0.0.1:8080. Stop the services with:

```bash
docker compose down
```

The supplied ollama.env.example contains settings for the host Ollama service, including one loaded model and one parallel inference request. Apply them to that service and restart it as appropriate for your installation; Compose does not configure the host service.

## How tool selection works

Each turn presents the model with the tool-name inventory and the schemas for tool_search and load_tools. The model can answer immediately, inspect tool descriptions, load schemas, execute a tool, read its result, and select the next action.

The default JSON protocol constrains the model to one typed tool action or final answer per response. The harness validates the completed response and arguments, executes the selected registered function, and supplies its result to the same model. No user-prompt keywords select a tool, recipe, workflow, or different model.

Tool progress appears during execution. Final answer text is buffered until a complete, valid response arrives, so partial action JSON never appears as a chat answer. Thinking is optional and requires both an explicitly compatible model and supports_thinking enabled in configuration.

## Configuration

Edit config/config.yaml. The primary settings are:

| Setting | Default | Purpose |
|---|---|---|
| agent.model | agent-main:4b | Every inference workload uses this model |
| agent.tool_protocol | json | Set native only for a compatible tool-calling template |
| agent.main_options.num_ctx | 16384 | Shared context size |
| agent.main_options.num_predict | 2048 | Per-response output limit |
| agent.max_model_calls_per_turn | 24 | Model-call budget |
| agent.max_tool_calls_per_turn | 48 | Tool-call budget, including discovery |
| agent.max_active_tools | 16 | Loaded task tools, plus discovery controls |
| agent.max_tool_schema_chars | 20000 | Loaded task-schema allowance |
| agent.turn_hard_timeout_seconds | 600 | Cooperative turn deadline |

AGENT_MODEL and OLLAMA_HOST override the corresponding configuration values. Legacy role settings are normalized to the same model and options; different role overrides are ignored with a warning.

The default distilled model is text-only. Image display, downloads, and document tools remain available; pixel understanding is unavailable. The harness never loads a vision model.

## Files and existing installations

Runtime data lives in workspace/ and memory/. Credentials use the existing Docker volume. These directories and credentials are excluded from this source archive.

For an existing deployment, back up those data directories, retain your integration settings, and merge the new single-model agent configuration before rebuilding. Existing chat rows, observations, jobs, and recipes remain usable. Recent raw chat history is loaded independently of legacy compaction watermarks; older records remain searchable through tools.

Explicit slash commands remain direct interface controls. Recipes, pipelines, and durable jobs still perform their documented operations when chosen by the model or explicitly requested through the interface.

## Validate your deployment

After Ollama and the containers are running:

```bash
docker compose exec webui python diagnostics/validate_ollama.py
```

This read-only smoke test requires the actual model to discover and execute calculate and report 437 for 19 × 23. It exits unsuccessfully if the interaction fails.

For development:

```bash
./scripts/bootstrap_venv.sh
.venv/bin/python -m pytest -q
.venv/bin/python diagnostics/simulate_turns.py
RUN_OLLAMA_LIVE_TESTS=1 .venv/bin/python -m pytest -q tests/test_ollama_conformance_live.py
```

Offline validation: 587 tests passed and four environment-dependent tests skipped. The simulator covers direct answering, real calculation dispatch, and model-directed argument repair with a scripted model. Real SDK tests exercise both wire protocols through mocked HTTP transport. Live Ollama, model quality, and Docker deployment were not validated in the refactoring environment.

## Generate a bug report

Use the wrench menu and choose **Generate Bug Report**. The report includes recent execution state, model-call traces, configuration, and repository details. Known secret fields are redacted, but reports can contain chat content; review them before sharing.

To view service logs:

```bash
docker compose logs -f webui worker
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Refactor findings, migration, and validation](docs/AUTONOMOUS_REFACTOR.md)
- [Current state](docs/CURRENT_STATE.md)
- [Documentation index](docs/README.md)

The UI binds to localhost by default and has no multi-user authentication. Preserve the existing credential, path-validation, and lifecycle controls when extending tools.
