# Al Agent

A local assistant for Ollama with autonomous tool selection and one all-purpose model. The default runtime alias is `agent-main:4b`.

## Features

- Browser chat, saved conversations, files, and generated artifacts.
- Model-selected tools, arguments, execution order, and final answers.
- Qwen3.8 XML-style tool calls using the model template’s `<tool_call>/<function>/<parameter>` grammar.
- Deterministic catalog prefilter with bounded native schemas; the resident 4B model makes all semantic tool decisions and executes the task.
- Conversation-scoped memory and tool observations.
- Research, reminders, browser automation, and durable background jobs.
- One model for chat, tool decisions, research, maintenance, and custom-tool generation.
- Strict argument validation, finite execution budgets, and cancellation.
- Optional read-only Google integrations and local encrypted credential storage.

## Prerequisites

- Linux and Docker with the Compose plugin.
- Ollama running on the host, normally at http://127.0.0.1:11434.
- Enough RAM or VRAM for your main model at a 32,768-token context.
- Python 3.11 or newer for development outside Docker.

## Quick start

Extract the repository and open a terminal in its root directory.

```bash
./scripts/create_ollama_aliases.sh
docker compose up -d --build
```

The runtime expects the existing `agent-main:4b` alias. The alias script reuses it when present; if it is missing, set `AGENT_MODEL_SOURCE` explicitly rather than having the harness guess an upstream 4B tag. No secondary routing model is required.

Open http://127.0.0.1:8080. Stop the services with:

```bash
docker compose down
```

The supplied ollama.env.example contains settings for the host Ollama service, including one resident model and one parallel inference request. Apply them to that service and restart it as appropriate for your installation; Compose does not configure the host service.

## How tool selection works

Tool routing uses no second LLM. A deterministic lexical/metadata prefilter scores the registered catalog, applies the existing persistent routing calibration only to genuinely related tools, and activates a bounded set of relevant schemas. High-confidence, well-separated matches activate one schema; ambiguous or multi-intent requests activate up to eight relevant schemas. The already-resident `agent-main:4b` then decides whether to call a tool, which tool to call, its arguments, and what to do with the result.

`tool_search` uses the same deterministic prefilter and never performs Ollama inference. It returns only names, short descriptions, and relevance metadata; complete schemas are supplied only through Ollama's native `tools` field. Each search replaces the prior active task-schema set so prompt size cannot grow monotonically across a turn. `load_tools` remains an explicit escape hatch for exact named capabilities.

Routing feedback remains in the durable harness SQLite database. Successful task outcomes can modestly reorder related candidates, while infrastructure failures are recorded without penalizing tools. Unrelated tools cannot gain relevance from historical feedback alone.

Tool progress appears during execution. Tool-call envelopes are buffered until complete and validated before execution. Thinking is opt-in: the default request sends `think: false`; the Web UI Think checkbox sends `think: true`.

## Configuration

Edit config/config.yaml. The primary settings are:

| Setting | Default | Purpose |
|---|---|---|
| agent.model | agent-main:4b | Main conversation, tools, research, and background inference |
| agent.tool_routing.candidate_limit | 8 | Maximum deterministic schema candidates passed to the main model |
| agent.tool_routing.auto_activate_threshold | 0.80 | Minimum top score for single-schema direct activation |
| agent.tool_routing.auto_activate_margin | 0.20 | Required separation from the runner-up for direct activation |
| agent.tool_routing.min_candidate_score | 0.18 | Minimum relevance for ambiguous candidate-set activation |
| agent.tool_protocol | qwen_xml | Use the supplied Qwen3.8 XML tool-call template |
| agent.main_options.num_ctx | 32768 | Shared context size |
| agent.main_options.num_predict | 2048 | Per-response output limit |
| agent.max_model_calls_per_turn | 24 | Model-call budget |
| agent.max_tool_calls_per_turn | 48 | Tool-call budget, including discovery |
| agent.max_active_tools | 16 | Loaded task tools, plus discovery controls |
| agent.max_tool_schema_chars | 20000 | Loaded task-schema allowance |
| agent.turn_hard_timeout_seconds | 600 | Cooperative turn deadline |

AGENT_MODEL and OLLAMA_HOST override the corresponding configuration values. Legacy role settings are normalized to the main model and options; legacy router settings are ignored with a warning.

The configured distilled GGUF is text-only. Image display, downloads, and document tools remain available; pixel understanding is unavailable. The harness never loads a vision model.

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

The deterministic-routing refactor is covered by focused routing, tool-session, action-loop, model-protocol, and bug-report tests. The simulator covers direct answering, real calculation dispatch, and model-directed argument repair with a scripted model. Real SDK tests exercise wire protocols through mocked HTTP transport. Live Ollama latency, model quality, and Docker deployment require validation on your host.

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

### Routing latency benchmark

The routing benchmark now measures only deterministic catalog retrieval; it performs zero LLM calls:

```bash
docker compose exec webui python diagnostics/benchmarks/benchmark_router.py --runs 1000
```

Use `diagnostics/benchmarks/benchmark_warmup.py` separately to measure the resident main model's cold-load, prefix-prime, and TTFT behavior.
