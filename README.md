# Al Agent

A local assistant for Ollama with autonomous tool selection and one all-purpose model. The default runtime alias is `agent-main:4b`.

## Features

- Browser chat, saved conversations, files, and generated artifacts.
- Model-selected tools, arguments, execution order, and final answers.
- Qwen3.8 XML-style tool calls using the model template’s `<tool_call>/<function>/<parameter>` grammar.
- Resident 0.5B System-1 tool router with compact fallback discovery, bounded native schemas, and persistent outcome calibration.
- Conversation-scoped memory and tool observations.
- Research, reminders, browser automation, and durable background jobs.
- One model for chat, tool decisions, research, maintenance, and custom-tool generation.
- Strict argument validation, finite execution budgets, and cancellation.
- Optional read-only Google integrations and local encrypted credential storage.

## Prerequisites

- Linux and Docker with the Compose plugin.
- Ollama running on the host, normally at http://127.0.0.1:11434.
- Enough RAM or VRAM for your main model at 32,768 tokens and the 0.5B router at 8,192 tokens to remain loaded together.
- Python 3.11 or newer for development outside Docker.

## Quick start

Extract the repository and open a terminal in its root directory.

```bash
./scripts/create_ollama_aliases.sh
docker compose up -d --build
```

The runtime expects the existing `agent-main:4b` alias. The alias script reuses it when present; if it is missing, set `AGENT_MODEL_SOURCE` explicitly rather than having the harness guess an upstream 4B tag. The script also pulls the configurable 0.5B router model (`qwen2.5:0.5b` by default).

Open http://127.0.0.1:8080. Stop the services with:

```bash
docker compose down
```

The supplied ollama.env.example contains settings for the host Ollama service, including two loaded models and one parallel inference request per model. Apply them to that service and restart it as appropriate for your installation; Compose does not configure the host service.

## How tool selection works

Before main-model inference, a tiny local router (`qwen2.5:0.5b` by default) receives a stable compact capability index, followed by up to eight candidate IDs and the current request. Startup warmup evaluates that same prefix so Ollama can reuse its cached tokens across different requests. The router emits a three-digit ID and confidence grade; successful task outcomes calibrate that confidence in durable SQLite state. The router never receives conversation history or full JSON schemas, and its turn-local request/candidate state is reset after every turn.

Routing feedback is stored in the durable harness SQLite database. Successful executions and completed tasks increase a bounded exponential moving average, genuine tool/task failures can reduce it, and transport/infrastructure failures are recorded without penalizing the tool. Context-specific and global scores decay toward neutral over time and are loaded again after restart.

`tool_search` returns only names, short descriptions, and compact relevance/confidence metadata. When the 0.5B router is confident, exactly one task schema is activated; otherwise discovery remains with `tool_search`/`load_tools` and no schema is guessed. Complete schemas are never duplicated into tool results; they are supplied only through Ollama’s native `tools` field. The Qwen3.8 Jinja template renders those schemas into its `<tools>` block, the model emits XML-style `<tool_call>` blocks, and observations return in exact `<tool_response>` user-message envelopes.

Tool progress appears during execution. Tool-call envelopes are buffered until complete and validated before execution. Thinking is opt-in: the default request sends `think: false`, which the supplied template converts into an empty `<think>

</think>

` generation prefix; the Web UI Think checkbox sends `think: true`.

## Configuration

Edit config/config.yaml. The primary settings are:

| Setting | Default | Purpose |
|---|---|---|
| agent.model | agent-main:4b | Main conversation, tools, research, and background inference |
| agent.router.model | qwen2.5:0.5b | Stateless System-1 tool selection only |
| agent.router.options.num_ctx | 8192 | Room for the stable compact capability index |
| agent.router.residency_check_seconds | 30 | Idle cache maintenance check interval |
| agent.tool_protocol | qwen_xml | Use the supplied Qwen3.8 XML tool-call template |
| agent.main_options.num_ctx | 32768 | Shared context size |
| agent.main_options.num_predict | 2048 | Per-response output limit |
| agent.max_model_calls_per_turn | 24 | Model-call budget |
| agent.max_tool_calls_per_turn | 48 | Tool-call budget, including discovery |
| agent.max_active_tools | 16 | Loaded task tools, plus discovery controls |
| agent.max_tool_schema_chars | 20000 | Loaded task-schema allowance |
| agent.turn_hard_timeout_seconds | 600 | Cooperative turn deadline |

AGENT_MODEL, AGENT_ROUTER_MODEL, and OLLAMA_HOST override the corresponding configuration values. Legacy role settings are normalized to the main model and options; different legacy role overrides are ignored with a warning.

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

The router-prefix update passed 100 focused and integration tests covering cache lifecycle, routing fallback, metrics, tool execution, model protocols, and bug reports. The simulator covers direct answering, real calculation dispatch, and model-directed argument repair with a scripted model. Real SDK tests exercise wire protocols through mocked HTTP transport. Live Ollama latency, model quality, and Docker deployment require validation on your host.

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
- [Router caching and latency measurement](docs/ROUTER_PREFIX_CACHE.md)
- [Documentation index](docs/README.md)

The UI binds to localhost by default and has no multi-user authentication. Preserve the existing credential, path-validation, and lifecycle controls when extending tools.

### System-1 router residency

Keep both the main model and router resident in Ollama (`OLLAMA_MAX_LOADED_MODELS=2`, `OLLAMA_KEEP_ALIVE=-1`). The router defaults to an 8192-token context, deterministic decoding, and four output tokens. The Web UI primes the real capability index at startup, then checks residency while idle and reprimes after detected reloads or catalog changes. Healthy resident models receive no periodic inference pings. Persistent routing calibration remains in `/app/memory/knowledge.db` across process/container restarts.

Measure actual routing latency and cached tokens on your machine:

```bash
docker compose exec webui python diagnostics/benchmarks/benchmark_router.py --runs 20
```

See [router caching](docs/ROUTER_PREFIX_CACHE.md) for metrics, cold-load measurement, and deployment requirements.
