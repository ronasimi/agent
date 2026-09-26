# Al Agent

A local assistant for Ollama with autonomous tool selection and one all-purpose model. The default runtime alias is `agent-main:4b`.

## Features

- Browser chat, saved conversations, files, and generated artifacts.
- Model-selected tools, arguments, execution order, and final answers.
- Qwen3.8 XML-style tool calls using the model template’s `<tool_call>/<function>/<parameter>` grammar.
- Deterministic catalog prefilter with bounded native schemas; the resident 4B model makes all semantic tool decisions and executes the task.
- Conversation-scoped memory, durable tool observations, and three-tier State Tape prompt compaction.
- Research, reminders, browser automation, and durable background jobs.
- Durable sparse-tape universal computation for practically Turing-complete deterministic workloads without removing foreground safety budgets.
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


## Prompt compaction

Each active turn receives only the schemas selected for that turn. While the turn is running, raw tool calls/results remain available for dependent steps. At turn completion, those schemas and protocol records are removed from model-facing history and replaced by a compact deterministic State Tape entry. The last three user/final-assistant turns remain for conversational continuity; older resolved tape entries roll into a bounded summary. Unfinished work survives only as compact unresolved state and its schemas are re-routed when work resumes.

This keeps the static system-policy prefix stable, bounds historical prompt growth, and prevents multi-kilobyte JSON/XML observations from being repeatedly prefetched on unrelated later requests. Full observations remain in SQLite and can be retrieved explicitly when needed.

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
| agent.model_transport.first_byte_timeout_seconds | 120 | Prefill/first stream-chunk deadline |
| agent.model_transport.stream_idle_timeout_seconds | 60 | Maximum idle time after streaming begins |
| agent.context.recent_conversation_turns | 3 | Raw conversational surface retained in prompts |
| agent.context.state_tape_entries | 6 | Recent compact completed-turn entries |
| agent.context.rolling_summary_chars | 3200 | Tier-3 deep-history summary bound |

AGENT_MODEL and OLLAMA_HOST override the corresponding configuration values. Legacy role settings are normalized to the main model and options; legacy router settings are ignored with a warning.

The configured distilled GGUF is text-only. Image display, downloads, and document tools remain available; pixel understanding is unavailable. The harness never loads a vision model.

## Files and existing installations

Runtime data lives in workspace/ and memory/. Credentials use the existing Docker volume. These directories and credentials are excluded from this source archive.

For an existing deployment, back up those data directories, retain your integration settings, and merge the new single-model agent configuration before rebuilding. Existing chat rows, observations, jobs, and recipes remain usable. Completed raw tool transcripts remain stored for audit/search but are not replayed into model prompts. Older resolved State Tape entries advance the prompt-facing compaction watermark; all original chat rows remain searchable/exportable through the existing history tools.

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
.venv/bin/python diagnostics/check_turing_completeness.py
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
