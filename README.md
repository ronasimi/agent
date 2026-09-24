<p align="center">
  <img src="webui/static/assets/agent-logo.png" alt="Al Agent logo" width="300">
</p>

# Al Agent

Al Agent is a local-first AI assistant for Ollama. It combines a browser chat interface with tools, persistent conversations, memory, reminders, research, browser automation, and background jobs while keeping the main model small and responsive.

## Features

- Browser-based chat UI with streaming responses
- Optional live **Thinking** stream above the composer
- Persistent conversations and timestamp-aware historical recall
- Four-tier context and memory system optimized for small local models
- Tool calling, reusable recipes, grounding, and bounded recovery
- Cached startup model-conformance detection for tool calling, thinking, and streaming
- Browser/UI automation with state tracking and verification
- Background jobs, reminders, research, and report generation
- Workspace file browser with uploads, previews, downloads, and inline media
- Optional read-only Gmail, Calendar, and Google Drive connections
- One-click **Generate Bug Report** for LLM-assisted troubleshooting
- Local SQLite/WAL storage and encrypted credential storage

## Prerequisites

You need:

- Linux
- Docker with the Docker Compose plugin
- Ollama running on the host
- Git for normal source-control workflows
- Enough RAM/VRAM for the models configured in `config/config.yaml`

The default setup expects Ollama at `http://127.0.0.1:11434`.

## Quick start

### 1. Clone the repository

```bash
git clone <repository-url>
cd agent
```

### 2. Create the configured Ollama model aliases

```bash
./scripts/create_ollama_aliases.sh
```

You can change model names and context sizes later in `config/config.yaml`.

### 3. Start Al Agent

```bash
docker compose up -d --build
```

Open:

```text
http://127.0.0.1:8080
```

### 4. Stop Al Agent

```bash
docker compose down
```

## First run

The Web UI guides you through basic profile setup. Google connections are optional and can be configured later from the wrench menu.

The main interface provides:

- **New Chat** and saved conversations in the left sidebar
- **Jobs** and **Reminders** in the sidebar
- a folder icon in the top bar for workspace files
- a **Think** toggle in the composer
- a wrench menu for **Profile Setup**, **Connections**, **Generate Bug Report**, and **UI Benchmarks**

## Thinking mode

Thinking is off by default for faster everyday responses. When enabled, model reasoning is streamed live in a separate bubble above the chat input and the final answer remains in the conversation normally.

## Files and persistent data

The repository uses two runtime data directories:

```text
workspace/   user files, generated artifacts, downloads, research output
memory/      SQLite state, conversations, context, profile data, traces
```

Both are ignored by Git.

The folder icon in the Web UI opens the workspace drawer.

## Generate a bug report

For troubleshooting:

1. Reproduce the problem.
2. Open the wrench menu.
3. Select **Generate Bug Report**.
4. Look in the repository root for a file named like:

```text
al-agent-bug-report-20260924-012530Z.md
```

The report is designed to be supplied to an LLM together with the matching codebase. It includes bounded runtime state, recent timestamped conversation history, active working state, effective system prompts, recent model-call traces, failures, tool health, storage information, Git state/diff information, runtime versions, and a compact repository map.

Known credential/token fields are redacted, but the report can contain conversation text and system prompts. Review it before sharing outside your trusted environment.

Bug-report files are ignored by Git.

## Configuration

The main configuration file is:

```text
config/config.yaml
```

Common settings include:

- Ollama model roles and context sizes
- thinking and generation limits
- memory/context compaction
- tools and recipes
- browser automation budgets
- background jobs and reminders
- research/report settings

Deployment-specific settings can also be supplied through environment variables in `docker-compose.yml` and `ollama.env.example`.

## Optional Google connections

Open **Connections** from the wrench menu to configure read-only Google access. The harness supports Gmail search/read, Calendar read access, and Google Drive metadata/listing. It does not send email or modify Google data through these read-only integrations.

## Troubleshooting

View container logs:

```bash
docker compose logs -f webui worker
```

Rebuild after source changes:

```bash
docker compose up -d --build
```

For harder problems, generate a bug report from the wrench menu and provide it with the repository to the diagnosing LLM.

Developer benchmarks and troubleshooting utilities live under `diagnostics/`. Setup and operational helper scripts remain under `scripts/`.

## Documentation

Engineering documentation is under [`docs/`](docs/README.md).

Useful starting points:

- [Architecture](docs/ARCHITECTURE.md)
- [Current state](docs/CURRENT_STATE.md)
- [Context and memory tiers](docs/CONTEXT_TIERS_2026-09-23.md)
- [Model capability/conformance layer](docs/MODEL_CAPABILITY_CONFORMANCE.md)
- [Bug reports](docs/BUG_REPORTS.md)

## Security

- The Web UI binds to `127.0.0.1` by default.
- The UI does not provide multi-user authentication.
- Credentials are kept outside model context in the local credential store.
- Google integrations are read-only.
- Bug reports redact known secret fields but may contain chat content and prompts.

