# Autonomous Local AI Agent Harness

A containerized local AI agent built around Ollama, durable SQLite state, explicit typed tools, a persistent worker, host/network awareness, desktop reminders, and bounded context management.

## Runtime architecture

The interactive CLI (`agent.py`) is deliberately not responsible for long-running work. Commands such as `/research <topic>` create a durable job in `memory/knowledge.db`. `worker.py` claims jobs, checkpoints every meaningful phase, renews heartbeats, retries transient failures, and resumes stale jobs after a crash or container restart.

Research jobs use an iterative state machine:

`plan → search/fetch → evaluate coverage → targeted gap search → synthesize → persist report`

Every source is scoped to its research job ID, so concurrent or historical research runs cannot overwrite one another.

## Features

- Local Ollama model integration with native structured tool calls.
- 16K main context and 4K fast-model context defaults for an 8 GiB VRAM budget.
- Rolling conversation summaries and query-specific memory retrieval instead of injecting the complete memory table into every prompt.
- Durable research jobs and checkpoints.
- Host snapshots for CPU, RAM, disk, temperatures, optional GPU telemetry, and loaded Ollama models.
- Host-network awareness for interfaces, routes, DNS, listening sockets, and bounded reachability checks.
- Optional mDNS discovery and bounded local network mapping.
- Persistent desktop reminders using systemd user timers and `notify-send`.
- Explicit typed tool registry. Missing tool arguments are errors; the harness never extracts shell commands or URLs from normal model prose.
- Safe outbound HTTP handling with private-network/loopback SSRF protection, bounded redirects, response-size limits, and content-type checks.
- Workspace path validation that resolves symlinks before allowing reads/writes.
- Optional custom tools loaded only when decorated with `@agent_tool`.

## 8 GiB VRAM deployment

The agent container does not configure the Ollama server. The Ollama server/container is expected to enforce the model residency policy. For a system with an approximately 8 GiB VRAM budget, use one loaded model at a time and keep request parallelism at one.

Recommended Ollama server environment:

```text
OLLAMA_MAX_LOADED_MODELS=1
OLLAMA_NUM_PARALLEL=1
OLLAMA_FLASH_ATTENTION=1
```

The exact way those variables are applied depends on how the existing Ollama container is created. They must be set on the **Ollama server/container**, not on `agent` or `agent-worker`.

The default agent configuration uses:

```yaml
agent:
  model: qwen3.5:4b
  fast_model: qwen2.5-coder:1.5b
  context:
    num_ctx: 16384
  main_options:
    num_ctx: 16384
  fast_options:
    num_ctx: 4096
```

The fast model is unloaded after short planning/evaluation work where supported by the installed Ollama Python client. The main research synthesis call also uses `keep_alive=0`, allowing the interactive session to reclaim VRAM afterward.

## Docker deployment

The compose file runs two containers from the same image:

- `agent` — interactive CLI.
- `agent-worker` — persistent background worker.

Both use the host network namespace because network awareness is a stated requirement. `pid: host` is retained for host process/telemetry visibility. The host filesystem is mounted read-only at `/host`; the host journal/log directory is mounted read-only at `/host_log`.

The container runs as UID/GID `1000:1000` and reaches the host user's session bus through `/run/user/1000/bus`. The host project directory and its `workspace/`, `memory/`, and `~/.config/systemd/user` paths must be writable/readable by that user.

Before starting, confirm the configured models exist on the Ollama server (`ollama ls`). Pull/rename the models in `config/config.yaml` as needed.

Start the runtime with:

```bash
docker compose build
docker compose up -d worker
docker compose run --rm agent
```

Or run the interactive frontend as the persistent `agent` service:

```bash
docker compose up -d agent worker
```

Then attach to the CLI container as appropriate for your workflow.

## Reminders

The model should call `schedule_reminder()` rather than writing a systemd unit itself. The tool persists the reminder in SQLite, writes a user timer/service pair under `/home/agent/.config/systemd/user` (a bind mount of the host user directory), runs `systemctl --user daemon-reload`, and activates the timer.

Examples of supported reminder scheduling inputs:

```text
schedule_reminder(
  title="Call Bob",
  message="Call Bob about the network installation.",
  when="2026-09-18T09:00:00-04:00",
  repeat="once"
)
```

Or a relative delay:

```text
schedule_reminder(
  title="Check backup",
  message="Check the overnight backup result.",
  delay_seconds=3600
)
```

For reminders to run after login, the host user session must be available to `systemd --user`. If a headless host needs timers to run without an active login, enable the host user's systemd lingering separately.

## CLI commands

```text
/research <topic>       queue durable research
/jobs                   list durable background jobs
/job <id>               inspect one job/checkpoint
/cancel-job <id>       cancel a job
/reminders              list active reminder records
/think [on/off]         toggle model thinking
/tools                  show active typed tools
/reload                 reload builtins/custom tools
/forget                 clear chat history and rolling summary
exit                    exit the interactive CLI
```

## Custom tools

Custom tools live under `workspace/custom_tools/` and must explicitly opt in:

```python
from tools.tool_registry import agent_tool

@agent_tool(name="hello", description="Say hello to a person.", readonly=True)
def hello(name: str = "world") -> str:
    return f"Hello, {name}!"
```

After saving the file, use `/reload`.

## Existing database compatibility

The runtime creates new job, checkpoint, reminder, and monitor tables without deleting the existing knowledge/chat tables. The research buffer migration keeps legacy rows under the `legacy` run ID; new research runs are isolated by job ID.
