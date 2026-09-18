# Autonomous Local AI Agent Harness — Performance Update

This bundle contains the performance-focused changes for the local Ollama agent harness. The update targets the two latency-sensitive parts of an interactive agent loop:

1. **Prefill:** reduce the amount of prompt and tool-schema data that must be processed before generation starts.
2. **Inference responsiveness:** keep background research from competing with an active interactive turn and avoid unnecessary model eviction/reload cycles.

The harness remains a local, containerized Ollama agent with durable SQLite state, explicit typed tools, a persistent research worker, host/network awareness, reminders, and bounded conversation context.

## What changed

### Lower prefill cost

The interactive loop previously rebuilt prompt-related state on every tool-call iteration and exposed the complete native tool inventory to every model request. The updated loop now:

- Builds the turn-level system prompt once.
- Resolves the conversation summary once per turn and reuses it through the tool loop.
- Selects a deterministic subset of native Ollama tool schemas for each user request.
- Keeps a small always-available set of core tools for safe fallback behavior.
- Replaces the full model-facing tool inventory with a compact tool policy. Native Ollama schemas remain authoritative for actual arguments and tool calling.
- Explicitly reserves prompt budget for the selected native tool schemas when building the active context.
- Continues truncating oversized tool output before it can consume the conversation budget.

The tool selector is deterministic and lexical rather than embedding-based, so selection itself adds negligible CPU work and does not require another model call.

### Faster normal turns

Normal interactive turns now default to `think: false`. This avoids paying reasoning-token latency on routine requests while preserving the existing `/think on` control for reasoning-heavy work.

### Avoid unnecessary model eviction

Context compaction defaults to the main model instead of the fast model. This matters on an Ollama deployment configured for a single resident model: using a second model for compaction can evict the main model and force another load before the user's next request.

Set `compaction_model` to the fast model only when the Ollama server has enough memory/VRAM to keep both models resident without causing reload pressure.

### Protect the interactive path

The CLI records an interactive-inference activity lease in SQLite. The worker checks that lease before starting resource-intensive research operations and defers work while the interactive path is active or was used recently.

The worker also runs at a lower OS scheduling priority (`nice(5)`) so background processing yields more readily to the interactive process.

### Faster retry behavior

Transient Ollama errors on the interactive path use a short retry delay rather than a one-second stall, reducing perceived latency after a recoverable connection hiccup.

### Prefill/inference telemetry

The interactive CLI now prints Ollama timing counters when available, including:

- `prompt_eval_count`
- `prompt_eval_cached_count`
- `prompt_eval_duration`
- `eval_count`
- `eval_duration`

These values make it possible to distinguish prompt/prefill cost from generated-token cost and to verify whether repeated tool-loop requests are benefiting from prompt caching on the installed Ollama version.

## Performance configuration

The optimized defaults are in `config/config.yaml`:

```yaml
agent:
  model: "qwen3.5:4b"
  fast_model: "qwen2.5-coder:1.5b"
  thinking_default: false
  show_perf_stats: true
  max_tools_per_turn: 20

  compaction_model: ""
  compaction_options:
    num_ctx: 4096
    temperature: 0.0
    top_p: 0.9
    top_k: 20
    num_predict: 512

  context:
    num_ctx: 16384
    reserve_tokens: 2048
    recent_messages: 12
    summary_keep_messages: 8
    compact_at_tokens: 9000
    max_tool_output_chars: 10000

worker:
  poll_interval_seconds: 3
  heartbeat_seconds: 15
  stale_job_seconds: 180
  interactive_cooldown_seconds: 10
  min_available_memory_mb: 900
  max_agent_vram_mb: 7200
```

### `max_tools_per_turn`

This is the primary prefill control. A value of `20` is a conservative default for a tool-rich harness. Lower values can reduce schema prefill further but increase the chance that a specialized tool is omitted from a turn. The selector always preserves the core fallback tools.

### `reserve_tokens`

This reserves context budget for the native tool schemas. Without the reservation, a long conversation can consume nearly the entire `num_ctx` before schemas are accounted for.

### `compaction_model`

An empty value means **reuse the main model**. This is the preferred setting for a one-model-at-a-time Ollama deployment.

For a system capable of keeping both models loaded, set for example:

```yaml
compaction_model: "qwen2.5-coder:1.5b"
```

## Recommended Ollama server settings

For a roughly 8 GiB VRAM budget, keep Ollama's residency and request parallelism conservative:

```text
OLLAMA_MAX_LOADED_MODELS=1
OLLAMA_NUM_PARALLEL=1
OLLAMA_FLASH_ATTENTION=1
```

These variables belong on the **Ollama server/container**, not on `agent` or `agent-worker`.

The harness intentionally does not try to manage Ollama's server-wide model residency policy.

## Runtime architecture

The runtime has two processes/containers:

- `agent` — interactive CLI and primary inference path.
- `agent-worker` — durable background research and monitoring.

Long-running research is represented as a durable job in SQLite rather than being kept inside the foreground conversation. The worker claims jobs, checkpoints progress, renews heartbeats, retries transient failures, and can resume stale work after a crash or restart.

Research follows the state machine:

`plan → search/fetch → evaluate coverage → targeted gap search → synthesize → persist report`

Interactive inference is treated as the latency-sensitive path. The worker therefore yields when the foreground interaction lease is active or recently active.

## Existing features retained

- Local Ollama integration with native structured tool calls.
- Bounded rolling conversation summaries and query-specific memory retrieval.
- Durable research jobs and checkpoints.
- Host snapshots for CPU, RAM, disk, temperatures, optional GPU telemetry, and loaded Ollama models.
- Host-network awareness for interfaces, routes, DNS, listening sockets, and bounded reachability checks.
- Optional mDNS discovery and bounded network mapping.
- Persistent desktop reminders using systemd user timers and `notify-send`.
- Explicit typed tool registry; malformed/missing arguments are rejected instead of extracting shell commands from model prose.
- SSRF-aware outbound HTTP handling with private/loopback protection, bounded redirects, response-size limits, and content-type checks.
- Workspace path validation with symlink resolution.
- Opt-in custom tools using `@agent_tool`.

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

After saving the file, run `/reload`.

## Deployment

The original project uses Docker Compose with separate `agent` and `agent-worker` services. Both containers use the host network namespace for network awareness, and the project mounts the host paths required for telemetry, workspace data, and user-level systemd reminders.

Before starting the updated runtime, verify that the configured Ollama models exist:

```bash
ollama ls
```

Typical startup:

```bash
docker compose build
docker compose up -d worker
docker compose run --rm agent
```

Or run both services persistently:

```bash
docker compose up -d agent worker
```

## Updating an existing checkout

This performance bundle is an **update bundle**, not a replacement for the original repository. Copy the updated files into the existing project while retaining the original Docker, dependency, and auxiliary tool files.

The files changed by this update are:

```text
agent.py
worker.py
tools/__init__.py
tools/context.py
config/config.yaml
tests/test_context.py
tests/test_registry.py
```

`agent.patch` contains the corresponding unified patch, and `PERFORMANCE_REVIEW.md` records the implementation and validation details.

## Validation

The updated test set passes:

```text
10 tests passed
```

The tests cover context/schema behavior and the registry changes. End-to-end Ollama latency was not benchmarked in this build environment because it does not contain the target Ollama/GPU runtime.

Use the CLI performance counters on the target machine to measure real prefill and generation latency after deployment. Compare `prompt_eval_duration` and `prompt_eval_cached_count` before and after changes, using the same model, context size, and workload.

## Tuning guidance

For an interactive laptop with limited VRAM, prioritize these settings in order:

1. Keep `OLLAMA_NUM_PARALLEL=1`.
2. Avoid keeping a second model resident unless there is enough VRAM for both.
3. Keep `thinking_default: false` and enable thinking only for tasks that benefit from it.
4. Keep `max_tools_per_turn` bounded; lower it if tool-schema prefill dominates your traces.
5. Keep `compact_at_tokens` comfortably below `num_ctx` so compaction happens before the model reaches a hard context limit.
6. Watch the printed Ollama counters rather than optimizing solely from subjective typing/token speed.

The target is not maximum throughput. The target is **low interactive latency with predictable background progress**.
