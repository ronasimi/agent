# Autonomous Local AI Agent Harness

Local Ollama agent with an interactive CLI, typed tools, durable SQLite state,
background research, rolling context summaries, host/network awareness, and
systemd-based reminders.

This version is optimized for low time-to-first-token, high prompt-cache reuse,
bounded context growth, and foreground responsiveness.

## Performance architecture

### Stable prompt prefixes

- The system prompt is static. Query-specific memories are inserted immediately
  before the current user request instead of modifying the system message.
- The active historical prefix is built once at the start of a user turn.
- Each tool iteration appends assistant calls and tool results as a suffix.
  Earlier messages are not rebuilt from a shifting message window.
- Tool schemas use a fixed six-tool core plus deterministic domain bundles.
  Generic turns do not carry the full tool inventory.

These rules give Ollama a byte-stable prefix across tool iterations, which is
the prompt shape most likely to benefit from prefix caching.

### Turn-aware context trimming

Context selection is token-budgeted and groups messages by complete user turns.
Old turns are removed as units, while the current user/tool transaction is
preserved. If the current turn itself is too large, trimming happens in this
order:

1. tool observations,
2. assistant prose,
3. current user content as a last resort.

The recent_messages setting remains for API compatibility, but selection now
uses the token budget and whole-turn boundaries.

### Foreground-free compaction

Conversation compaction never runs before interactive generation. Once a final
answer has streamed, the CLI may enqueue a low-priority context_compaction job.
The worker waits for the interactive lease to become idle, summarizes a fixed
history prefix, and atomically advances the compacted_through_id watermark.

On restart, only chat rows newer than that watermark are loaded. Already
summarized rows cannot re-enter context or be summarized repeatedly.

### Large observation handles

Tool results larger than max_tool_output_chars are stored in
tool_observations. The model receives a head/tail preview and an opaque
observation ID. read_observation retrieves a bounded slice without carrying
the entire result through every later inference.

### Foreground-priority model scheduling

The worker checks the interactive activity lease immediately before every
research or compaction model request: planning, each page distillation,
evaluation, final synthesis, and context compaction. If the foreground is busy,
the job is checkpointed and released without consuming a retry.

The worker no longer performs a duplicate main-model warmup at startup.

## Default models

    agent:
      model: "qwen3.5:4b"
      fast_model: "qwen3.5:2b"

Qwen3.5 2B is used for non-thinking research planning, source distillation,
coverage evaluation, and ordinary custom-tool generation. The larger model
handles interactive responses, final research synthesis, and rolling-summary
compaction by default. Routine fast-model calls explicitly use think: false.

## Important configuration

The optimized defaults are in config/config.yaml:

    agent:
      thinking_default: false
      show_perf_stats: true
      max_tools_per_turn: 12

      context:
        num_ctx: 16384
        reserve_tokens: 1536
        recent_messages: 16
        summary_keep_messages: 8
        compact_at_tokens: 7500
        max_tool_output_chars: 5000
        tool_loop_reserve_tokens: 4096

      main_options:
        num_ctx: 16384
        temperature: 0.4
        top_p: 0.9
        top_k: 20

      fast_options:
        num_ctx: 8192
        temperature: 0.0
        top_p: 0.9
        top_k: 20

    worker:
      interactive_cooldown_seconds: 10
      fast_model_keep_alive: -1

fast_model_keep_alive: -1 keeps Qwen3.5 2B resident between worker calls.
Use 0 when the machine cannot keep both models resident without memory pressure.

## Ollama server settings

Apply ollama.env.example to the Ollama server/container, not the agent:

    OLLAMA_MAX_LOADED_MODELS=2
    OLLAMA_NUM_PARALLEL=1
    OLLAMA_FLASH_ATTENTION=1
    OLLAMA_KV_CACHE_TYPE=q8_0

If both models do not fit comfortably, use OLLAMA_MAX_LOADED_MODELS=1 and set
worker.fast_model_keep_alive to 0.

KV-cache quantization should be benchmarked on the target backend. The harness
does not set experimental llama-server RAM-cache variables.

## Telemetry

When show_perf_stats is enabled, each streamed response reports available
Ollama counters:

    TTFT 620 ms; prompt 2180; cached 1734 (79.5%); uncached 446;
    prefill 198.0 tok/s; generation 31 in 2820 ms (11.0 tok/s); load 0 ms

This distinguishes time to first output, prompt cache reuse, uncached prefill,
generation speed, and model load time. Not every Ollama version or backend
returns every field.

## Runtime layout

- agent.py — interactive CLI and foreground tool loop.
- worker.py — durable research, compaction, and host monitoring.
- tools/context.py — turn grouping, token budgeting, and trimming.
- tools/memory.py — memories, history watermark, and observations.
- tools/runtime.py — durable jobs, checkpoints, leases, and deferral.
- tools/deep_research.py — planning, collection, distillation, and evaluation.
- config/config.yaml — models, context budgets, and worker policy.

SQLite WAL mode allows both containers to share memory/knowledge.db.

## Research lifecycle

Research runs as a durable state machine:

    plan -> search/fetch/distill -> evaluate -> gap search -> synthesize -> persist

Jobs checkpoint after each phase and source query. They can be cancelled,
retried after transient failures, or recovered after a stale worker heartbeat.
Reports are written to workspace/research as Markdown and, when available, PDF.

## CLI commands

    /research <topic>      queue durable research
    /jobs                  list durable jobs
    /job <id>              inspect a job
    /cancel-job <id>       cancel a job
    /reminders             list reminders
    /think [on/off]        toggle explicit thinking
    /tools                 show the complete tool inventory
    /reload                reload built-in and custom tools
    /forget                clear history, summary, and stored observations
    exit                   close the CLI

## Custom tools

Custom tools live in workspace/custom_tools and must opt in explicitly:

    from tools.tool_registry import agent_tool

    @agent_tool(name="hello", description="Say hello.", readonly=True)
    def hello(name: str = "world") -> str:
        return f"Hello, {name}!"

Run /reload after adding or changing a tool.

## Deployment

Ensure the configured models exist:

    ollama pull qwen3.5:4b
    ollama pull qwen3.5:2b
    ollama ls

Build and start:

    docker compose build
    docker compose up -d worker
    docker compose run --rm agent

Or keep both services running:

    docker compose up -d agent worker

The project uses host networking and read-only host mounts for existing
system/network inspection features. Review docker-compose.yml before deployment.

## Validation

    python -m compileall -q .
    python -m pytest -q

The included suite covers token budgeting, turn boundaries, durable jobs,
foreground deferral, compaction watermarks, observation slices, reminders,
network URL validation, and tool selection.

End-to-end latency must be benchmarked on the target Ollama/GPU runtime. Use
the printed counters with identical prompts and model state when comparing
settings.
