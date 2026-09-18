# Autonomous Local AI Agent Harness

Local Ollama agent with an interactive CLI, typed tools, durable SQLite state,
background research, rolling context summaries, host/network awareness,
systemd-based reminders, and a bounded self-optimization pipeline.

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
background model request, including research, compaction, and optimization.
If the foreground is busy, the job is checkpointed and released without
consuming a retry.

The worker no longer performs a duplicate main-model warmup at startup.

### Last-iteration tool-loop validation

Before the final allowed iteration of a tool loop, the fast model reviews a
bounded recent transcript and returns one constrained decision: `finish`,
`corrective_tool`, or `blocked`. An optional suggested tool must come from the
turn's existing allowlist. The harness converts that decision into its own
deterministic recovery prompt for the main model; validator prose and untrusted
tool output are never copied into the prompt, and the prompt is not persisted
to conversation history.

If validation times out or fails, the safe fallback permits at most one
distinct corrective tool call before finalization. Set
`agent.tool_loop_validator.enabled` to `false` to disable the check. The default
`keep_alive: 0` releases the fast model after validation on memory-constrained
hosts.

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
      max_iterations: 12
      max_tools_per_turn: 12

      tool_loop_validator:
        enabled: true
        timeout_seconds: 45
        max_transcript_chars: 12000
        keep_alive: 0

      context:
        num_ctx: 8192
        reserve_tokens: 1280
        recent_messages: 12
        summary_keep_messages: 8
        compact_at_tokens: 4500
        max_tool_output_chars: 4000
        tool_loop_reserve_tokens: 2048

      main_options:
        num_ctx: 8192
        temperature: 0.4
        top_p: 0.9
        top_k: 20

      fast_options:
        num_ctx: 4096
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
    OLLAMA_MAX_QUEUE=8
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
- tools/loop_validator.py — bounded fast-model tool-loop classification.
- tools/memory.py — memories, history watermark, and observations.
- tools/runtime.py — durable jobs, checkpoints, leases, and deferral.
- tools/deep_research.py — planning, collection, distillation, and evaluation.
- tools/repo_map.py — bounded file/symbol inventory and source retrieval.
- tools/self_optimization.py — candidate generation, gating, and approval export.
- scripts/benchmark_harness.py — fixed model-free repository-size benchmark.
- scripts/promote_optimization.py — explicit host-side approved-patch application.
- config/config.yaml — models, context budgets, and worker policy.

SQLite WAL mode allows both containers to share memory/knowledge.db.

## Research lifecycle

Research runs as a durable state machine:

    plan -> search/fetch/distill -> evaluate -> gap search -> synthesize -> persist

Jobs checkpoint after each phase and source query. They can be cancelled,
retried after transient failures, or recovered after a stale worker heartbeat.
Reports are written to workspace/research as Markdown and, when available, PDF.

## Self-optimization safety model

Self-optimization is deliberately a proposal pipeline, not live self-modifying
code. A job performs these steps:

1. copy only allowlisted source files from the read-only `/app/source` mount;
2. commit that snapshot and create a detached Git worktree;
3. run the fixed baseline test and benchmark commands;
4. ask the fast model for a narrow file plan and the main model for one bounded
   unified diff;
5. reject paths outside the allowlist, binary changes, symlinks, oversized
   patches, and patches that do not apply cleanly;
6. submit baseline and candidate snapshots to the dedicated
   `optimizer-validator` container, which has no network, a read-only root, no
   host/runtime mounts, dropped capabilities, a PID limit, a timeout, and a
   memory/CPU limit;
7. persist the patch, SHA-256 digest, metrics, logs, and approval state in
   SQLite and `workspace/self_optimization`.

If the validator is unavailable, validation times out and fails closed. A
Bubblewrap path remains available only for operators who explicitly disable
the dedicated runner; it also fails closed when namespaces are unavailable.
The model-facing tools can enqueue and inspect candidates but cannot approve
them. The `/approve-optimization` CLI command exports a digest-pinned patch;
it still does not alter the live source.

Queue and inspect a candidate:

    /optimize reduce prompt tokens without changing tool behavior
    /optimizations

After reviewing the report, changed files, test logs, and full digest:

    /approve-optimization <candidate-id> <full-sha256>

Then stop the harness and apply the approved patch from the host in a clean Git
checkout:

    python scripts/promote_optimization.py \
      --repo . \
      --patch workspace/self_optimization/approved/<candidate-id>.patch \
      --sha256 <full-sha256> \
      --confirm APPLY_APPROVED_PATCH

Review the diff and commit it normally. If the extracted project is not already
a Git repository, initialize and commit the baseline before running promotion.

### Repository context size

`scripts/benchmark_harness.py` estimates source context at roughly one token per
four bytes. The current harness is below the configured 80,000-token full-tree
gate, but normal agent and optimization turns do not inject the full tree.
`get_repo_map`, `search_repo_symbols`, and `read_repo_symbol` retrieve only the
map and relevant slices, while optimization source context is capped at 32,000
characters.

## CLI commands

    /research <topic>      queue durable research
    /optimize <objective>  queue an isolated optimization candidate
    /optimizations         list candidate reports and approval states
    /approve-optimization <candidate-id> <sha256>
                            export a reviewed patch; never applies it
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

The Dockerfile performs the Arch upgrade and package install in one `pacman`
transaction, uses `--needed`, and clears the package cache. A long package
download is not necessarily a crash; `docker compose build --progress=plain`
shows the active package and final exit status. Re-run the build after an
interrupted transaction rather than attaching to a half-created container.

Or keep both services running:

    docker compose up -d agent worker

On the first `docker compose up`, the one-shot `storage-init` service creates
`memory`, `workspace`, the research/custom-tool directories, and the complete
self-optimization validation tree. It assigns them to the runtime
`1000:1000` user before any long-running service starts. The initializer is
idempotent: subsequent runs repair ownership and permissions without deleting
the database, reports, tools, or candidate state. A successful run remains
visible as an exited container and is expected:

    docker compose ps -a storage-init
    docker compose logs storage-init

SQLite creates `memory/knowledge.db` and initializes its schema when the file
does not exist. Existing databases and WAL files are preserved.

The project uses host networking and read-only host mounts for existing
system/network inspection features. The validator is intentionally separate
and uses `network_mode: none`; do not add the host mounts or Docker socket to
that service. Review docker-compose.yml before deployment.

## Validation

    python -m compileall -q .
    python -m pytest -q

The included suite covers token budgeting, turn boundaries, durable jobs,
foreground deferral, compaction watermarks, observation slices, reminders,
network URL validation, repository traversal rejection, candidate audit state,
digest-pinned approval, patch policy, and tool selection.

End-to-end latency must be benchmarked on the target Ollama/GPU runtime. Use
the printed counters with identical prompts and model state when comparing
settings.
