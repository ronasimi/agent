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
- Tool schemas use a small fixed read-oriented core plus deterministic domain bundles and lexical matches.
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

### Cross-model context sharing and tool-loop validation

The main and fast models share **bounded semantic context**, not a KV cache.
Different model weights cannot safely reuse each other's attention cache. The
harness instead gives the fast validator the same relevant rolling summary,
recent conversational setup, recalled memory, current objective, and selected
tool capabilities that informed the main model. Raw tool observations stay in
the separate loop transcript and remain explicitly untrusted.

After three deterministic failed/no-progress attempts on one step, the fast
model validates the loop before an unvalidated fourth attempt. It returns a
constrained control decision (`retry`, `switch_tool`, `finish`, or `blocked`)
plus a structured diagnosis such as `bad_arguments`, `wrong_tool`, or
`task_complete`. Only those structured fields are shared back to the main
model; free-form validator reasoning and untrusted tool text are never copied
into the main-model control prompt. If a validator-approved retry of the same
stalled step also fails, that requirement is deterministically marked blocked
for the rest of the turn rather than consuming another three retries. A separate
final-iteration validator remains as a last safety net.

If validation times out or fails, the safe fallback permits one bounded
corrective attempt rather than terminating the task silently. Set
`agent.tool_loop_validator.enabled` to `false` to disable validation, or
`agent.model_context_sharing.enabled` to `false` to keep the validator limited
to the current request and loop transcript.

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

      model_context_sharing:
        enabled: true
        max_chars: 6000
        summary_chars: 2200
        recent_chars: 1800
        memory_chars: 1200
        tool_chars: 1600

      tool_loop_validator:
        enabled: true
        failed_step_attempts: 3
        max_interventions_per_turn: 3
        max_candidate_tools: 24
        timeout_seconds: 45
        max_transcript_chars: 12000
        keep_alive: -1

      context:
        num_ctx: 16384
        reserve_tokens: 2048
        recent_messages: 12
        summary_keep_messages: 8
        compact_at_tokens: 9000
        max_tool_output_chars: 4000
        tool_loop_reserve_tokens: 3072

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
- tools/model_context.py — bounded semantic state bridge shared by main/fast models.
- tools/loop_validator.py — bounded fast-model tool-loop classification.
- tools/turn_policy.py — deterministic per-turn user tool restrictions and conditional unlocks.
- tools/host_diagnostics.py — process, PSI, filesystem, and systemd health snapshots.
- tools/network_diagnostics.py — neighbors, connections, DNS, path, endpoint, and HTTP diagnostics.
- tools/web_research.py — metadata, links, feeds, documents, fingerprints, and page diffs.
- tools/repo_diagnostics.py — Git status/diff, bounded repo checks, dependency and tool health.
- tools/observation_tools.py — deterministic diffs between durable tool observations.
- tools/memory.py — memories, history watermark, and observations.
- tools/runtime.py — durable jobs, checkpoints, leases, and deferral.
- tools/deep_research.py — planning, collection, distillation, and evaluation.
- tools/repo_map.py — bounded file/symbol inventory and source retrieval.
- tools/self_optimization.py — candidate generation, gating, and approval export.
- scripts/benchmark_harness.py — fixed model-free repository-size benchmark.
- scripts/promote_optimization.py — explicit host-side approved-patch application.
- config/config.yaml — models, context budgets, and worker policy.

SQLite WAL mode allows both containers to share memory/knowledge.db.

## Structured diagnostics and research utilities

The harness exposes narrow tools so the 4B model does not need to construct shell commands for common diagnosis:

- Host: `process_snapshot`, `pressure_snapshot`, `filesystem_snapshot`, `service_health`.
- Network: `neighbor_snapshot`, `connection_snapshot`, `dns_diagnose`, `network_path`, `endpoint_probe`, `http_probe`. `network_path` prefers MTR JSON but also parses the standard text report emitted by distro builds that ignore/override `--json`.
- Web/document research: `page_metadata`, `page_links`, `discover_site`, `read_feed`, `extract_document`, `page_fingerprint`, `page_diff`.
- Harness/repository: `repo_status`, `repo_diff`, `repo_checks`, `dependency_audit`, `tool_health`, `diff_observations`.

`page_diff` stores only a bounded normalized prior page snapshot in monitor state and returns a bounded unified diff. `repo_checks` accepts only the known `compile`, `config`, `ruff`, and `pytest` checks and runs against a temporary source copy. `tool_health` reports missing command dependencies before the model wastes retries on an unavailable tool.

Explicit per-turn restrictions are enforced by the harness. For example, `do not use execute_shell unless your first approach fails` removes `execute_shell` from the first inference schema and exposes it only after an unsuccessful allowed-tool iteration. Read-only requests disable mutating tools for the whole turn. Non-zero shell/Python exits with genuinely useful stdout are labeled `status=partial`; non-zero exits without useful stdout remain `status=error`.

## Research lifecycle

Research runs as a durable state machine:

    plan -> search/fetch/distill -> evaluate -> gap search -> report plan
         -> collect media -> write sections -> write overview -> assemble -> persist

The fast model plans the report and assigns source IDs to sections. The main
model writes each section independently with only its relevant evidence, so a
long evidence bundle no longer competes with the entire report for one context
window. Source pages retain ranked `og:image`, `twitter:image`, and content-image
candidates; selected images are downloaded into a sibling `*_assets` directory
and embedded with relative paths. Decorative images are not required.

Jobs checkpoint after each phase, source query, and drafted section. They can be
cancelled, retried after transient failures, or recovered after a stale worker
heartbeat. Each completed run records `markdown_path`, `pdf_path`, `asset_dir`,
and `report_word_count` in job state. Markdown and PDF are written to
`workspace/research`, and the PDF resolves the same local media used by Markdown.

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

### Repository context size and shared working state

`scripts/benchmark_harness.py` estimates source context at roughly one token per
four bytes. The full-tree guard is 160,000 estimated tokens; normal agent and
optimization turns never inject the full tree. `get_repo_map`,
`search_repo_symbols`, and `read_repo_symbol` retrieve only the map and relevant
slices, while optimization source context is capped at 32,000 characters.

Interactive turns also maintain a small persisted harness working state in
SQLite. It is the common source of truth for the 4B main model and 2B validator:
current objective, explicit constraints, selected tool capabilities,
provenance-tagged tool evidence, failed approaches, current recovery plan, and
structured validator decisions. With working state enabled, the main prompt
keeps only the current raw conversation turn instead of re-ingesting older raw
turns already represented by the state/rolling summary. Tool evidence previews
remain untrusted data, and only harness code can commit state changes. For broad
explicit checklists, satisfied/blocked requirement schemas are also removed from
subsequent Ollama calls and exact successful repeats are suppressed unless the
user explicitly requests a recheck/monitoring workflow. This reduces prefill and
helps the 4B model move through remaining checks instead of revisiting completed
ones.

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

### Tool-produced media / vision

The main Qwen3.5 model is multimodal. Tools can return a media-aware result containing
normal text plus local image references. The frontend encodes those files and injects
them into the next Ollama message through the native `images` field, so the model sees
the actual pixels rather than only a filename.

`take_web_screenshot()` now automatically returns its PNG as attached media and includes
bounded visible page text, title, and final URL in the tool result. This means a request
such as `take a screenshot of cnn.com and describe it` can be completed in one tool loop.
If the screenshot is blank or blocked, the model receives that blank image and is
explicitly instructed to report that fact instead of guessing the page layout.

The generic `attach_media(path, context="")` tool can re-attach an existing workspace
PNG/JPEG/WebP image or the first page of a PDF. Public image URLs are also supported and
remain subject to the existing public-URL and response-size checks. Tool media is bounded
by `agent.vision.max_images_per_turn` and `agent.vision.max_image_bytes`.

## Optional Web UI sidecar

A browser frontend is included under `webui/`. It uses the same `agent.py` runtime, SQLite state, tools, validator, jobs, reminders, and working-state store as the CLI. It is a second frontend, not a second agent implementation.

Start the normal stack plus the Web UI:

```bash
docker compose --profile web up -d --build
```

By default the UI binds only to the local machine at:

```text
http://127.0.0.1:8080
```

To expose it on the LAN, explicitly opt in before starting the profile:

```bash
WEBUI_HOST=0.0.0.0 docker compose --profile web up -d --build
```

If you expose the interface beyond localhost, put it behind an authenticated reverse proxy. The sidecar intentionally does not expose direct HTTP endpoints for shell execution or arbitrary tool invocation; messages always pass through the existing agent control loop and its policies.

The UI provides:

- a ChatGPT-inspired three-column layout with a left navigation rail, centered chat, and optional **Files in workspace** drawer
- colors loaded from the host `~/.Xresources` at startup (with the bundled Base16-style fallback palette if unavailable)
- streaming assistant output over WebSocket with lightweight Markdown/code rendering
- collapsible tool execution/result cards and fast-validator events
- image/PDF and bounded text-file uploads
- a ChatGPT-style **+** button in the composer plus drag-and-drop file/media attachment onto the message box
- actual multimodal image attachment through the existing Ollama media path
- a toggleable `/app/workspace` browser with folder navigation, filtering, open/download, **attach to message**, and **Add file** uploads directly into the current workspace folder
- current harness working-state inspection
- durable jobs and reminders views
- stop/cancel for an active browser turn
- shared CLI/Web conversation history
- automatic inline previews for files created by agent tools during a Web UI turn, with persistent Open/Download actions (images, PDF first pages, text/Markdown, audio, and video where supported)
- workspace artifact access without exposing arbitrary host paths

The CLI and Web UI serialize foreground inference using `/app/workspace/.agent_inference.lock`, which prevents two frontend processes from racing the single Ollama inference slot. Background durable worker jobs retain their existing scheduling behavior.

Optional environment variables:

```text
WEBUI_HOST=127.0.0.1
WEBUI_PORT=8080
WEBUI_MAX_UPLOAD_BYTES=16777216
WEBUI_WORKSPACE_LIST_LIMIT=250
WEBUI_PREVIEW_TEXT_BYTES=49152
WEBUI_ARTIFACT_SCAN_LIMIT=10000
WEBUI_ARTIFACT_MAX_PER_TURN=24
```

The Compose profile points `WEBUI_XRESOURCES` at the host user's `~/.Xresources` through the existing read-only `/host` mount. Recognized `*.foreground`, `*.background`, `*.cursorColor`, and `*.color0` through `*.color15` values are applied as CSS theme variables; unrelated X resources are ignored.

## Deterministic clock and environment primitives

The agent includes small read-only primitives for common questions that should not require shell execution:

- `current_time()` returns the current UTC timestamp plus configured local time/date/timezone.
- `hostname()` returns host and runtime hostnames without performing DNS lookups.
- `environment_summary()` returns a bounded non-secret runtime summary (clock, platform, Python, configured models, and workspace availability).

`current_time()` is part of the always-available read-only tool core and is also a deterministic requirement for direct current-time/date requests. The runtime policy explicitly forbids inferring the current clock from uptime or stale observation timestamps. Runtime containers receive `TZ` plus a read-only `/etc/localtime` bind mount so local timestamps follow the host/configured timezone.

## Unix-style primitives, pipelines, and recipes

Al Agent exposes small typed primitives for filesystem discovery/reads, text and
JSON transforms, process/system inspection, network layers, web extraction,
documents/media metadata, Git, safe SQLite reads, arithmetic/time, encoding, and
IP/URL utilities. The main model normally sees only the small subset selected for
the current request.

`run_pipeline` can execute up to sixteen **read-only** primitive stages inside the
harness. Later stages can consume earlier structured output without copying the
intermediate data through model context:

```json
[
  {"id":"s1","tool":"resolve_host","args":{"host":{"$param":"host"}}},
  {"id":"s2","tool":"route_lookup","args":{"target":{"$ref":"s1","path":"addresses.0"}}}
]
```

Pipeline references use `{"$ref":"stage-id","path":"field.0"}` and recipe
parameters use `{"$param":"name","default":"optional value"}`. Generic pipelines
reject mutating tools, shell/Python execution, recursive pipeline/recipe calls,
and more than sixteen stages. Pipelines also support bounded `foreach` fan-out, conditional `when` stages, optional stages, and `$item` references so high-level workflows can be expressed without round-tripping intermediate data through the model.

Reusable recipes are stored separately in `/app/memory/recipes.db`. The store has
an FTS5 semantic index over recipe names, descriptions, tags, and tool names, plus
usage/success counters and parameterized pipeline JSON. After a successful
nontrivial read-only workflow, the harness checks for an existing semantic match.
If none exists it asks whether the workflow should be saved. Reply `yes save it`,
`save it as <name>`, or `no thanks`. The Web UI presents the same prompt with Save
and Not now buttons.

Useful recipe tools are `search_recipes`, `list_recipes`, `run_recipe`, and
`save_recipe`. A semantically relevant saved recipe is surfaced automatically on
future turns, but current user constraints and tool policy always take precedence.

Harness-owned compatibility recipes are seeded automatically into the same semantic recipe table. They are versioned with `origin=builtin`, use names such as `compat.host_snapshot`, and reproduce high-level diagnostic/research tools from smaller primitives wherever the semantics can be preserved safely. `recipe_coverage()` reports the full/partial/native-only coverage matrix and documents why a remaining monolithic tool cannot be represented as a read-only recipe. User recipes remain separate (`origin=user`) and builtin recipe names are reserved.

## Modular extension architecture

The runtime has been refactored so the top-level `agent.py` and `worker.py` are compatibility/composition entry points rather than feature monoliths.  Tool loading is declarative and provider-based, Unix primitives are split by domain, background job types are auto-discovered providers, CLI slash commands are registered handlers, and Web UI filesystem/chat/theme concerns live in separate modules.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the module map and extension recipes.  The short version is:

- add primitives under `tools/primitive_modules/`;
- add builtin tool manifests under `tools/provider_groups/`;
- add durable job handlers under `al_agent/background/job_providers/`;
- add CLI commands in `al_agent/cli_commands.py`;
- keep deterministic multi-step behavior in pipelines/recipes instead of growing the turn engine;
- keep `agent.py`, `worker.py`, and `tools/primitive_ops.py` as stable compatibility facades.
