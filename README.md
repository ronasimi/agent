<p align="center">
  <img src="webui/static/assets/agent-logo.png" alt="Al Agent logo" width="360">
</p>

# Al Agent

**Al Agent** is a local, Ollama-powered assistant harness designed for long-running work without making the interactive chat feel sluggish. It combines a responsive foreground agent with typed tools, reusable recipes, durable memory, background research, system/network diagnostics, reminders, and an optional browser-based UI.

The default configuration uses three base Qwen3.5 roles plus an embedding model:

- **Main model:** `agent-main:4b` → `qwen3.5:4b`
- **Fast model:** `agent-fast:2b` → `qwen3.5:2b`
- **Report model:** `agent-report:9b` → `qwen3.5:9b`
- **Embedding model:** `nomic-embed-text`
- **Context window:** 16K for the main agent

The main model handles conversation and interactive reasoning. The 2B fast model handles recovery validation, lightweight planning, source distillation, and other bounded reasoning. The 9B model is loaded only for long-form report synthesis. `nomic-embed-text` provides semantic retrieval for memory, knowledge, and recipes.

## Quick start

### 1. Requirements

You need:

- Docker with the Compose plugin
- a running Ollama server reachable from the host network
- the models configured in `config/config.yaml`

Create the stable role aliases after pulling the base Qwen3.5 models:

```bash
# Create the role aliases from already-pulled base Qwen3.5 models:
./scripts/create_ollama_aliases.sh
```

The aliases map to:

- `agent-main:4b` → `qwen3.5:4b`
- `agent-fast:2b` → `qwen3.5:2b`
- `agent-report:9b` → `qwen3.5:9b`

The harness passes explicit sampling/context settings per role, so behavior does not depend on alias-local Modelfile parameters.

### 2. Start the agent

From the repository root:

```bash
docker compose up -d --build
docker attach agent
```

The `storage-init` service creates and fixes ownership for persistent `workspace/` and `memory/` storage on first run, so a fresh clone can be started directly.

Detach from the CLI without stopping it with `Ctrl-P`, `Ctrl-Q`.

### 3. Start the Web UI

The Web UI is optional and uses the Compose `web` profile:

```bash
docker compose --profile web up -d --build
```

Then open:

```text
http://127.0.0.1:8080
```

To expose the UI on your LAN, set `WEBUI_HOST=0.0.0.0` deliberately. The default is localhost-only.

## Web UI

The Web UI provides a ChatGPT-style local interface with:

- streaming responses and inline tool activity
- a status message directly below the active user prompt
- persistent command history with Up/Down navigation
- slash-command autocomplete: type `/` to see every command, usage, and description
- deterministic slash-command routing before the LLM (commands never become model prompts)
- durable per-conversation history in the Recent sidebar
- non-destructive New Chat creation with isolated working state/observations
- a complete-chat copy button
- collapsible navigation and workspace sidebars
- drag-and-drop attachments
- inline previews for generated artifacts
- workspace upload/download controls
- jobs, reminders, and working-state views
- persistent branding and profile images

The full Al Agent image is used as the application logo. The yellow smiley face is used as the browser favicon.

On first run, the Web UI opens a profile questionnaire for stable user context (name, role, timezone, optional location/email/interests, response style, research depth, and optional profile image). **Profile setup** in the sidebar can rerun it and replace questionnaire-owned context. The CLI runs the same OOBE on an interactive TTY and exposes `/profile` to rerun it.

Each browser thread has its own conversation ID, chat rows, rolling summary, tool observations, compaction watermark, and working state. Creating a new chat no longer deletes the previous thread.

### User profile picture

The footer avatar is loaded from durable memory at:

```text
/app/memory/profile/user_picture.png
```

The Web UI and agent containers share the `memory/` volume, so changing the profile image survives restarts and rebuilds.

When you explicitly identify an attached image as a picture of yourself, the agent is instructed to ask whether you want to use it as the Web UI profile image. If you approve, the agent uses the typed `set_profile_image` tool to validate the workspace image, normalize it to PNG, and copy it into durable profile storage. The Web UI refreshes the avatar after a successful tool call.

For compatibility with older conversations, the Web UI can migrate a previously remembered `user_photo`/`user_picture` by finding the corresponding attached workspace image in chat history. It does **not** use face recognition to decide that a person in an image is you.

## How the harness is organized

Al Agent follows a small-tools/recipes approach rather than giving the model a few giant monolithic tools.

```text
User request
    │
    ├─ recipe preflight
    ├─ relevant memory/context retrieval
    ├─ small tool-schema selection
    ▼
Main model ──────── typed primitive tools
    │                    │
    │                    ├─ files/text/JSON
    │                    ├─ host/process/storage
    │                    ├─ network/DNS/LAN
    │                    ├─ web/document research
    │                    ├─ reminders/jobs
    │                    └─ profile/media operations
    │
    ├─ working-state ledger
    └─ fast-model validator on repeated failure/stall
```

Successful multi-tool workflows can be saved as recipes so the model can reuse a known deterministic sequence instead of rediscovering it on every run.

## Tool behavior

The harness does **not** inject all tools into every model request. It selects a small relevant subset based on the current request, recent conversational context, explicit completion requirements, and recipe matches. This reduces prompt prefill and improves tool choice on smaller local models.

Live-fact routing is separated from implementation intent before schemas reach the model. For example, “weather in London” can expose weather tools, while “refactor the weather validator” cannot accidentally become a forecast request. The same distinction applies to time, news, host, network, and repository prompts. Deictic follow-ups such as “latest local headlines” may inherit the prior news location, while a new topical request such as “latest AI news” starts a new frame.

Examples of available tool families include:

- **Time/system:** `current_time`, `environment_summary`, `host_snapshot`
- **Files/text:** `read_text`, `read_lines`, `text_search`, `json_query`, `write_file`
- **Network:** `local_subnets`, `scan_subnet`, `dns_diagnose`, `network_path`, `http_probe`
- **Web:** `web_search`, `browse_url`, `extract_document`, `page_diff`
- **Weather:** `geocode_location`, `weather_forecast` (structured first; verified web fallback)
- **News:** `news_search` (location-aware query, filtering, and grounding)
- **Git/repository:** `repo_status`, `repo_diff`, `repo_checks`, `get_repo_map`
- **Research:** `enqueue_research`, `get_research_status`
- **Automation:** `schedule_reminder`, `list_reminders`, durable jobs
- **Media/profile:** `image_info`, `attach_media`, `set_profile_image`, `profile_image_info`
- **Recipes:** `search_recipes`, `run_recipe`, `save_recipe`, `run_pipeline`

Generic shell and Python execution exist as fallback capabilities, but structured tools are preferred and are only exposed when relevant or explicitly requested.

## Recipes

Recipes are durable reusable workflows built from primitives. The harness performs a recipe preflight before planning a task.

When a successful workflow with at least two meaningful stages does not match an existing recipe, the agent can ask whether you want to save it. A built-in workflow such as the structured weather path is not suggested as a duplicate recipe. Recipes are stored separately from model prompts and remain subject to current tool policy and user constraints. Explicit `save recipe` wording is used for recipe persistence so a request such as “save it as report.md” remains an artifact-save request.

Typical examples:

```text
weather lookup       geocode_location -> weather_forecast  (web_search -> browse_url fallback)
host health check    host_snapshot -> pressure_snapshot -> filesystem_snapshot
LAN discovery        local_subnets -> scan_subnet
repository review    get_repo_map -> repo_status -> repo_checks
```

## Research jobs

Long research tasks are moved out of the foreground conversation so the main loop remains responsive.

```text
/research <topic>
```

The worker checkpoints the research lifecycle:

```text
plan
  -> search/fetch/distill
  -> evaluate coverage
  -> fill evidence gaps
  -> build source-verbatim claim ledger
  -> plan report with dedicated report_model
  -> collect media
  -> write sections
  -> factuality gate + bounded repair
  -> write/audit front matter
  -> assemble Markdown/PDF + audit sidecars
```

Completed reports are written under:

```text
workspace/research/
```

Research jobs can survive process restarts because state and checkpoints are stored in SQLite.

### Dedicated report model and factuality gate

`/research` now separates source collection from long-form synthesis. Search planning,
source distillation, and gap detection continue to use the small fast model, while the
report stage uses the configurable `agent.report_model` (default:
`agent-report:9b`). Before any prose is drafted, the report
model builds a per-source claim ledger. Every retained claim must include a support
excerpt that the harness verifies occurs in the fetched raw source text.

Sections are written only from that verified ledger. Each generated section and the
front matter then pass a structured factuality gate that checks for unsupported facts,
citation mismatches, overstated causality, unattributed analysis, invented specifics,
and source-scope errors. Failed passages receive bounded repair passes; sections that
still fail degrade to a deterministic ledger-only form instead of shipping unsupported
prose. When enabled, audit sidecars are written next to the report as
`*.claims.json` and `*.factuality.json`.

The worker also swaps Ollama residency around synthesis: it unloads the interactive
and fast models before loading the report model, keeps the larger writer resident only
for the report stage, then unloads it and restores the main model. Only that main-model
restore occurs inside the shared inference lock. The fast model is prewarmed afterward
on a deduplicated maintenance thread during an idle window, so foreground work never
queues behind the optional preload. The report model has a finite keep-alive TTL as a
crash-safety backstop. If a user turn arrives, the report worker yields and the
interactive path evicts any lingering report model before loading main. This behavior
is controlled by `agent.report_model`, `agent.report_options`, and
`agent.report_restore_models_after_stage`.

## Reminders and scheduled work

The harness supports durable reminders and background jobs. The Web UI exposes them in dedicated views.

The CLI and Web UI share the same slash-command registry. In the Web UI, type `/` to open the searchable command menu.

Useful slash commands include:

```text
/jobs                  list durable jobs
/job <id>              inspect a job
/cancel-job <id>       cancel a job
/research <topic>      queue research
/optimize <objective>  queue an isolated optimization candidate
/optimizations         list optimization candidates
```

Model-facing reminder tools use the host's user-level systemd environment rather than generating ad-hoc unit files directly. If that backend is unavailable, the tool removes any partial unit files, reports a terminal `tool_unavailable` result, and is not retried with degraded or missing arguments during the same turn.

## Conversation memory and context

Al Agent separates several kinds of state:

- `memory/knowledge.db` — durable memories, conversation rows, job state, observations
- rolling conversation summary — older conversation context after compaction
- working state — current objective, requirements, evidence, failures, and validator decisions
- `workspace/` — files, uploads, generated artifacts, reports, recipes/custom tools where applicable

Large tool outputs are stored as durable observations. The model receives a bounded preview plus an observation ID and can retrieve another slice with `read_observation` instead of carrying a huge result through every inference.

### Why old chats do not continuously grow the prompt

Conversation compaction runs in the background after a turn rather than before the next response. A durable watermark prevents already summarized history from being pulled back into the raw model context.

The Web UI's **copy entire chat** function is different: it can export the complete stored conversation, including rows that have already been compacted out of model context.

## Model protocol reliability

The Ollama boundary is deliberately isolated from the semantic/tool loop. Streamed `tool_calls` are accumulated across chunks, tool results use Ollama-native `tool_name`, and local bookkeeping fields are stripped before messages are sent on the wire. A transient model transport failure may be retried only **before** the first streamed chunk; after any output or tool call arrives, the request is never replayed because doing so could duplicate output or side effects. These settings live under `agent.model_transport`.

See `HARNESS_BEST_PRACTICES_REVIEW.md` for the 2026 small-local-model architecture review and comparison with smolagents, LangGraph/Deep Agents, PocketFlow, Ollama, and llama.cpp patterns.

## Failure recovery and validator

Repeated identical tool failures are tracked by the harness. After the configured threshold, the fast model acts as a bounded validator and returns a structured control decision such as:

```text
retry
switch_tool
finish
blocked
```

The main model receives the structured diagnosis, not unrestricted hidden validator reasoning. Deterministic fallback behavior is used if the validator itself times out or emits malformed output.

If normal correction still fails, the harness now has one final fall-through before it gives up: the fast validator may propose a small **ephemeral read-only recovery recipe** made from allowlisted typed primitives. The harness re-validates the recipe, rejects repeated calls and side-effecting tools, executes it at most once, and then finalizes from whatever evidence it obtained. The recipe is never silently persisted; if it succeeds and does not match an existing recipe, the normal opt-in save prompt is shown afterward.

### Hard fact grounding

Fact-retrieval turns have a deterministic finalization gate in addition to the model-based loop validator. The gate classifies the fact type requested by the user and checks harness-owned observation provenance/content before a factual answer can finalize. A successful unrelated observation does not satisfy the gate.

Weather is deliberately strict: a current-turn answer requires a weather-bearing structured provider observation, a verified weather recipe, linked `web_search` + `browse_url` observations, or a sufficiently fresh carried weather observation. `current_time` never satisfies weather. Before the first answer generation, the harness executes the built-in `weather.current_forecast` recipe (`geocode_location` → `weather_forecast` → composed evidence) using the requested location or the explicitly stored OOBE location. If the structured provider fails, it falls back to `web_search` → `browse_url`. Pre-generation evidence acquisition is normal tool work and does not surface a `missing_evidence` validator warning; that warning is reserved for an actual recovery/finalization failure. Candidate factual prose is buffered until the grounding gate passes.

The default stored-weather freshness window is 10,800 seconds (3 hours) and can be changed with `agent.grounding.weather_max_age_seconds`. The grounding registry also covers current time, host state, network state, repository state, and explicit web-fact retrieval, and is intended to be extended with additional fact types as typed tools are added.

Location-scoped news uses the same principle. The task frame carries a canonical city/region, the search receives separate query and location fields, and the grounding gate only accepts observations that match that scope. Same-name-city noise and conflicting country domains are filtered before a local headline answer can finalize.

See `PROMPT_ROUTING_AND_FALLTHROUGH_REVIEW.md` for the reviewed prompt pairs, corrected failure paths, and control-loop invariants.

This is intended to prevent loops such as repeatedly calling the same failing web endpoint or repeatedly retrying a tool with unchanged bad arguments, while still allowing a materially different primitive composition as the final recovery attempt.

## Local network diagnostics

For LAN discovery, use the structured path:

```text
local_subnets -> scan_subnet
```

`scan_subnet` performs bounded read-only discovery and can report available information such as:

- IP/hostname
- reverse DNS
- MAC/vendor when available
- listening service/port information
- bounded service/OS hints

`network_reachability` is for external/public reachability checks and is intentionally not used as a LAN scanner.

## Configuration

Most behavior is configured in `config/config.yaml`.

Important defaults:

```yaml
agent:
  model: "agent-main:4b"
  fast_model: "agent-fast:2b"
  report_model: "agent-report:9b"
  fast_model_keep_alive: -1
  report_model_keep_alive: "10m"
  report_restore_models_after_stage: true

  report_options:
    num_ctx: 8192
    temperature: 0.6
    top_p: 0.95
    top_k: 20
  thinking_default: false
  max_iterations: 12
  max_tools_per_turn: 12

  context:
    num_ctx: 16384
    reserve_tokens: 2048
    compact_at_tokens: 9000
    max_tool_output_chars: 4000
    volatile_blocks_last: true

  warmup:
    enabled: true
    fast_model_prewarm: true
    prime_system_prefix: false

  working_state:
    minimize_schema_churn: true

  grounding:
    max_candidate_discards: 3

  main_options:
    num_ctx: 16384
    temperature: 0.6
    top_p: 0.95
    top_k: 20

  fast_options:
    num_ctx: 4096
    temperature: 0.6
    top_p: 0.95
    top_k: 20

  recipes:
    validator_fallback_enabled: true
    validator_fallback_max_stages: 4
    validator_fallback_max_tools: 12
```

`fast_model_keep_alive: -1` pins the canonical 4K fast runner after its first load. With `OLLAMA_MAX_LOADED_MODELS=2`, the 4B main and 2B fast roles can normally coexist. Startup warms main first and then schedules fast prewarming; report teardown restores main synchronously and schedules fast only after foreground inference is available again.

### Ollama server settings

`ollama.env.example` contains suggested server-side settings:

```text
OLLAMA_MAX_LOADED_MODELS=2
OLLAMA_MAX_QUEUE=8
OLLAMA_NUM_PARALLEL=1
OLLAMA_FLASH_ATTENTION=1
OLLAMA_KV_CACHE_TYPE=q8_0
```

Apply these to the Ollama server/container, not to the agent container itself.

On a machine that cannot keep both models resident comfortably, use one loaded model and set the fast-model keep-alive to `0`.

## Performance design

The harness is optimized around local-model constraints:

- stable system/prompt prefixes for better Ollama prefix-cache reuse
- small per-turn tool schema sets
- foreground-priority inference locking
- background compaction instead of pre-response compaction
- bounded tool observations
- whole-turn context trimming
- fast-model offload for validation and research support
- removal of completed requirement schemas during long checklist tasks

Performance telemetry separates user-visible and backend costs: queue wait, turn preparation, model load, prompt evaluation, model TTFT, first visible answer, cache-hit percentage, generation counts, and total turn time. The latest measurements are also persisted in monitor state as `agent.last_model_stats` and `agent.last_turn_metrics`.

### Time to first token

On a local server the dominant term in TTFT is prompt prefill, and Ollama/llama.cpp can only skip prefill for a prompt prefix that is **byte-identical** to the previous request. Chat templates render tool schemas and the system prompt at the very top of that prompt, so anything that changes early in the prompt costs a full re-prefill. The harness is built around that fact:

- **Volatile blocks go last.** The harness working state and evidence digest are rewritten on every tool-loop iteration. They are emitted after the stable system prompt and conversation history (`context.volatile_blocks_last`), so a changed working state no longer invalidates the cached prefix. Set it to `false` for a chat template that requires every system message to precede the conversation.
- **The tool set stays byte-stable.** Pruning satisfied requirement schemas, and reordering them pending-first, both invalidate the whole prefix. Under `working_state.minimize_schema_churn` they happen only during an iteration that must change the set anyway to expose a still-pending requirement. Repeats of a completed check are still suppressed deterministically, and the pending list is still carried by the working state.
- **Warm-up loads weights; prefix priming is measurement-driven.** Both frontends start a background warm-up (`warmup.enabled`) that loads the model using the same `main_options` as interactive turns. `warmup.prime_system_prefix` defaults to `false`: tool-capable chat templates may place dynamic schemas before messages, so a system-only prime is not guaranteed to be a reusable prefix. Use `scripts/benchmark_warmup.py` and `prompt_eval_cached_count` before enabling it.
- **Background compaction does not evict the foreground model.** When `compaction_model` is empty the worker reuses the interactive model, and it now reuses the interactive `num_ctx` as well. Requesting the same model with a smaller context would unload and reload it, making the next user turn pay a full model load plus a full prefill.
- **Harness control notes are de-duplicated.** Idempotent guidance ("the previous call was rejected", "the candidate answer was discarded") is appended once per turn instead of once per iteration, so the prompt stops growing when the loop is not making progress.
- **Selection has a relevance floor.** A single incidental description-word match no longer fills the per-turn schema budget, so conversational turns send no tool schemas at all instead of a dozen irrelevant ones.
- **The fallback finalizer streams.** The "safety limit reached" summary is streamed and emits deltas rather than blocking until the whole answer is generated.

`scripts/simulate_turns.py` measures prompt-prefix behavior without a model server. On the target Ollama host, `scripts/benchmark_warmup.py` compares cold, weight-preloaded, and system-prefix-primed turns using the real harness options and reports load/prefill/cache metrics.

## Self-optimization

Self-optimization is a proposal pipeline, not live autonomous modification.

A candidate is built in an isolated worktree, tested, benchmarked, and validated in the restricted `optimizer-validator` container. The model cannot approve or deploy its own patch.

Typical flow:

```text
/optimize reduce prompt tokens without changing tool behavior
/optimizations
/approve-optimization <candidate-id> <full-sha256>
```

After approval, promotion is an explicit host-side action:

```bash
python scripts/promote_optimization.py \
  --repo . \
  --patch workspace/self_optimization/approved/<candidate-id>.patch \
  --sha256 <full-sha256> \
  --confirm APPLY_APPROVED_PATCH
```

Review and commit the resulting diff normally.

## Repository layout

```text
agent.py                 interactive CLI entry point
worker.py                background worker
al_agent/                turn engine, prompts, state, events
webui/                   FastAPI sidecar and browser UI
tools/                   tool registry, primitives, diagnostics, recipes
config/config.yaml       model and harness configuration
scripts/                 initialization, benchmarks, optimization utilities
tests/                   model-free/unit reliability tests
workspace/               persistent working files and generated artifacts
memory/                  persistent SQLite state and user profile assets
```

Some especially useful modules:

- `al_agent/turn_engine.py` — foreground model/tool state machine
- `tools/catalog.py` — dynamic tool loading and schema selection
- `tools/loop_validator.py` — fast-model recovery decisions
- `tools/working_state.py` — durable objective/evidence/requirements state
- `tools/recipe_store.py` — semantic recipe storage and lookup
- `tools/network_diagnostics.py` — host/network diagnosis
- `tools/user_profile.py` — profile data and durable profile image handling
- `webui/server.py` — Web UI API
- `webui/static/app.js` — browser interaction and streaming UI

## Model roles and performance benchmarking

The runtime deliberately keeps the model hierarchy small:

- `agent-main:4b` handles interactive reasoning, coding, conversation, and tool orchestration.
- `agent-fast:2b` handles tool-loop validation, recovery reasoning, research planning, source distillation, and other bounded auxiliary work.
- `agent-report:9b` is admitted only for long-form research synthesis and factuality repair.
- `nomic-embed-text` supplies semantic vectors for memory, knowledge, and recipe retrieval.

Deterministic fast paths remain preferred for exact requests such as current time, structured weather, and market quotes; those paths avoid an unnecessary model call entirely.

For a fresh clone or extracted ZIP, bootstrap the repo-local Python environment once:

```bash
./scripts/bootstrap_venv.sh
```

The `.venv/` directory is intentionally not committed or packaged because Python virtual environments are platform-specific and may contain absolute interpreter paths. The bootstrap script recreates it from `requirements.txt`/`pyproject.toml`. Host-side scripts such as the model-role benchmark automatically re-exec under `.venv/bin/python` once it exists.

Use the deployment-host benchmark to measure whether model-role changes actually improve the target machine:

```bash
python scripts/benchmark_model_roles.py --runs 20 --report-runs 1
```

It reports cold and warm 4B TTFT and 2B validator latency separately, includes every
warm sample, records 9B throughput and residency snapshots, and issues a foreground
main request while fast prewarming is in flight to expose server-level resource
contention. Use `--skip-residency` when you do not want the benchmark to disturb
current Ollama residency (`--skip-load-swap` remains a compatibility alias).

## Testing

Run the unit suite with:

```bash
python -m pytest -q
```

Run architecture checks with:

```bash
python scripts/check_architecture.py
```

Run the fixed harness benchmark with:

```bash
python scripts/benchmark_harness.py
```

Benchmark live Ollama model roles on the deployment host with:

```bash
python scripts/benchmark_model_roles.py --runs 20 --report-runs 1
```

Simulate turns without an Ollama server. A scripted client replaces the model
transport and can only call tools that were actually supplied in the request,
so the traces exercise the turn state machine rather than its malformed-call
path. The prefix report approximates how much of each request Ollama can serve
from its KV cache:

```bash
python scripts/simulate_turns.py            # traces plus prefix reuse
python scripts/simulate_turns.py --prefix   # prefix reuse only
python scripts/simulate_turns.py --prompts  # include rendered prompts
```

It writes to a temporary database and never touches durable storage.

The local test environment must have packages from `requirements.txt` installed. On a fresh checkout/ZIP, run `./scripts/bootstrap_venv.sh`; this creates `.venv/` and installs the project in editable mode with its declared dependencies. In particular, registry/Web UI imports require the Ollama Python package even when no live Ollama server is contacted.

## Troubleshooting

### `docker compose up` reports `getwd: no such file or directory`

Your shell is currently inside a directory that was deleted or replaced. Change to a real path before running Compose:

```bash
cd /path/to/agent
pwd
docker compose --profile web up -d --build
```

### Web UI is not running

The Web UI uses a Compose profile. Start it with:

```bash
docker compose --profile web up -d webui
```

Then inspect:

```bash
docker compose logs -f webui
```

### A tool exists but the model says it is unavailable

The harness intentionally exposes only a relevant subset of tool schemas each turn. The model's system policy tells it that absence from the current schema set does not mean the capability does not exist. For discovery, the harness should use `tool_health`; `/reload` is for an actual registry reload, not routine discovery.

### Profile image does not appear

Check that the durable file exists:

```bash
ls -l memory/profile/user_picture.png
```

If you previously told the agent that an uploaded image was a photo of you, the Web UI will attempt a one-time migration from legacy `user_photo`/chat history when `/api/profile-image` is requested. The original upload must still exist in `workspace/` for migration to succeed.

## Security notes

- The Web UI binds to `127.0.0.1` by default.
- Host filesystem mounts are read-only where possible.
- Structured read-only tools are preferred over generic command execution.
- Custom tools are statically validated before loading.
- Self-optimization validation runs in a restricted container and cannot self-promote.
- Profile images are only installed from files already inside the agent workspace and only after explicit user approval through the profile-image tool workflow.

## License

Use the project under the terms of the repository's license, if present.
