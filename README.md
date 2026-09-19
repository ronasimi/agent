<p align="center">
  <img src="webui/static/assets/agent-logo.png" alt="Al Agent logo" width="360">
</p>

# Al Agent

**Al Agent** is a local, Ollama-powered assistant harness designed for long-running work without making the interactive chat feel sluggish. It combines a responsive foreground agent with typed tools, reusable recipes, durable memory, background research, system/network diagnostics, reminders, and an optional browser-based UI.

The default configuration is tuned for a small local model pair:

- **Main model:** `qwen3.5:4b`
- **Fast model:** `qwen3.5:2b`
- **Context window:** 16K for the main agent

The main model handles the conversation and final synthesis. The fast model handles bounded validation, research planning, source distillation, and other work that can be offloaded without blocking the main loop.

## Quick start

### 1. Requirements

You need:

- Docker with the Compose plugin
- a running Ollama server reachable from the host network
- the models configured in `config/config.yaml`

Pull the default models if you do not already have them:

```bash
ollama pull qwen3.5:4b
ollama pull qwen3.5:2b
```

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
- conversation search
- a complete-chat copy button
- collapsible navigation and workspace sidebars
- drag-and-drop attachments
- inline previews for generated artifacts
- workspace upload/download controls
- jobs, reminders, and working-state views
- persistent branding and profile images

The full Al Agent image is used as the application logo. The yellow smiley face is used as the browser favicon.

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

Examples of available tool families include:

- **Time/system:** `current_time`, `environment_summary`, `host_snapshot`
- **Files/text:** `read_text`, `read_lines`, `text_search`, `json_query`, `write_file`
- **Network:** `local_subnets`, `scan_subnet`, `dns_diagnose`, `network_path`, `http_probe`
- **Web:** `web_search`, `browse_url`, `extract_document`, `page_diff`
- **Git/repository:** `repo_status`, `repo_diff`, `repo_checks`, `get_repo_map`
- **Research:** `enqueue_research`, `get_research_status`
- **Automation:** `schedule_reminder`, `list_reminders`, durable jobs
- **Media/profile:** `image_info`, `attach_media`, `set_profile_image`, `profile_image_info`
- **Recipes:** `search_recipes`, `run_recipe`, `save_recipe`, `run_pipeline`

Generic shell and Python execution exist as fallback capabilities, but structured tools are preferred and are only exposed when relevant or explicitly requested.

## Recipes

Recipes are durable reusable workflows built from primitives. The harness performs a recipe preflight before planning a task.

When a successful workflow does not match an existing recipe, the agent can ask whether you want to save it. Recipes are stored separately from model prompts and remain subject to current tool policy and user constraints.

Typical examples:

```text
weather lookup       web_search -> browse_url
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
  -> plan report
  -> collect media
  -> write sections
  -> assemble Markdown/PDF
```

Completed reports are written under:

```text
workspace/research/
```

Research jobs can survive process restarts because state and checkpoints are stored in SQLite.

## Reminders and scheduled work

The harness supports durable reminders and background jobs. The Web UI exposes them in dedicated views.

Useful CLI commands include:

```text
/jobs                  list durable jobs
/job <id>              inspect a job
/cancel-job <id>       cancel a job
/research <topic>      queue research
/optimize <objective>  queue an isolated optimization candidate
/optimizations         list optimization candidates
```

Model-facing reminder tools use the host's user-level systemd environment rather than generating ad-hoc unit files directly.

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

Weather is deliberately strict: a current-turn answer requires weather-bearing `web_search` **and** `browse_url` observations, a verified weather recipe/API observation, or a sufficiently fresh carried weather observation. `current_time` alone therefore produces a `missing_evidence` validator event. The harness then executes the built-in `weather.current_forecast` recipe (search → verified page → composed evidence), falling back to the same typed primitive chain if the recipe store is unavailable. Candidate factual prose is buffered until this gate passes, so a premature weather answer is discarded rather than streamed as a final response.

The default stored-weather freshness window is 10,800 seconds (3 hours) and can be changed with `agent.grounding.weather_max_age_seconds`. The grounding registry also covers current time, host state, network state, repository state, and explicit web-fact retrieval, and is intended to be extended with additional fact types as typed tools are added.

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
  model: "qwen3.5:4b"
  fast_model: "qwen3.5:2b"
  fast_model_keep_alive: 0
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
    prime_system_prefix: true

  working_state:
    minimize_schema_churn: true

  grounding:
    max_candidate_discards: 3

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

  recipes:
    validator_fallback_enabled: true
    validator_fallback_max_stages: 4
    validator_fallback_max_tools: 12
```

`fast_model_keep_alive: 0` is deliberate: every fast-model validator/research request asks Ollama to unload that model immediately after the request. This lowers host memory pressure at the cost of a possible cold-load delay the next time the fast model is needed. The interactive main model keeps its existing residency behavior.

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

When performance telemetry is enabled, responses can show TTFT, prompt tokens, cached tokens, prefill throughput, generation throughput, and load time when Ollama reports them.

### Time to first token

On a local server the dominant term in TTFT is prompt prefill, and Ollama/llama.cpp can only skip prefill for a prompt prefix that is **byte-identical** to the previous request. Chat templates render tool schemas and the system prompt at the very top of that prompt, so anything that changes early in the prompt costs a full re-prefill. The harness is built around that fact:

- **Volatile blocks go last.** The harness working state and evidence digest are rewritten on every tool-loop iteration. They are emitted after the stable system prompt and conversation history (`context.volatile_blocks_last`), so a changed working state no longer invalidates the cached prefix. Set it to `false` for a chat template that requires every system message to precede the conversation.
- **The tool set stays byte-stable.** Pruning satisfied requirement schemas, and reordering them pending-first, both invalidate the whole prefix. Under `working_state.minimize_schema_churn` they happen only during an iteration that must change the set anyway to expose a still-pending requirement. Repeats of a completed check are still suppressed deterministically, and the pending list is still carried by the working state.
- **Warm-up loads weights and primes the prefix.** Both frontends start a background warm-up (`warmup.enabled`) that loads the model and, with `warmup.prime_system_prefix`, sends the system prompt once at `num_predict: 1` so the first real turn prefills only what that turn adds. The warm-up deliberately uses `main_options`, because Ollama keys a loaded runner by context size.
- **Background compaction does not evict the foreground model.** When `compaction_model` is empty the worker reuses the interactive model, and it now reuses the interactive `num_ctx` as well. Requesting the same model with a smaller context would unload and reload it, making the next user turn pay a full model load plus a full prefill.
- **Harness control notes are de-duplicated.** Idempotent guidance ("the previous call was rejected", "the candidate answer was discarded") is appended once per turn instead of once per iteration, so the prompt stops growing when the loop is not making progress.
- **Selection has a relevance floor.** A single incidental description-word match no longer fills the per-turn schema budget, so conversational turns send no tool schemas at all instead of a dozen irrelevant ones.
- **The fallback finalizer streams.** The "safety limit reached" summary is streamed and emits deltas rather than blocking until the whole answer is generated.

`scripts/simulate_turns.py` measures the prefix-reuse effect of these without a model server.

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

The local test environment must have packages from `requirements.txt` installed. In particular, registry/Web UI imports require the Ollama Python package even when no live Ollama server is contacted.

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
