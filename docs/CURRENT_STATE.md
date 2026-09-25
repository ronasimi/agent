# Al Agent current state — 2026-09-24

This file is the canonical point-in-time summary of the harness as shipped in this repository. `README.md` is the operator/user guide and `ARCHITECTURE.md` describes extension boundaries. The other dated review/audit documents are historical records; where they disagree with this file, this file and the current code/configuration are authoritative.

## Runtime model roles

The default deployment uses four aliased text-generation roles, a separate multimodal vision runner, and an optional embedding model. Deterministic code is always preferred before inference.

| Role | Default model | Context | Residency/use |
| --- | --- | ---: | --- |
| Decision | `agent-micro` → `qwen2.5-coder:0.5b` | 8K | Constrained JSON plan compilation, loop validation, retry/switch/block arbitration; never normal user prose |
| Executor | `agent-main` → `qwen2.5-coder:1.5b` | 16K | Default foreground native tool selection, bounded parameter construction, routine synthesis; also the compatibility `model`/`fast_model` alias |
| Reasoning | `agent-reasoning` → `hf.co/empero-ai/Qwen3.8-4B-Distill-GGUF:Q4_K_M` | 16K | Lazy text-only escalation for explicit Think, complex no-tool analysis, executor failure/capability recovery, and structured-plan final synthesis |
| Vision | `qwen3.5:4b` | 16K | Multimodal image/screenshot understanding; separate because the configured reasoning GGUF has no vision projector |
| Research | `agent-research` → `qwen3.5:9b` | 8K | Long-form research synthesis/factuality repair only; loaded for report stages and released afterward |
| Embedding (optional) | `nomic-embed-text` | n/a | Used only by semantic-memory embedding calls when semantic memory is enabled |

Steady-state interactive residency is **executor + decision**. With `OLLAMA_MAX_LOADED_MODELS=2`, a reasoning or vision request evicts the 0.5B decision runner while preserving the 1.5B executor. After the temporary 4B runner is released, the decision model is restored asynchronously. This prevents arbitrary Ollama eviction and keeps routine TTFT low.

### Escalation and tool-call policy

The 4B reasoner is not consulted on successful deterministic or ordinary 1.5B tool turns. Escalation is bounded and occurs for explicit Think mode, final synthesis of a compiled multi-step plan, clearly complex direct no-tool reasoning, missing executor capabilities, repeated executor no-progress, or low/medium-confidence `agent-micro` validator diagnoses such as `wrong_tool` / `bad_arguments`. An operational scheduler step that returns prose instead of a required native tool call consumes the no-progress budget; it can no longer retry for free.

Tool capability probing is behavioral: accepting a `tools` parameter without actually producing a native tool call is recorded as `accepted_unverified` and is not sufficient for tool-bearing execution. The executor then escalates rather than silently looping.

### Prompt-boundary policy

Simple conversational turns use a minimal prompt path and do not receive working-state, evidence-digest, recipe-control, or scheduler blocks. When evidence is needed, it is private system context rather than a synthetic user message. Output leakage guards reject internal harness headings.

### Model installation

`scripts/create_ollama_aliases.sh` installs the four aliases and the separate vision runner:

```bash
./scripts/create_ollama_aliases.sh
```

`nomic-embed-text` is not required for normal operation while `semantic_memory_enabled: false`.

## Tool routing and execution

The harness exposes a bounded, deterministic subset of the registered tools rather than placing the complete catalog in every prompt. The generated builtin manifest currently contains **237 tools**.

Important routing properties:

- deterministic fact/tool requirements are derived before model inference where possible;
- obvious exact paths such as current time and many structured fact checks can complete without a model call;
- `tool_search` is an escape hatch for a capability that was not initially exposed, not the normal path for already-visible tools;
- requirement-key-scoped evidence prevents one repeated primitive from accidentally satisfying multiple independent phases of a workflow;
- direct primitives are preferred over recipes for simpler one-step requests;
- shell/Python remain fallbacks rather than the primary interface.

The hard per-turn model-call budget remains a safety bound. Deterministic evidence collection, requirement closure, truncation recovery, and final structured formatting should not consume model calls merely for bookkeeping. Repeated empty responses, unusable/invented tool calls, and repeated inference exceptions are independently bounded by `model_no_progress_max_retries` (default **2**) so the global six-call budget remains a last-resort circuit breaker rather than the normal loop terminator.

### Durable deterministic compute

Universal-duration deterministic work is deliberately separated from that bounded foreground loop. `start_computation` queues a `durable_compute` job whose versioned machine metadata and sparse bidirectional tape are persisted in SQLite. Tape cells live in their own indexed table; each worker claim hydrates only the addresses reachable during its bounded quantum (10,000 transitions by default), then atomically commits changed cells plus lightweight checkpoint metadata and either completes or defers the job. A healthy defer does not consume retry attempts, and there is no mandatory total transition/yield ceiling. Optional `max_steps`, `max_tape_cells`, and `max_wall_time_seconds` policies default to `0` (unbounded) and are explicit job policy rather than hidden runtime limits.

`start_computation` statically rejects undefined transition targets and supports direct sparse `initial_tape` maps, negative/non-zero `initial_head` positions, and SHA-256-pinned workspace `input_file` sources (`text` or `tape_json`) so large finite inputs do not have to traverse model context. `get_computation_status` exposes bounded progress and a bounded tape window; `cancel_computation` is the operator escape hatch. Active start requests are idempotency-protected so ambiguous/retried tool calls do not create duplicate jobs.

Infrastructure recovery is tracked separately from ordinary job attempts. Stale claims and worker watchdog restarts increment a consecutive `recovery_failures` counter and return the ordinary claim attempt; healthy compute progress resets that counter. Durable starts default to `max_recovery_failures=10`, while `0` explicitly selects unlimited infrastructure recovery. The Web UI Jobs panel shows machine state, transitions, yields, tape-cell count, recovery count, and a cancel action. See `DURABLE_COMPUTE.md` for the machine schema and recovery contract.

The generic `execute_shell` / `execute_python` capability contract and implementation both cap one subprocess call at **120 seconds**. Recipes and the foreground LLM loop remain bounded; practical Turing completeness comes from arbitrarily many resumable deterministic worker quanta, not from removing those safeguards.

## Deterministic profile-fact reads

Explicit read-only questions for OOBE/profile-owned facts are resolved before recipe preflight, tool selection, prompt construction, or Ollama lock acquisition. The deterministic resolver currently covers name, saved location, timezone, role, email, interests, response style, research depth, profile-image presence, and broad saved-profile inspection. Mutation wording is never intercepted. If a specifically requested fact is absent, the turn falls through to the normal memory/model path rather than inventing a profile value.

## Working state and requirement ledger

Working state schema version 3 is persisted per conversation. The durable requirement ledger and model-visible requirement window are intentionally separate:

- persistent requirement capacity: **96 entries**;
- model-facing requirement window: **24 entries**;
- explicit required tool schemas are also bounded by `requirement_tool_cap: 24`;
- the working-state renderer remains subject to its overall character budget.

This allows large deterministic plans to remain inspectable/resumable without injecting the entire ledger into every model request. Requirement entries retain key, status, attempts, reason, scope, fingerprint, and provenance/evidence where applicable. Direct tool-backed requirements additionally persist a stable observation reference and a bounded evidence excerpt. Small direct results are force-archived when they close an explicit requirement, so evidence durability is independent of the shared `verified_observations` retention window. Persisted excerpts are deliberately removed from the model-facing requirement rendering; only their compact provenance/reference metadata is shown there.

## Observation storage and truncation recovery

Large tool results are persisted as durable observations. Prompt/state previews may contain `…[clipped]…`; this is display/storage compaction and **is not an observation-truncation signal**.

Actual middle truncation is tracked by structured harness metadata and a recoverable observation ID. The turn engine performs deterministic recovery during execution and then performs an authoritative **final settlement after the last deterministic tool call** (including cleanup/recipe-retention checks that may themselves create archived results). Rule/audit/finalization state is evaluated only from that settled snapshot. Recovery is bounded. If recovery fails or stops making contiguous progress, the affected evidence is marked terminally unresolved instead of reopening the interactive-model recovery loop until the model-call budget is exhausted. `read_observation` results do not recursively generate artificial truncation requirements.

## Recipes

Recipes live in `/app/memory/recipes.db` and are executable deterministic pipelines built from registered tools. Recipe lookup does **not** use `nomic-embed-text`. `search_recipes()` uses local SQLite FTS5 candidate retrieval when available plus bounded token-overlap scoring; its `semantic_score` field is a lexical similarity score, not a vector-embedding score.

The lifecycle is:

```text
successful read-only multi-tool trace
        ↓
search for an equivalent recipe
        ↓
workflow-wide deterministic generalization
        ↓
optional fast-model semantic naming hints
        ↓
deterministic validation / parameter rewrite
        ↓
user opt-in save
        ↓
load / run with normal tool policy
```

### Automatic parameter inference/generalization

Successful read-only workflows can be generalized automatically before a recipe candidate is offered:

- repeated or task-defining literals can become shared parameters;
- the same hostname used directly and inside URLs becomes one `hostname` parameter;
- derived strings use bounded `$template` references such as `https://{hostname}`;
- parameter types are inferred deterministically;
- operational controls such as timeouts, limits, booleans, offsets, ordinary fixed ports, and DNS record types stay constants unless the objective explicitly makes them variable;
- secret-like keys/values, credentials, observation IDs, timestamps, and transient tool output are excluded;
- successful runtime values are never written back into the stored recipe definition.

The fast 2B model may make one bounded advisory pass to suggest semantic parameter names. It cannot rewrite a pipeline, authorize an action, or introduce a value that was not present in the successful trace. Deterministic code validates every hint and falls back cleanly if the fast-model call fails or returns unusable JSON.

Recipe pipelines support direct `$param` references and bounded `$template` derivation. The same saved recipe can therefore be executed repeatedly with different runtime values without cloning one recipe per target.

## Fast-model responsibilities

The fast model is intentionally used only where a small, schema-constrained inference can reduce foreground work. Current uses include bounded loop validation/recovery, research planning/source distillation, and advisory recipe-parameter naming. Deterministic code remains authoritative for routing, policy, evidence accounting, parameter safety, recipe execution, and final structured formatting.

The recipe-parameter inference settings are:

```yaml
agent:
  recipes:
    fast_parameter_inference: true
    fast_parameter_min_stages: 2
    fast_parameter_max_calls_per_turn: 1
```

This auxiliary naming pass is bounded independently and is not a reason to raise the foreground hard model-call budget.

## Four-tier context storage

The interactive runtime now separates context by access pattern instead of treating every saved message as prompt text:

1. **Hot turn/task state:** the current turn lives in process memory; active working state is kept in a bounded process-local cache with SQLite/WAL write-through durability. Working-state schema/migration checks run once per database path rather than on every connection.
2. **Conversation continuity:** raw timestamped chat rows and rolling summaries remain in SQLite/WAL. Recent-history and summary reads use bounded process-local caches. Compaction is checked after every turn but runs only after the configured token threshold; it advances a watermark and never deletes the raw transcript.
3. **Historical recall:** `chat_history_fts` plus timestamp indexes support cross-conversation lexical/date retrieval without an embedding or model call. Requests such as “what did we discuss yesterday?” are date-resolved in the configured timezone and injected as bounded timestamped evidence.
4. **Durable memory:** explicit stable facts live separately in the `memory` table and use `memory_fts` for fast lexical retrieval. Semantic/vector memory remains opt-in.

Exact post-compaction model requests, including effective system messages, continue to be recorded in `memory/model_calls.jsonl` when tracing is enabled. See `CONTEXT_TIERS_2026-09-23.md`.

## Memory

Normal durable memory and semantic memory are separate:

- `memory/knowledge.db` contains ordinary memory, conversations, observations, jobs, and per-thread context;
- ordinary memory retrieval uses lexical relevance scoring by default;
- semantic memories are stored in a separate `semantic_memory` table with Ollama-generated vectors;
- semantic-memory query failure falls back to ordinary keyword memory search;
- the full memory table is never injected into every turn.

`semantic_memory_enabled: false` is the default and recommended setting when the optional embedding model is not installed.

## Web UI and artifacts

The Web UI is the only supported user interface. It provides persistent conversations, workspace browsing, jobs/reminders, inline media/doc previews, email cards, slash commands, tool status, recipe-save decisions, and optional live reasoning when Think is enabled. The old Working State panel is no longer exposed in the UI; working state remains an internal harness mechanism.

The wrench menu includes **Generate Bug Report**. Generation runs off the FastAPI event loop and writes a timestamped Markdown report to the repository root. The report includes a triage summary, bounded runtime/config state, timestamped recent conversation history, rolling summary, active working state, recent model-call wire traces and effective system prompts, tool/dependency health, browser benchmark state, storage health, Git state/diff data, a compact repository map, and runtime versions. Known credential/token fields are redacted.

Artifact rendering is presentation-only and remains separate from tool/storage semantics. Internal/transient stress fixtures such as `generalized_recipe_test/targets.txt` are deliberately excluded from automatic inline artifact cards and deterministic fallback file summaries. The file remains available to workspace tools, evidence accounting, and cleanup while it exists.

## Google Workspace

Gmail, Calendar, and Drive integrations are read-only. Tokens/client material are encrypted in the shared credential volume. The configured scopes are:

```text
https://www.googleapis.com/auth/gmail.readonly
https://www.googleapis.com/auth/calendar.readonly
https://www.googleapis.com/auth/drive.metadata.readonly
```

Drive metadata access also requires the Google Drive API to be enabled in the OAuth client's Google Cloud project.

## Research and reports

`/research` collects sources with bounded web primitives, uses the fast role for planning/distillation, and admits `agent-research` only for long-form synthesis/factuality repair. Reports target approximately 1,700–2,400 words by default, can include bounded inline images, and use claim/evidence checks before final output.

## Current validation baseline

For this repository state:

- broad offline regression suite in the build sandbox: **612 passed, 2 skipped** (the sandbox lacked the real Ollama/DDGS packages, so import-only test shims were used; live Ollama conformance remains deployment-host testing);
- Python byte-compilation: pass;
- Web UI JavaScript `node --check`: pass;
- architecture check: pass.

Live model latency/quality remains deployment-specific; use `diagnostics/benchmarks/benchmark_model_roles.py` on the actual Ollama host for TTFT, validator latency, report throughput, residency, and optional embedding measurements. Use `diagnostics/benchmarks/benchmark_durable_compute.py` for model-independent transition/checkpoint throughput.
