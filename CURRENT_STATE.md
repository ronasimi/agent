# Al Agent current state — 2026-09-23

This file is the canonical point-in-time summary of the harness as shipped in this repository. `README.md` is the operator/user guide and `ARCHITECTURE.md` describes extension boundaries. The other dated review/audit documents are historical records; where they disagree with this file, this file and the current code/configuration are authoritative.

## Runtime model roles

The default deployment uses three generative Ollama roles. The embedding model is optional because semantic memory is disabled by default.

| Role | Default model | Context | Residency/use |
| --- | --- | ---: | --- |
| Main | `agent-main:4b` → `hf.co/empero-ai/Qwen3.8-4B-Distill-GGUF:Q4_K_M` | 16K | Foreground chat, reasoning, coding, tool orchestration; also the default vision role |
| Fast | `agent-main:2b` → `hf.co/empero-ai/Qwen3.8-2B-Distill-GGUF:Q8_0` | 16K | Bounded validator/recovery work, research support, source distillation, and advisory recipe-parameter naming |
| Report | `agent-report:9b` → `qwen3.5:9b` | 8K | Long-form research synthesis/factuality repair only; loaded for report stages and released afterward |
| Embedding (optional) | `nomic-embed-text` | n/a | Used only by semantic-memory embedding calls and the embedding benchmark when semantic memory is enabled |

`vision_model` currently aliases `agent-main:4b`. `vision.supports_images` is enabled experimentally in `config/config.yaml`; if the configured runner rejects image payloads, disable that flag or point `vision_model` at a supported multimodal model.

The main and fast roles use distinct models but both use a 16K runner configuration. `fast_model_keep_alive: -1` pins the fast role after it is loaded; startup prewarms main and then fast. The report role has a finite keep-alive and temporarily displaces interactive residency when required.

### Model installation

`scripts/create_ollama_aliases.sh` pulls and creates the main and fast aliases. The report alias is separate:

```bash
./scripts/create_ollama_aliases.sh
ollama pull qwen3.5:9b
ollama cp qwen3.5:9b agent-report:9b
```

`nomic-embed-text` is **not required for normal operation with the default configuration**:

```yaml
agent:
  semantic_memory_enabled: false
  embed_model: "nomic-embed-text"
```

With semantic memory disabled, normal turn memory lookup uses the deterministic lexical `search_memory()` path. Pull the embedding model only if semantic memory or explicit semantic-memory tools are required:

```bash
ollama pull nomic-embed-text
```

The embedding model is currently used by `remember_semantic()`, `search_semantic_memory()` / `get_relevant_memories()` when semantic memory is enabled, and the optional embedding-latency benchmark. Recipe search, skill search, tool discovery, observation retrieval, and ordinary routing do not require it.

## Tool routing and execution

The harness exposes a bounded, deterministic subset of the registered tools rather than placing the complete catalog in every prompt. The generated builtin manifest currently contains **232 tools**.

Important routing properties:

- deterministic fact/tool requirements are derived before model inference where possible;
- obvious exact paths such as current time and many structured fact checks can complete without a model call;
- `tool_search` is an escape hatch for a capability that was not initially exposed, not the normal path for already-visible tools;
- requirement-key-scoped evidence prevents one repeated primitive from accidentally satisfying multiple independent phases of a workflow;
- direct primitives are preferred over recipes for simpler one-step requests;
- shell/Python remain fallbacks rather than the primary interface.

The hard per-turn model-call budget remains a safety bound. Deterministic evidence collection, requirement closure, truncation recovery, and final structured formatting should not consume model calls merely for bookkeeping.

## Working state and requirement ledger

Working state schema version 3 is persisted per conversation. The durable requirement ledger and model-visible requirement window are intentionally separate:

- persistent requirement capacity: **96 entries**;
- model-facing requirement window: **24 entries**;
- explicit required tool schemas are also bounded by `requirement_tool_cap: 24`;
- the working-state renderer remains subject to its overall character budget.

This allows large deterministic plans to remain inspectable/resumable without injecting the entire ledger into every model request. Requirement entries retain key, status, attempts, reason, scope, fingerprint, and provenance/evidence where applicable.

## Observation storage and truncation recovery

Large tool results are persisted as durable observations. Prompt/state previews may contain `…[clipped]…`; this is display/storage compaction and **is not an observation-truncation signal**.

Actual middle truncation is tracked by structured harness metadata and a recoverable observation ID. The turn engine runs deterministic `read_observation` recovery before truncation/evidence audits. Recovery is bounded. If recovery fails or stops making contiguous progress, the affected evidence is marked terminally unresolved instead of reopening the main-model recovery loop until the model-call budget is exhausted. `read_observation` results do not recursively generate artificial truncation requirements.

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

## Memory

Normal durable memory and semantic memory are separate:

- `memory/knowledge.db` contains ordinary memory, conversations, observations, jobs, and per-thread context;
- ordinary memory retrieval uses lexical relevance scoring by default;
- semantic memories are stored in a separate `semantic_memory` table with Ollama-generated vectors;
- semantic-memory query failure falls back to ordinary keyword memory search;
- the full memory table is never injected into every turn.

`semantic_memory_enabled: false` is the default and recommended setting when the optional embedding model is not installed.

## Web UI and artifacts

The Web UI is the only supported user interface. It provides persistent conversations, workspace browsing, jobs/reminders, inline media/doc previews, email cards, slash commands, tool status, and recipe-save decisions.

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

`/research` collects sources with bounded web primitives, uses the fast role for planning/distillation, and admits `agent-report:9b` only for long-form synthesis/factuality repair. Reports target approximately 1,700–2,400 words by default, can include bounded inline images, and use claim/evidence checks before final output.

## Current validation baseline

For this repository state:

- full deterministic/offline test suite: **520 passed, 1 skipped**;
- Python byte-compilation: pass;
- Web UI JavaScript `node --check`: pass;
- generated builtin manifest: current at **232 tools**.

Live model latency/quality remains deployment-specific; use `scripts/benchmark_model_roles.py` on the actual Ollama host for TTFT, validator latency, report throughput, residency, and optional embedding measurements.
