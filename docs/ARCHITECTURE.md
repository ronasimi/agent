# Al Agent architecture

> **Current-state note (2026-09-23):** This document describes the current architecture. `CURRENT_STATE.md` is the compact canonical deployment/status summary; dated review files are historical snapshots.

The repository uses **thin composition roots + domain modules + declarative providers**.  The goal is that adding a capability is normally additive: create a focused implementation and register a small manifest, rather than editing the agent loop.

## Runtime layers

```text
webui/__main__.py / worker.py           executable entry points
        │
        ├── al_agent/                   application orchestration
        │   ├── runtime.py              stable Web UI runtime facade
        │   ├── state.py                configuration + long-lived services
        │   ├── events.py               frontend event bridge + inference lock
        │   ├── prompts.py              stable policy, memory, media encoding
        │   ├── turn_support.py         deterministic loop helpers
        │   ├── turn_engine.py          one responsibility: model/tool state machine
        │   ├── model_protocol.py       Ollama wire normalization + safe stream transport behavior
        │   ├── fast_tasks.py           bounded advisory fast-model extraction/classification helpers
        │   ├── slash_commands.py       browser command registry + handlers
        │   ├── compute/                deterministic resumable machine core
        │   │   └── machine.py          versioned sparse-tape quantum executor
        │   └── background/             durable worker subsystem
        │       ├── resources.py        foreground/resource arbitration
        │       ├── research.py         research/report job handler
        │       ├── maintenance.py      compaction + monitoring
        │       ├── types.py            provider-neutral JobHandler contract
        │       ├── handlers.py         job-provider discovery/dispatch
        │       └── job_providers/      one provider module per job family
        │
        ├── tools/                      capabilities and durable stores
        │   ├── catalog.py              loading, metadata, schema selection
        │   ├── providers.py            provider discovery + selection policy
        │   ├── provider_groups/        builtin tool manifests
        │   ├── primitive_modules/      Unix-style primitive domains
        │   ├── primitive_ops.py        compatibility re-export facade
        │   ├── pipeline.py             bounded read-only composition
        │   ├── recipe_store.py         recipe persistence + local FTS/token lookup
        │   ├── recipe_learning.py      successful-trace generalization + save candidates
        │   ├── credential_store.py     encrypted provider-neutral secret vault
        │   ├── google_workspace_auth.py OAuth state/PKCE/token lifecycle
        │   ├── google_workspace.py     bounded read-only Gmail/Calendar tools
        │   └── ...                     focused domain tools/stores
        │
        └── webui/                      sole user interface
            ├── __main__.py             container entry point
            ├── server.py               FastAPI route composition root
            ├── chat.py                 WebSocket turn streaming
            ├── workspace_ops.py        workspace/artifact/upload operations
            ├── theme.py                Xresources palette loading
            ├── history.py              browser history serialization
            └── static/                 dependency-free SPA, rich renderers, interaction state
```

## Adding a primitive

1. Implement the function in the matching `tools/primitive_modules/<domain>.py` module (or create a new focused domain module).
2. Add its `(module, function)` pair to a provider file in `tools/provider_groups/`.  A new provider file is auto-discovered.
3. Add selector terms to `tools/providers.py` only when lexical/bundle discovery needs help.
4. Add a focused test.  Keep the primitive deterministic, bounded, typed, and read-only when possible.

Historical imports from `tools.primitive_ops` continue to work, but new code should use the domain module.

## Adding a normal builtin tool

Implement the tool in a focused `tools/<domain>.py` module and add a `TOOL_SPECS` entry to an existing/new file under `tools/provider_groups/`.  `tools.catalog` discovers provider modules automatically.  The catalog and turn engine should not need changes.

Mutating/repeat-safe/artifact policy belongs in `tools/providers.py`, not the implementation.

## Adding a custom workspace tool

Use the existing `@agent_tool` decorator and save the validated module under `/app/workspace/custom_tools`.  `tools.catalog.load_tools()` discovers decorated functions after static safety validation.

## Adding a durable background job

Create `al_agent/background/job_providers/pNN_name.py` and export exactly one `JOB_HANDLER` from `al_agent.background.types`. The generic worker loop discovers it automatically. Do not add another branch to `worker.py`.

Long-running deterministic work must be **cooperative**: execute one bounded quantum, persist all information needed to reconstruct the continuation, and atomically defer the job. Large mutable state should not be recopied into every checkpoint; `durable_compute` stores sparse tape cells separately and commits only the changed cells with lightweight machine metadata. A healthy yield is not a retry and must not consume the job-attempt budget. Keep worker watchdogs scoped to one claim rather than treating them as a lifetime limit. `durable_compute` is the reference implementation; see `DURABLE_COMPUTE.md`.

## Durable deterministic computation

`al_agent.compute.machine` provides the deterministic execution substrate for work that may require an arbitrary number of state transitions. The main LLM loop remains deliberately bounded. `durable_compute` instead executes a bounded quantum, loads only its reachable sparse-tape window, commits tape deltas plus versioned machine metadata, yields the queue claim, and may be claimed again without a predetermined number of resumptions. `HALT`, explicit cancellation, a malformed program/checkpoint, or an explicitly configured resource policy are terminal. Transition targets are validated before enqueueing so references must resolve to another transition state or an explicit halt state.

Checkpoint metadata, sparse tape deltas, and the queue-state transition are committed in one SQLite transaction so a worker cannot advertise a resumable job without the corresponding continuation state. Initial input may be supplied as inline text, an arbitrary sparse tape/head configuration, or a hash-pinned workspace file; the file digest prevents queued semantics from changing after submission. Status APIs expose only bounded progress/tape windows rather than the entire tape.

Worker/process recovery has a distinct budget from handler attempts. Infrastructure recovery returns the consumed claim attempt and increments `recovery_failures`; healthy progress resets the consecutive counter. This separation prevents a months-long healthy computation from exhausting its normal retry budget merely because the worker was restarted occasionally, while still allowing an operator-configured finite recovery ceiling. Together these boundaries provide practical universal-computation semantics without turning the probabilistic foreground model loop into an unbounded `while True`.

## Adding a browser command

Add slash commands to `al_agent/slash_commands.py`. The Web UI consumes the registry for autocomplete and deterministic execution, which stays outside the model prompt loop.

## Adding a Web UI capability

Keep HTTP route composition in `webui/server.py`, but put domain behavior in a sibling module (`workspace_ops`, `chat`, `theme`, etc.).  The browser must not bypass the main agent tool/policy loop.

Rich output remains split by responsibility: `rich_output.js` performs escaped semantic rendering for Markdown, email cards, and stored attachment markers; `app.js` owns DOM placement and media controls; `interaction_state.js` provides bounded local persistence for recipe decisions. Workspace media is always served through the path-validated `/api/files` and preview routes. Modern office-document previews extract only bounded text from selected archive parts.

## External integration boundary

OAuth client data and user tokens are encrypted by `LocalCredentialStore` before reaching SQLite. Provider, account, and record-kind identifiers are authenticated inside each encrypted envelope; browser status APIs expose only deliberately non-secret metadata. New service integrations should reuse this store rather than create plaintext token files.

The Google Workspace adapter is intentionally split into three layers: FastAPI only orchestrates setup redirects, `google_workspace_auth.py` owns one-time state/PKCE/refresh/revocation, and `google_workspace.py` owns bounded REST reads. Model-facing tool output never contains credentials, and provider text is explicitly marked untrusted. Every new provider should preserve the same separation between setup/control-plane routes and model-facing data-plane tools.

## Fact grounding gate

Before a fact-retrieval answer can finalize, `tools/grounding.py` compares the requested fact type with harness-owned successful observations. This is a deterministic control boundary: executor/reasoning or decision-validator text cannot override `missing_evidence`. Weather requires verified weather-bearing provenance (current `web_search` + `browse_url`, a verified weather recipe/API result, or a fresh carried observation). The turn engine performs one built-in weather recipe/fallback recovery and re-runs the gate before finalization. Working-state observations persist compact `fact_types`, `source_tools`, `weather_verified`, `turn_id`, and timestamp metadata so the check does not depend on clipped model-visible excerpts.

## Pipelines and recipes

`run_pipeline` is the preferred composition mechanism for deterministic read-only chains. Intermediate stage values remain in the harness and can be referenced using `$ref`; bounded `foreach`, `$item`, conditional `when`, and optional stages cover common Unix-style map/filter/branch patterns without another model turn. Successful reusable workflows may be saved to the recipe database without changing Python code. Recipe lookup is local SQLite FTS5 candidate retrieval plus bounded token-overlap scoring; it does not use the embedding model.

Harness-owned compatibility recipes live in auto-discovered `tools/recipe_provider_groups/` manifests. They are seeded idempotently into the recipe database with `origin=builtin`, a stable key/version, and a target monolithic tool. Every high-level tool in the diagnostic/web/repository compatibility surface must either have a full/partial recipe or an explicit `NATIVE_ONLY` reason; `recipe_coverage()` exposes that matrix. Add new compatibility recipes as provider manifests rather than hard-coding them into the recipe store.

`tools/recipe_learning.py` generalizes successful read-only traces before a user recipe candidate is offered. It infers shared task inputs across stages, keeps operational constants fixed unless the objective makes them variable, rejects secret-like literals, and rewrites derived strings with bounded `$template` references. `al_agent/fast_tasks.py` may ask the fast role for semantic parameter-name hints, but the fast model cannot rewrite the pipeline or introduce unseen values; deterministic code validates every hint and performs the actual rewrite.

## Working state, requirements, and observations

Working state is persisted per conversation with schema version 3. The durable requirement ledger stores up to 96 entries, while only 24 requirement rows are rendered into the model-facing working-state block. Required native tool schemas are independently bounded by `requirement_tool_cap` (24 by default). This separation lets large deterministic plans remain inspectable and resumable without paying their full token cost on every model call.

Requirement evidence is provenance-aware and can distinguish direct tool observations, tool-surface discovery, `tool_search` discovery, and derived audits. Repeated use of the same primitive can be recorded against a specific requirement key so an early call does not accidentally satisfy a later workflow phase.

Large results are persisted as tool observations. Preview compaction may render `…[clipped]…`, but only structured middle-truncation metadata creates a recovery obligation. The turn engine performs bounded `read_observation` recovery before truncation/evidence audits; a failed or non-progressing recovery becomes a terminal unresolved evidence gap instead of consuming the model-call budget in a recovery loop.

## Dependency direction

- the Web UI depends on `al_agent` application APIs;
- `al_agent` depends on `tools` capabilities/stores;
- tool implementations must not import a frontend;
- provider manifests describe capabilities but do not execute them;
- generic loops (`turn_engine`, worker `runner`) should not contain domain-specific tool/job implementations.

`tests/test_modular_architecture.py` and `diagnostics/check_architecture.py` enforce the main structural invariants.

## Model roles

The runtime uses a deterministic-first **decision → executor → reasoning → research** hierarchy with a separate vision path:

- `agent-micro` (`qwen2.5-coder:0.5b`) handles constrained plan compilation, validation, and retry/switch/block arbitration. It never authors normal user-facing prose or arbitrary tool arguments.
- `agent-main` (`qwen2.5-coder:1.5b`) is the default executor and native tool caller. It handles bounded arguments, ordinary conversation, and support/extraction work after deterministic routing has reduced the action space.
- `agent-reasoning` (`hf.co/empero-ai/Qwen3.8-4B-Distill-GGUF:Q4_K_M`) is the lazy text reasoning escalation role for explicit Think, complex no-tool analysis, structured-plan final synthesis, capability fallback, and bounded recovery.
- `qwen3.5:4b` is the distinct multimodal vision runner. The configured reasoning GGUF is text-only, so image inputs never route to `agent-reasoning`.
- `agent-research` (`qwen3.5:9b`) is admitted only for long-form research synthesis and factuality repair.
- `nomic-embed-text` remains optional while semantic memory is disabled.

Steady state keeps the 1.5B executor and 0.5B decision model resident. Reasoning or vision explicitly frees the decision-model slot while preserving the executor, then restores `agent-micro` asynchronously. This matches a two-runner Ollama budget without forcing ordinary turns through a 4B model.

Deterministic routing, requirements, grounding, safety policy, parameter validation, recipe execution, scheduler advancement, and obvious typed steps such as route-table checks bypass model inference whenever possible. An operational step that receives prose instead of a required native tool call consumes a bounded no-progress budget and escalates/terminalizes rather than looping. Capability probes must verify actual native tool-call behavior before a cached profile is considered tool-capable.

Internal working state and evidence remain system-private. Simple chat omits those blocks entirely; tool/evidence context is injected only when relevant and is never represented as a user-authored evidence digest.

`diagnostics/benchmarks/benchmark_model_roles.py` measures executor TTFT, decision-validator latency, reasoning TTFT, research throughput, residency swaps, and contention.

## Multi-fact turn model

The routing layer distinguishes the primary conversational frame from factual completion requirements. `task_frame` remains the single primary compatibility frame, while `fact_frames` contains one scope per requested fact type. Grounding metadata, deterministic recovery, and fact-tool pruning consume the fact-specific frame rather than reusing the primary frame across domains.

`FactGroundingLedger` preserves satisfaction independently for each factual requirement. A failure or retry for one fact does not reopen a previously grounded fact. This is especially important for compound turns such as weather plus news, market quotes plus current time, or other mixed retrieval requests.

The parser detects fact types before clause decomposition and assigns shared/local time, location, and topic modifiers afterward. This avoids naive conjunction splitting and guarantees a frame exists for every detected required fact type. See `MULTI_FACT_GROUNDING_2026-09-22.md` for details.
