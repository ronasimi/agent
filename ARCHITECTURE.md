# Al Agent architecture

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
        │   ├── slash_commands.py       browser command registry + handlers
        │   └── background/             durable worker subsystem
        │       ├── resources.py        foreground/resource arbitration
        │       ├── research.py         research/report job handler
        │       ├── maintenance.py      compaction + monitoring
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
        │   ├── recipe_store.py         semantic recipe persistence
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

Create `al_agent/background/job_providers/pNN_name.py` and export exactly one `JOB_HANDLER`.  The generic worker loop discovers it automatically.  Do not add another branch to `worker.py`.

## Adding a browser command

Add slash commands to `al_agent/slash_commands.py`. The Web UI consumes the registry for autocomplete and deterministic execution, which stays outside the model prompt loop.

## Adding a Web UI capability

Keep HTTP route composition in `webui/server.py`, but put domain behavior in a sibling module (`workspace_ops`, `chat`, `theme`, etc.).  The browser must not bypass the main agent tool/policy loop.

Rich output remains split by responsibility: `rich_output.js` performs escaped semantic rendering for Markdown, email cards, and stored attachment markers; `app.js` owns DOM placement and media controls; `interaction_state.js` provides bounded local persistence for recipe decisions. Workspace media is always served through the path-validated `/api/files` and preview routes. Modern office-document previews extract only bounded text from selected archive parts.

## External integration boundary

OAuth client data and user tokens are encrypted by `LocalCredentialStore` before reaching SQLite. Provider, account, and record-kind identifiers are authenticated inside each encrypted envelope; browser status APIs expose only deliberately non-secret metadata. New service integrations should reuse this store rather than create plaintext token files.

The Google Workspace adapter is intentionally split into three layers: FastAPI only orchestrates setup redirects, `google_workspace_auth.py` owns one-time state/PKCE/refresh/revocation, and `google_workspace.py` owns bounded REST reads. Model-facing tool output never contains credentials, and provider text is explicitly marked untrusted. Every new provider should preserve the same separation between setup/control-plane routes and model-facing data-plane tools.

## Fact grounding gate

Before a fact-retrieval answer can finalize, `tools/grounding.py` compares the requested fact type with harness-owned successful observations. This is a deterministic control boundary: main-model or fast-validator text cannot override `missing_evidence`. Weather requires verified weather-bearing provenance (current `web_search` + `browse_url`, a verified weather recipe/API result, or a fresh carried observation). The turn engine performs one built-in weather recipe/fallback recovery and re-runs the gate before finalization. Working-state observations persist compact `fact_types`, `source_tools`, `weather_verified`, `turn_id`, and timestamp metadata so the check does not depend on clipped model-visible excerpts.

## Pipelines and recipes

`run_pipeline` is the preferred composition mechanism for deterministic read-only chains. Intermediate stage values remain in the harness and can be referenced using `$ref`; bounded `foreach`, `$item`, conditional `when`, and optional stages cover common Unix-style map/filter/branch patterns without another model turn. Successful reusable workflows may be saved to the semantic recipe database without changing Python code.

Harness-owned compatibility recipes live in auto-discovered `tools/recipe_provider_groups/` manifests. They are seeded idempotently into the semantic recipe database with `origin=builtin`, a stable key/version, and a target monolithic tool. Every high-level tool in the diagnostic/web/repository compatibility surface must either have a full/partial recipe or an explicit `NATIVE_ONLY` reason; `recipe_coverage()` exposes that matrix. Add new compatibility recipes as provider manifests rather than hard-coding them into the recipe store.

## Dependency direction

- the Web UI depends on `al_agent` application APIs;
- `al_agent` depends on `tools` capabilities/stores;
- tool implementations must not import a frontend;
- provider manifests describe capabilities but do not execute them;
- generic loops (`turn_engine`, worker `runner`) should not contain domain-specific tool/job implementations.

`tests/test_modular_architecture.py` and `scripts/check_architecture.py` enforce the main structural invariants.

## Model roles

The runtime deliberately uses three generative Ollama roles plus one embedding model:

- `agent-main:4b` (`qwen3.5:4b`) is the foreground reasoning, coding, conversation, and tool-orchestration model.
- `agent-fast:2b` (`qwen3.5:2b`) is the bounded auxiliary model for loop validation, recovery planning, research planning, source distillation, and related lightweight reasoning.
- `agent-report:9b` (`qwen3.5:9b`) is admitted only for long-form research synthesis and factuality repair; the worker evicts the normal interactive roles before loading it and restores them afterward.
- `nomic-embed-text` creates semantic vectors for memory, knowledge, and recipe retrieval and does not generate user-facing text.

Deterministic routing, requirements, grounding, safety policy, and exact fast-path renderers remain authoritative and bypass model inference whenever possible. `scripts/benchmark_model_roles.py` measures the live deployment cost of each role so additional model tiers are added only when they demonstrate a net benefit on the target host.

## Multi-fact turn model

The routing layer distinguishes the primary conversational frame from factual completion requirements. `task_frame` remains the single primary compatibility frame, while `fact_frames` contains one scope per requested fact type. Grounding metadata, deterministic recovery, and fact-tool pruning consume the fact-specific frame rather than reusing the primary frame across domains.

`FactGroundingLedger` preserves satisfaction independently for each factual requirement. A failure or retry for one fact does not reopen a previously grounded fact. This is especially important for compound turns such as weather plus news, market quotes plus current time, or other mixed retrieval requests.

The parser detects fact types before clause decomposition and assigns shared/local time, location, and topic modifiers afterward. This avoids naive conjunction splitting and guarantees a frame exists for every detected required fact type. See `MULTI_FACT_GROUNDING_2026-09-22.md` for details.
