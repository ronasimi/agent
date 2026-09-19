# Al Agent architecture

The repository uses **thin composition roots + domain modules + declarative providers**.  The goal is that adding a capability is normally additive: create a focused implementation and register a small manifest, rather than editing the agent loop.

## Runtime layers

```text
agent.py / worker.py                    compatibility entry points
        │
        ├── al_agent/                   application orchestration
        │   ├── state.py                configuration + long-lived services
        │   ├── events.py               frontend event bridge + inference lock
        │   ├── prompts.py              stable policy, memory, media encoding
        │   ├── turn_support.py         deterministic loop helpers
        │   ├── turn_engine.py          one responsibility: model/tool state machine
        │   ├── cli.py                  terminal frontend
        │   ├── cli_commands.py         registered slash-command handlers
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
        │   └── ...                     focused domain tools/stores
        │
        └── webui/                      optional browser frontend
            ├── server.py               FastAPI route composition root
            ├── chat.py                 WebSocket turn streaming
            ├── workspace_ops.py        workspace/artifact/upload operations
            ├── theme.py                Xresources palette loading
            ├── history.py              browser history serialization
            └── static/                 dependency-free SPA
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

## Adding a CLI command

Add a `CliCommand` to `al_agent/cli_commands.py`.  Command behavior stays outside the prompt loop.

## Adding a Web UI capability

Keep HTTP route composition in `webui/server.py`, but put domain behavior in a sibling module (`workspace_ops`, `chat`, `theme`, etc.).  The browser must not bypass the main agent tool/policy loop.

## Pipelines and recipes

`run_pipeline` is the preferred composition mechanism for deterministic read-only chains.  Intermediate stage values remain in the harness and can be referenced using `$ref`.  Successful reusable workflows may be saved to the semantic recipe database without changing Python code.

## Dependency direction

- frontends depend on `al_agent` application APIs;
- `al_agent` depends on `tools` capabilities/stores;
- tool implementations must not import a frontend;
- provider manifests describe capabilities but do not execute them;
- generic loops (`turn_engine`, worker `runner`) should not contain domain-specific tool/job implementations.

`tests/test_modular_architecture.py` and `scripts/check_architecture.py` enforce the main structural invariants.
