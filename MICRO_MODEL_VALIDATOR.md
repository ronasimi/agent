# 0.8B micro-model completion gate

The harness uses `agent-micro:0.8b` for one narrow question inside the existing
tool-loop validator path:

```text
deterministic requirements + grounding
  -> agent-micro:0.8b: {"complete": true|false}
      -> true: finish immediately
      -> false / invalid / error: agent-fast:2b validator
```

There is no separate classifier service and no pre-turn micro-model routing.
Normal deterministic intent/tool selection stays exactly as before, so ordinary
conversation does not pay an extra model call.

The micro model cannot select tools, grant mutating authority, declare a task
blocked, override an explicit no-tools policy, or satisfy factual grounding. It
is consulted only when deterministic requirements are already satisfied. Every
non-complete result falls through to the 2B model because recovery/blocking can
require semantic reasoning and exact tool selection.

## Model roles

- `agent-micro:0.8b`: one-bit completion gate
- `agent-fast:2b`: recovery validation, research planning/distillation
- `agent-main:4b`: interactive reasoning, tool orchestration, final answers
- `agent-report:9b`: long-form research synthesis/factuality repair
- `nomic-embed-text`: semantic memory/recipe embeddings

## Aliases

```bash
./scripts/create_ollama_aliases.sh
```

The helper creates:

```text
agent-micro:0.8b -> qwen3.5:0.8b
agent-fast:2b    -> qwen3.5:2b
agent-main:4b    -> qwen3.5:4b
agent-report:9b  -> qwen3.5:9b
```

## Benchmark

Compare 0.8B and 2B on the same constrained completion schema:

```bash
python scripts/benchmark_micro_model.py --runs 20 --compare-fast
```

The script bootstraps the repository path itself and does not initialize the
harness SQLite database.
