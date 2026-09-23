# Tool, Primitive, and Recipe Soak Testing

> **Current-state note (2026-09-23):** The builtin manifest currently contains 232 tools. This is the current soak-test operator guide; dated audit/review files retain their historical counts.

`scripts/soak_test_tools.py` exercises the live tool registry and compatibility recipe catalog across five deterministic passes by default. It is intended to catch intermittent provider failures, schema drift, hangs, slow primitives, missing host dependencies, recipe regressions, and state contamination between passes.

## Run it in the worker container

The harness primitives intentionally use the production container namespace (`/app`, `/app/workspace`, `/app/memory`, `/host`). Run the audit in the worker container so it sees the same mounts, dependencies, PID namespace, and environment as the agent. The convenience wrapper checks that the worker service is running and then starts the test in `/app`:

```bash
docker compose up -d --build
./scripts/run_soak_test.sh
```

That runs **5 complete passes** with two workers and `--mutating-mode isolated`. You can invoke the Python runner directly if preferred:

```bash
docker compose exec -w /app worker \
  python scripts/soak_test_tools.py --workers 2 --mutating-mode isolated
```

The default five pass profiles are `baseline`, `alternate`, `minimal`, `unicode`, and `boundary`. Each pass gets a fresh fixture directory, fresh job/task/observation/work IDs, and a disposable same-UID process for the process-inspection primitives. Valid probe values rotate across passes, including both TCP and TLS recipe branches. This catches state leakage without making failures non-reproducible.

`--duration` is now optional and defaults to `0` (disabled). `--passes` defaults to `5`. A positive duration can still be supplied as a hard outer deadline, and `--passes 0 --duration 8h` remains available for duration-only stress testing. Per-call child timeouts still prevent one hanging tool from wedging the run.

The default `isolated` mutation mode invokes every read-only tool and every recipe. Mutating primitives that can be safely redirected to the audit database/profile or to disposable workspace fixtures are also invoked. Workspace-bounded deletion (`remove_path`) is tested only against a per-pass disposable directory. System/external-state mutators such as package installation, real reminders/desktop notifications, the legacy live work queue, and generated production tools are schema/argument contract-tested but are not executed.

Use `--mutating-mode all` only inside a disposable test environment. It permits calls that may install packages, schedule OS timers, modify the live queue, generate tools, or trigger other persistent side effects.

## Focused runs

```bash
./scripts/run_soak_test.sh --passes 1
./scripts/run_soak_test.sh --passes 3 --only 'news_search|web_search|browse_url'
./scripts/run_soak_test.sh --passes 2 --only 'compat.*'
./scripts/run_soak_test.sh --passes 0 --duration 8h --exclude 'gmail_*' --exclude 'google_calendar_*'
```

## Resuming

Every result is flushed to `results.jsonl` immediately. To continue an interrupted report directory:

```bash
./scripts/run_soak_test.sh \
  --resume \
  --output-dir /app/workspace/tool_soak_reports/20260921-120000
```

A resumed run retains prior metrics. If the previous pass was interrupted, it first runs only targets that were not recorded in that pass; otherwise it begins the next pass. Pass profiles are deterministic, while mutable state IDs and the process fixture are regenerated so one-shot probes such as job cancellation receive fresh valid state. A malformed/partial trailing JSONL line from an abrupt kill is ignored. Ctrl-C and SIGTERM trigger final summary generation after already-running bounded child probes finish.

## Output

Unless `--output-dir` is supplied, reports are written under `workspace/tool_soak_reports/<timestamp>/` (or `/app/workspace/tool_soak_reports/<timestamp>/` in the container):

- `results.jsonl` — append-only result for every invocation, including arguments, latency, result class, error class, and bounded output preview.
- `summary.json` — complete aggregate metrics for automation.
- `summary.md` — human-readable final/checkpoint report.
- `targets.csv` — per-target success/error/timeout and p50/p95/max latency metrics.
- `inventory.json` — exact tool/recipe inventory and schemas used by the run.
- `isolated_state/` — audit-only SQLite/profile state used to prevent production-memory contamination.

The summary distinguishes `success`, `partial`, `error`, `timeout`, `skipped`, and `contract_error`. Errors are further grouped into credential, dependency, network, not-found, and generic runtime classes. It reports targets that never succeeded, flaky targets (both successes and failures), and the slowest p95 targets. Coverage is measured against the complete selected inventory, so an interrupted run cannot look artificially complete. In addition to the ordinary runtime success rate, `harness_success_rate` excludes probes blocked solely by missing credentials, host dependencies, or network access; those failures remain visible in the raw/error metrics.

## Safety model

Each runtime probe is executed in a fresh Python child process and killed by the controller timeout if it hangs. Database-backed memory, jobs, observations, recipe state, and profile state are redirected to the report directory **before the harness tools package is imported**. Files generated by safe mutation probes are confined to a timestamped `workspace/tool_soak/` directory and removed at the end unless `--keep-output` is used.

Some harness functions intentionally use absolute production paths or external system state. Those remain contract-only in the recommended `isolated` mode and are clearly counted as skipped rather than silently reported as successful.

For CI-style smoke tests, add `--fail-on-errors` to make harness/runtime errors or timeouts produce a non-zero exit status. Credential/dependency/network blocks are excluded from that exit decision by default; add `--fail-on-environment` if CI should fail on those too. Multi-pass runs normally leave these flags off because environmental/provider failures are useful metrics rather than a reason to abort collection.
