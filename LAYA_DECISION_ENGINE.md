# Laya decision-engine prototype

This prototype adds an optional non-autoregressive decision layer in front of
ambiguous agent turns. It is deliberately **not** an authority layer: exact
slash commands, deterministic intent routes, mutation policy, required tools,
and factual grounding remain harness-owned.

## Decision hierarchy

```text
user turn
  -> deterministic command / exact fast path
  -> Laya sidecar (one batched encoder pass)
  -> constrained 2B route classifier when Laya is unavailable/low-confidence
  -> normal 4B main-model/tool loop

post-tool loop
  -> deterministic requirements + grounding
  -> high-confidence Laya terminal validator (finish / blocked only)
  -> existing 2B validator for recovery or low-confidence cases
```

The Laya route batch asks six small categorical questions in one pass:

- broad capability family
- whether a tool is required
- freshness requirement
- continuation vs new task
- direct-render vs model-synthesis preference
- read-only/mutation risk

Only the first three routing signals currently influence schema selection and
continuation. Other signals are captured for measurement/training; mutation
risk can never grant mutation authority.

## Why a sidecar

CLI, Web UI, and background worker are separate processes. Loading Laya inside
each would keep multiple encoder copies in RAM. `decision-engine` is therefore
a single localhost-only FastAPI service bound to `127.0.0.1:8091` and shared by
all frontends.

The sidecar uses the upstream `convaiinnovations/laya` model with the
`typed-decisions` subfolder on CPU. The checkpoint cache is persisted under
`memory/huggingface/`.

If the service is unavailable, cold, times out, or returns a low-confidence
answer, the harness fails open to its existing deterministic/2B path. It does
not make the turn fail.

## Confidence gates

Prototype thresholds are intentionally conservative in `config/config.yaml`:

```yaml
decision_engine:
  routing:
    fallback_to_fast_model: true
    thresholds:
      route_family: 0.92
      tool_requirement: 0.97
      continuation: 0.95
      freshness: 0.95
      renderer: 0.95
      risk: 0.98
  validator:
    min_confidence: 0.95
```

Do not lower these from public benchmark numbers alone. The stock checkpoint
is not trained on this harness's labels, and the upstream project explicitly
recommends domain specialization and local calibration for typed-decision
workflows.

## Security boundaries

Laya is advisory. In particular it cannot:

- add `execute_shell`, `execute_python`, file writes, or other mutating tools;
- override a user `no tools` constraint;
- satisfy a grounding requirement;
- bypass schema validation or tool-policy checks;
- authorize a side effect;
- invent a tool name or arguments.

At most, a high-confidence route narrows existing schemas or adds a small
read-only starter set for a family. Deterministic requirements and explicit
user tool names are always preserved.

## Validator use

The decision sidecar may bypass the 2B validator only for a high-confidence
terminal `finish` or `blocked` classification. A `recover` classification
always falls through to the 2B validator, because exact recovery-tool choice is
better handled by the existing structured validator path.

## `/research` memory handoff

Before the 9B report writer is loaded, the research worker requests
`POST /unload` on the sidecar, then evicts the normal 4B/2B Ollama models. On
report completion it restores the interactive Ollama models and requests a
Laya preload again.

The sidecar serializes predictions and cancels an in-flight cold-load
publication when an unload arrives, preventing a delayed encoder load from
reappearing while the report writer is resident.

## Training capture

Every Laya route prediction is appended to:

```text
memory/laya_training.jsonl
```

When the turn completes, the harness writes a supervision record for the same
trace. Successful tool families are strong labels; no-tool turns intentionally
do not receive a synthetic `conversation` family label because that would
mislabel coding/explanation tasks.

Export joined examples:

```bash
python scripts/export_laya_training.py
```

Evaluate confidence gates against your own captured outcomes:

```bash
python scripts/evaluate_laya_capture.py
```

The evaluator reports empirical precision and coverage at several confidence
thresholds. Treat the suggested threshold only as a deployment aid; collect a
reasonable number of representative turns before using it.

## Benchmarking

Once the sidecar has downloaded its checkpoint:

```bash
python scripts/benchmark_laya.py --runs 20 --compare-fast
```

The benchmark reports cold/preload time and median/p95 routing latency across a
small mixed prompt set; `--compare-fast` also measures the configured 2B Ollama
routing fallback and reports median milliseconds saved by Laya. The useful
comparison on the deployment host is not
just encoder latency: compare total turn TTFT, prompt-eval tokens, validator
calls, and the fraction of turns that avoid the 2B validator.

## Compose lifecycle

Build/start normally:

```bash
docker compose --profile web up -d --build
```

Inspect the sidecar:

```bash
docker compose logs -f decision-engine
curl http://127.0.0.1:8091/health
```

The Web UI `/api/health` endpoint also reports sidecar readiness.
