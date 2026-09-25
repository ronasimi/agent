# Model Role Hierarchy — 2026-09-24

## Purpose

Al Agent now treats inference as a scarce fallback behind deterministic control flow. The runtime no longer assumes one "main" model should parse, route, execute, validate, recover, and synthesize every turn.

## Roles

| Role | Default model | Context | Responsibility |
| --- | --- | ---: | --- |
| Decision | `agent-micro` (`qwen2.5-coder:0.5b`) | 8192 | Constrained plan decomposition when deterministic parsing cannot resolve a plan, validator classification, confidence-labelled recovery advice |
| Executor | `agent-main` (`qwen2.5-coder:1.5b`) | 16384 | Normal interactive generation, bounded native tool calls, simple argument construction, concise synthesis |
| Reasoning | `agent-reasoning` (`Qwen3.8-4B-Distill Q4_K_M`) | 16384 | Explicit Think turns, complex direct analysis, bounded executor/validator recovery, structured-plan final synthesis |
| Report | `agent-research` (`qwen3.5:9b`) | 8192 | Explicit long-form research/report synthesis and factuality repair |

Vision uses the separate multimodal `qwen3.5:4b` runner because the configured `agent-reasoning` GGUF is text-only.

`model` remains a compatibility alias for the executor. `fast_model` remains a compatibility/support alias for the 1.5B coder executor so older extraction/research helpers are not silently moved onto the 0.5B coder classifier.

## Deterministic-first invariant

The runtime must attempt deterministic work before inference:

1. Parse explicit numbered plans deterministically.
2. Compile typed requirements and narrow tool schemas.
3. Execute deterministic grounding/preflight checks.
4. Close evidence-backed scheduler steps without model narration.
5. Use the executor only when a semantic/tool decision remains.
6. Use the decision model only for constrained classification/validation.
7. Escalate to reasoning only under bounded policy triggers.

The decision model must never be used as the normal user-facing prose generator or as an unconstrained arbitrary tool-argument generator.

## Escalation policy

The 4B reasoning role is selected when one of these bounded conditions applies:

- the user explicitly enables Think;
- a compiled structured plan is in final synthesis;
- a no-tool request is materially complex and matches analysis/debug/architecture/code-reasoning cues;
- executor reasoning recovery is required;
- a zero-tool protocol recovery is required;
- the decision validator reports a low-confidence retry/switch/corrective decision;
- the validator diagnoses `wrong_tool` or `bad_arguments` without high confidence;
- the executor lacks a required capability and a reasoning call remains within budget.

High-confidence ordinary tool decisions stay on the 1.5B coder executor. A validator `blocked` result does not trigger 4B merely to narrate the blocker; the scheduler terminalizes that atomic step and continues.

## Model residency

The intended steady state for `OLLAMA_MAX_LOADED_MODELS=2` is:

- `agent-main` (`qwen2.5-coder:1.5b`) executor resident;
- `agent-micro` (`qwen2.5-coder:0.5b`) decision resident.

On a 4B reasoning escalation, the harness explicitly unloads the decision model first while keeping the executor resident. After the foreground turn releases the inference lock, the decision model is restored asynchronously during an idle window.

The 9B report stage explicitly evicts interactive roles, loads the report model, then restores the executor and schedules decision-model prewarm after report completion.

## Capability behavior

Executor and decision capability probes remain background/idle best-effort operations. Reasoning stays lazy so the first escalation does not pay a separate probe generation before useful work. Runtime capability failures on the executor can request bounded 4B escalation rather than failing the entire turn immediately.

## Configuration

Relevant keys live under `agent` in `config/config.yaml`:

```yaml
executor_model: agent-main
decision_model: agent-micro
reasoning_model: agent-reasoning
vision_model: qwen3.5:4b
report_model: agent-research

model_escalation:
  enabled: true
  complex_direct_reasoning: true
  min_complex_chars: 280
  escalate_on_low_validator_confidence: true
  max_reasoning_calls_per_turn: 4
```

Decision context is 8192 tokens. This is intentionally smaller than the interactive contexts while still accommodating bounded validator transcripts and non-numbered plan compiler payloads.

## Installation and benchmarking

Run:

```bash
./scripts/create_ollama_aliases.sh
```

This creates/pulls the configured executor, decision, and reasoning roles.

Benchmark the hierarchy with:

```bash
python diagnostics/benchmarks/benchmark_model_roles.py --runs 20
```

The benchmark reports executor TTFT, decision-validator latency, reasoning TTFT, report timing, and residency transitions.
