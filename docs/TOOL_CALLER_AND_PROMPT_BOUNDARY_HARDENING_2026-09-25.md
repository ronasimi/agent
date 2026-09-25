# Tool Caller and Prompt-Boundary Hardening — 2026-09-25

## Model roles

- `agent-micro` → `qwen2.5-coder:0.5b`: constrained decision/validator role.
- `agent-main` → `qwen2.5-coder:1.5b`: default executor and native tool caller.
- `agent-reasoning` → `hf.co/empero-ai/Qwen3.8-4B-Distill-GGUF:Q4_K_M`: text reasoning/escalation.
- `qwen3.5:4b`: separate multimodal vision runner because the reasoning GGUF has no vision projector.
- `agent-research` → `qwen3.5:9b`: long-form research/report synthesis.

## Scheduler no-progress invariant

A tool-required structured-plan step may never discard a prose-only model response and retry for free. Each prose-only miss consumes the no-progress budget. The normal sequence is one corrective executor retry, bounded reasoning escalation, optional validator arbitration, then terminalization of only the active step. The scheduler continues with later independent requirements.

Obvious typed route checks compile directly to `route_list` and `route_lookup`; a harmless public-address cross-check uses `8.8.8.8` only as a kernel route-selection target and sends no packet.

## Behavioral tool capability

A provider accepting the `tools` request field is not enough to establish native tool-call support. Capability probe version 2 requires an actual native probe call and continuation. A cached `accepted_unverified` tool mode is rejected for tool-bearing execution and can trigger bounded reasoning fallback.

## Prompt boundary

Working state, evidence, validator state, and scheduler controls are private harness context. Evidence is merged into the leading system context rather than emitted as a synthetic user message. Simple zero-tool conversation omits these blocks entirely. Output leakage detection rejects reproduced internal headings such as `Harness Evidence Digest`, `Harness working state`, and scheduler-control headings.

## Residency

With two Ollama runners, steady state is `agent-main` + `agent-micro`. Reasoning or vision temporarily replaces the decision runner while preserving the executor. The temporary 4B runner is removed before `agent-micro` is restored asynchronously.
