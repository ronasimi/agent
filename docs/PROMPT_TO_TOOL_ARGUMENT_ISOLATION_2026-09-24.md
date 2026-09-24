# Prompt-to-Tool Argument Isolation — 2026-09-24

## Problem

A structured-plan audit step that asked the agent to *identify* the appropriate primitive without executing it was re-parsed as live work. Fact-like examples such as `current weather`, `network routes`, and `repository status` created grounding requirements and executable tool schemas. Separately, whole-prompt lexical target extraction could interpret prose such as `DNS requests` or `Network Path: Use ...` as tool targets.

This allowed surrounding prompt/control text to contaminate tool selection, requirement scopes, and potentially model-emitted selector arguments.

## Fixes

- Added explicit detection for non-executing tool-selection/capability-comparison requests.
- Selection-only steps now create no live fact frames, fact-grounding requirements, or executable requirement ledger entries.
- Selection-only steps receive a compact text-only capability candidate index instead of native executable schemas.
- Structured-plan mode no longer reparses the complete original prompt into a parallel model-facing requirement ledger; the scheduler is authoritative and only the active step ledger is rendered.
- Explicit numbered plan extraction removes following `PHASE` headings from the preceding atomic task.
- DNS/network-path/URL target extraction now rejects descriptive prose and strips Markdown punctuation.
- Added a pre-execution guard that suppresses tool calls when selector-like arguments contain copied structured prompt/control text.

## Validation

- Focused routing/grounding/protocol/scheduler tests: 81 passed.
- Full offline suite: 655 passed, 8 live tests deselected.
