# Reasoning-only completion recovery

## Failure reproduced

The interactive loop treated an Ollama response as empty whenever it contained
neither `message.content` nor a tool call. With the configured
`empero-ai/Qwen3.8-*-Distill-GGUF` roles, a successful response can instead end
with only `message.thinking` populated. Two such responses exhausted the
bounded no-progress guard and marked even a trivial turn such as `Hello?` as
blocked.

The browser stress prompt is especially exposed because ordinary tool-selection
turns are capped at 384 generated tokens. A reasoning model can spend that whole
budget before it reaches the final tool-call envelope.

## Fix

- Classify `thinking != "" && content == "" && tool_calls == []` as a dedicated
  `thinking_only_response` condition rather than a generic empty response.
- Never surface or feed the hidden reasoning text back into the model.
- Retry that condition once, still inside the existing two-call no-progress
  bound.
- On the retry, use the configured short-reasoning mode (`low` by default) and a
  larger generation allowance: 1024 tokens for tool-selection turns and 2048
  for final-answer turns.
- Emit structured `model_no_progress` and `reasoning_recovery` events containing
  only metadata such as `done_reason`, `eval_count`, and `num_predict`.
- If the retry also produces reasoning only, stop deterministically and report
  the specific failure class.
- Add an opt-in live Ollama conformance test that verifies the configured
  recovery mode can actually produce non-empty `message.content`.

## Validation

Targeted and adjacent harness tests pass with an Ollama import stub:

- `tests/test_model_no_progress_budget.py`
- `tests/test_model_protocol.py`
- `tests/test_browser_p2.py`
- `tests/test_control_loop_completion.py`
- `tests/test_harness_regressions.py`
- `tests/test_qa_stress_edges.py`
- `tests/test_browser_ui.py`
- `tests/test_latency_and_prefix.py`

77 tests passed in the broader regression subset. The full suite could not be
collected in the analysis sandbox because the optional `ddgs` dependency is not
installed there.
