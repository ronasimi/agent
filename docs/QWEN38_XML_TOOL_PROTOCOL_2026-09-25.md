# Qwen3.8 XML tool protocol hardening — 2026-09-25

The configured `empero-ai/Qwen3.8-9B-Distill-GGUF` runner uses the custom Jinja tool grammar supplied by the model. Tool definitions are still passed through Ollama's `tools` request field so the template can serialize them into its `<tools>` block, but tool invocations are XML-style rather than JSON action objects.

## Runtime contract

- Default protocol: `qwen_xml`.
- Default reasoning mode: no-think (`think: false`).
- Context: 32,768 tokens.
- Model tool call grammar: `<tool_call><function=...><parameter=...>...</parameter></function></tool_call>`.
- Tool observations cross the provider boundary as user content wrapped in exact `<tool_response>...</tool_response>` tags.
- Stored history still uses logical assistant/tool roles and correlation IDs; conversion happens only at the Ollama boundary.

The XML parser accepts multiline parameter values, rejects duplicate parameters, rejects malformed/incomplete envelopes, rejects non-whitespace suffix text after the final tool call, and never executes arbitrary JSON or prose as a tool call. Schema-directed coercion converts XML text into integer, number, boolean, array, or object values only where the tool schema requires it; string arguments are preserved as strings.

## No-think behavior

The harness does not paste special tokens into message content. It sends Ollama `think: false`. The model template then emits the required generation prefix immediately after `<|im_start|>assistant\n`:

```text
<think>

</think>

```

The Web UI Think checkbox changes the request to `think: true` for that turn.

## Warmup

Startup warmup is a valid tiny chat turn because this template rejects an empty message list. Warmup uses the same `num_ctx`, system prefix, initial discovery schemas, native `tools` field, and `think: false` setting as a normal first turn. This primes the actual tool-schema/template prefix instead of an approximate plain-text schema summary.

## Hybrid Web UI streaming

Foreground final-answer prose streams to the browser through `assistant_delta` events after a short XML safety guard. Thinking remains a separate opt-in `thinking_delta` stream. The browser uses `RichOutput.streamingMarkdownSnapshot()` to keep incomplete formatted blocks out of the visible answer until they are structurally complete. Fenced blocks, Markdown tables, trailing lists/block quotes, email cards, and specialized weather/status lines are buffered; ordinary conversational prose remains live. `assistant_final` performs the canonical complete Markdown render. If a non-conforming model emits visible prose before a later Qwen XML tool call, `assistant_reset` retracts that provisional prose before tool execution.
