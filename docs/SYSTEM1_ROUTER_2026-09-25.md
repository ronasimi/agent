# System-1 tool router

The foreground harness uses `agent-main:4b` for task execution and a small
`qwen2.5:0.5b` model for initial/fallback tool selection.

## Prompt contract

The router receives no conversation history, no system prompt, and no JSON tool
schemas. A cheap lexical prefilter narrows the catalog to at most eight entries.
The generated prompt contains only compact `ID|name|description` lines and the
current request, capped at 700 characters. The router emits `ID+H/M/L`; `0L`
means no tool.

Each Ollama `generate` call is independent. The router object clears its cached
prompt, request, and candidate list in the foreground turn's `finally` block.
Keeping the model resident (`keep_alive=-1`) retains weights for latency but does
not resend or preserve prior conversation context.

## Persistent calibration

Task outcomes update the existing SQLite routing statistics in
`/app/memory/knowledge.db`. Global/context EMA values survive process/container
restarts and may adjust router confidence by at most +/-0.15. Tool execution
success alone does not train the router; completed task outcomes do. Transport
failures are diagnostic events and do not penalize a capability.

## Cache/prompt discipline

The 4B model receives only the stable discovery controls plus at most one
router-selected full task schema. `tool_search` returns compact candidate
metadata; full schemas appear only in Ollama's native `tools` field. New searches
replace the active task-schema set rather than accumulating schemas.
