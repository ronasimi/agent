# Four-Tier Context Storage

Implemented 2026-09-23 for the 4B local-agent runtime.

## Tier 1 — Hot turn/task state

- Current turn: Python process memory.
- Working state: process-local LRU cache with SQLite/WAL write-through durability.
- Goal: repeated model/tool-loop reads avoid SQLite JSON parsing while state survives process restarts.

## Tier 2 — Conversation continuity

- Raw timestamped `chat_history` remains in SQLite/WAL.
- Recent-history and conversation-summary reads use process-local bounded caches.
- Background rolling compaction is checked after every turn and only runs past the token threshold.
- Compaction advances a watermark; it never deletes the raw transcript.

## Tier 3 — Historical recall

- `chat_history_fts` is an FTS5 external-content index maintained by insert/update/delete triggers.
- B-tree indexes cover global and per-conversation timestamps.
- Relative dates such as `yesterday`, `last night`, `last week`, and weekday names are resolved in the configured timezone, converted to UTC query bounds, then retrieved deterministically.
- Explicit recall is injected into the current request with local timestamps, so the 4B model summarizes evidence instead of deciding how to search for it.
- `search_conversation_history` remains available for explicit/manual searches.

## Tier 4 — Durable memory

- Explicit stable facts remain in the `memory` table.
- `memory_fts` provides indexed lexical retrieval without loading an embedding model.
- Semantic memory remains opt-in and separate.

## Storage performance choices

- SQLite schema initialization is once per DB path, not once per connection.
- SQLite uses WAL and `synchronous=NORMAL`.
- FTS indexes are external-content to avoid a second authoritative copy.
- Historical recall and ordinary durable-memory lookup do not require LLM inference.
- No per-turn summary model call is added; compaction stays low-priority/background.

## Provenance

- Chat history now exposes persisted `created_at` timestamps.
- Historical recall includes both stored UTC and rendered local timestamps.
- Exact effective system/user/tool request messages remain available in `memory/model_calls.jsonl` when tracing is enabled.
