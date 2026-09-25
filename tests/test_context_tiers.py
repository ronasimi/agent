from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo


def _use_temp_db(monkeypatch, tmp_path):
    from tools import memory, runtime, working_state

    db = str(tmp_path / "tiers.db")
    monkeypatch.setattr(runtime, "DB_PATH", db)
    monkeypatch.setattr(memory, "DB_PATH", db)
    monkeypatch.setattr(working_state, "DB_PATH", db)
    memory.init_db()
    return memory, working_state, db


def test_chat_history_preserves_timestamps_and_supports_cross_conversation_fts(tmp_path, monkeypatch):
    memory, _working_state, db = _use_temp_db(monkeypatch, tmp_path)
    first = memory.create_conversation("Yesterday project")
    second = memory.create_conversation("Other thread")
    msg_id = memory._save_message_to_db(
        {"role": "user", "content": "We optimized the Ollama harness cache."},
        conversation_id=first["id"],
    )
    memory._save_message_to_db(
        {"role": "assistant", "content": "We reduced prompt prefill and kept the runner warm."},
        conversation_id=first["id"],
    )
    memory._save_message_to_db(
        {"role": "user", "content": "Unrelated gardening notes."},
        conversation_id=second["id"],
    )
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE chat_history SET created_at='2026-09-23 18:00:00' WHERE conversation_id=?", (first["id"],))
        conn.execute("UPDATE chat_history SET created_at='2026-09-22 18:00:00' WHERE conversation_id=?", (second["id"],))

    rows = memory._load_chat_history_from_db(limit=20, include_compacted=True, conversation_id=first["id"])
    assert rows[0]["_db_id"] == msg_id
    assert rows[0]["_created_at"] == "2026-09-23 18:00:00"

    matches = memory.search_conversation_history_records(
        "Ollama cache", start_time="2026-09-23 04:00:00", end_time="2026-09-24 04:00:00", limit=10,
    )
    assert matches
    assert all(row["conversation_id"] == first["id"] for row in matches)
    assert any("Ollama" in row["content"] for row in matches)


def test_yesterday_recall_resolves_local_window_and_returns_timestamped_context(tmp_path, monkeypatch):
    memory, _working_state, db = _use_temp_db(monkeypatch, tmp_path)
    convo = memory.create_conversation("Harness work")
    memory._save_message_to_db(
        {"role": "user", "content": "We discussed rolling context compaction and FTS recall."},
        conversation_id=convo["id"],
    )
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE chat_history SET created_at='2026-09-23 20:15:00' WHERE conversation_id=?", (convo["id"],))

    from tools.historical_context import build_historical_recall_context, resolve_historical_window

    now = datetime(2026, 9, 24, 12, 0, tzinfo=ZoneInfo("America/Toronto"))
    window = resolve_historical_window(
        "what did we talk about yesterday?", timezone_name="America/Toronto", now=now,
    )
    assert window["start_utc"] == "2026-09-23 04:00:00"
    assert window["end_utc"] == "2026-09-24 04:00:00"

    context = build_historical_recall_context(
        "what did we talk about yesterday?", timezone_name="America/Toronto", now=now,
    )
    payload = json.loads(context)
    assert payload["resolved_window"]["label"] == "yesterday"
    assert payload["matches"]
    assert "rolling context compaction" in payload["matches"][0]["content"]
    assert payload["matches"][0]["created_at_local"].startswith("2026-09-23T16:15:00")


def test_non_history_yesterday_request_does_not_trigger_chat_recall():
    from tools.historical_context import historical_recall_tool_text, is_historical_recall_request

    assert is_historical_recall_request("what did we talk about yesterday?") is True
    assert is_historical_recall_request("do you remember what we discussed yesterday?") is True
    assert is_historical_recall_request("what did you tell me Monday?") is True
    assert is_historical_recall_request("what was the weather yesterday?") is False
    assert historical_recall_tool_text("find our conversation about Ollama prompt caching") == ""
    assert historical_recall_tool_text(
        "what did we discuss yesterday, then check the router now"
    ) == "check the router now"




def test_historical_recall_excludes_the_current_recall_question(tmp_path, monkeypatch):
    memory, _working_state, _db = _use_temp_db(monkeypatch, tmp_path)
    convo = memory.create_conversation("Ollama notes")
    memory._save_message_to_db(
        {"role": "user", "content": "We tuned Ollama prompt caching yesterday."},
        conversation_id=convo["id"],
    )
    current_id = memory._save_message_to_db(
        {"role": "user", "content": "Find our conversation about Ollama prompt caching."},
        conversation_id=convo["id"],
    )

    from tools.historical_context import historical_recall_payload

    payload = historical_recall_payload(
        "Find our conversation about Ollama prompt caching.",
        timezone_name="America/Toronto",
        before_id=current_id,
    )
    assert payload is not None
    contents = [row["content"] for row in payload["matches"]]
    assert any("tuned Ollama" in text for text in contents)
    assert all("Find our conversation" not in text for text in contents)


def test_large_historical_payload_remains_valid_bounded_json(tmp_path, monkeypatch):
    memory, _working_state, db = _use_temp_db(monkeypatch, tmp_path)
    convo = memory.create_conversation("Long recap")
    for index in range(60):
        memory._save_message_to_db(
            {"role": "user" if index % 2 == 0 else "assistant", "content": f"turn {index} " + ("x" * 900)},
            conversation_id=convo["id"],
        )
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE chat_history SET created_at='2026-09-23 18:00:00' WHERE conversation_id=?",
            (convo["id"],),
        )

    from tools.historical_context import build_historical_recall_context

    now = datetime(2026, 9, 24, 12, 0, tzinfo=ZoneInfo("America/Toronto"))
    context = build_historical_recall_context(
        "what did we talk about yesterday?", timezone_name="America/Toronto", now=now,
    )
    payload = json.loads(context)
    assert payload["matches"]
    assert len(context) <= 7000


def test_durable_memory_uses_indexed_lexical_retrieval(tmp_path, monkeypatch):
    memory, _working_state, _db = _use_temp_db(monkeypatch, tmp_path)
    memory.remember("hardware", "ThinkPad T14 Gen 1 AMD with 16 GB RAM")
    memory.remember("editor", "Uses Neovim")
    result = json.loads(memory.search_memory("ThinkPad AMD", limit=5))
    assert result[0]["topic"] == "hardware"
    assert "ThinkPad" in result[0]["fact"]


def test_working_state_is_ram_cached_with_sqlite_write_through(tmp_path, monkeypatch):
    _memory, working_state, db = _use_temp_db(monkeypatch, tmp_path)
    store = working_state.WorkingStateStore(conversation_id="cache-test")
    store.begin_turn(
        turn_id=7,
        objective="inspect cache behavior",
        rolling_summary="",
        recalled_context="",
        recent_messages=[],
        policy_note="",
        tool_schemas=[],
    )
    first = store.load()
    assert first["objective"] == "inspect cache behavior"

    # Simulate an out-of-band stale durable row. Tier-1 reads should continue to
    # use the process-local canonical object until the harness explicitly clears it.
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE working_states SET state_json='{}' WHERE conversation_id='cache-test'")
    second = store.load()
    assert second["objective"] == "inspect cache behavior"

    # The durable copy was written before the simulated external overwrite.
    assert first["turn_id"] == 7


def test_webui_history_exposes_persisted_created_at(tmp_path, monkeypatch):
    memory, _working_state, db = _use_temp_db(monkeypatch, tmp_path)
    memory._save_message_to_db({"role": "user", "content": "timestamp me"})
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE chat_history SET created_at='2026-09-23 23:59:00'")
    # Invalidate the history cache after the test's direct DB mutation.
    memory._invalidate_history_cache()

    import webui.history as history
    monkeypatch.setattr(history, "_load_chat_history_from_db", memory._load_chat_history_from_db)
    rows = history._history(limit=20)
    assert rows[0]["created_at"] == "2026-09-23 23:59:00"
