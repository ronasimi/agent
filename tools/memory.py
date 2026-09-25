# ==========================================
# FILE: tools/memory.py
# ==========================================
"""Persistent memories and bounded conversation history."""
from __future__ import annotations

import copy
import json
import math
import os
import re
import sqlite3
import threading
import time
import uuid
from collections import OrderedDict
from typing import Any

from .config import load_config
from .runtime import DB_PATH, DB_TIMEOUT, init_runtime_db
from .conversation_context import DEFAULT_CONVERSATION_ID, get_active_conversation_id, normalize_conversation_id

config = load_config()


# Hot-path caches are process-local and keyed by the active database path so
# tests/embedders can safely swap AGENT_DB_PATH. SQLite remains the durable
# source of truth; these caches only avoid repeated reads within one process.
_CACHE_LOCK = threading.RLock()
_SCHEMA_LOCK = threading.RLock()
_INITIALIZED_DB_PATHS: set[str] = set()
_HISTORY_CACHE: OrderedDict[tuple[Any, ...], list[dict[str, Any]]] = OrderedDict()
_CONTEXT_CACHE: OrderedDict[tuple[str, str], tuple[str, int, float]] = OrderedDict()
_CACHE_MAX = 96
_FTS_AVAILABLE: dict[str, bool] = {}
# conversation_context is also updated by the background compaction worker,
# which is a separate process. Keep a tiny TTL so repeated reads inside one
# foreground turn stay in RAM while a later turn observes cross-process writes.
_CONTEXT_CACHE_TTL_SECONDS = 0.25


def _db_key() -> str:
    return os.path.abspath(str(DB_PATH))


def _cache_put(cache: OrderedDict, key: Any, value: Any) -> None:
    with _CACHE_LOCK:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > _CACHE_MAX:
            cache.popitem(last=False)


def _invalidate_history_cache(conversation_id: str | None = None) -> None:
    cid = _conversation_id(conversation_id) if conversation_id is not None else ""
    db = _db_key()
    with _CACHE_LOCK:
        for key in list(_HISTORY_CACHE):
            if key and key[0] == db and (not cid or (len(key) > 1 and key[1] == cid)):
                _HISTORY_CACHE.pop(key, None)


def _invalidate_context_cache(conversation_id: str | None = None) -> None:
    cid = _conversation_id(conversation_id) if conversation_id is not None else ""
    db = _db_key()
    with _CACHE_LOCK:
        for key in list(_CONTEXT_CACHE):
            if key[0] == db and (not cid or key[1] == cid):
                _CONTEXT_CACHE.pop(key, None)


def _connect() -> sqlite3.Connection:
    init_db()
    conn = sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=15000")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _conversation_id(value: str | None = None) -> str:
    return normalize_conversation_id(value or get_active_conversation_id())


def init_db() -> None:
    """Initialize persistent memory/history storage once per database path.

    SQLite WAL + NORMAL synchronous mode is the durable low-latency store. FTS5
    indexes are maintained by triggers when the bundled SQLite supports them.
    """
    db = _db_key()
    with _SCHEMA_LOCK:
        if db in _INITIALIZED_DB_PATHS and os.path.exists(db):
            return
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        with sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT) as conn:
            conn.execute("PRAGMA busy_timeout=15000")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("CREATE TABLE IF NOT EXISTS memory (topic TEXT PRIMARY KEY, fact TEXT, updated_at TEXT DEFAULT CURRENT_TIMESTAMP)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_updated_at ON memory(updated_at DESC)")
            conn.execute("CREATE TABLE IF NOT EXISTS semantic_memory (id INTEGER PRIMARY KEY AUTOINCREMENT, topic TEXT, fact TEXT, embedding TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP)")
            conn.execute("CREATE TABLE IF NOT EXISTS chat_history (id INTEGER PRIMARY KEY AUTOINCREMENT, role TEXT, content TEXT, name TEXT, extra TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP)")
            chat_columns = {row[1] for row in conn.execute("PRAGMA table_info(chat_history)").fetchall()}
            if "conversation_id" not in chat_columns:
                conn.execute("ALTER TABLE chat_history ADD COLUMN conversation_id TEXT NOT NULL DEFAULT 'default'")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_history_conversation_id ON chat_history(conversation_id, id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_history_created_at ON chat_history(created_at DESC, id DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_history_conversation_time ON chat_history(conversation_id, created_at DESC, id DESC)")

            # Keep the legacy singleton table for migration/backward compatibility,
            # but use conversation_context as the canonical per-thread state.
            conn.execute("CREATE TABLE IF NOT EXISTS conversation_state (id INTEGER PRIMARY KEY CHECK(id = 1), summary TEXT NOT NULL DEFAULT '', compacted_through_id INTEGER NOT NULL DEFAULT 0, updated_at TEXT DEFAULT CURRENT_TIMESTAMP)")
            state_columns = {row[1] for row in conn.execute("PRAGMA table_info(conversation_state)").fetchall()}
            if "compacted_through_id" not in state_columns:
                conn.execute("ALTER TABLE conversation_state ADD COLUMN compacted_through_id INTEGER NOT NULL DEFAULT 0")
            conn.execute("INSERT OR IGNORE INTO conversation_state(id, summary) VALUES (1, '')")

            conn.execute(
                """CREATE TABLE IF NOT EXISTS conversations (
                       id TEXT PRIMARY KEY,
                       title TEXT NOT NULL DEFAULT '',
                       created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                       updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                   )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS conversation_context (
                       conversation_id TEXT PRIMARY KEY,
                       summary TEXT NOT NULL DEFAULT '',
                       compacted_through_id INTEGER NOT NULL DEFAULT 0,
                       updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                       FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                   )"""
            )
            conn.execute(
                "INSERT OR IGNORE INTO conversations(id, title) VALUES (?, ?)",
                (DEFAULT_CONVERSATION_ID, "Current conversation"),
            )
            legacy = conn.execute("SELECT summary, compacted_through_id FROM conversation_state WHERE id = 1").fetchone()
            conn.execute(
                "INSERT OR IGNORE INTO conversation_context(conversation_id, summary, compacted_through_id) VALUES (?, ?, ?)",
                (DEFAULT_CONVERSATION_ID, str(legacy[0] or "") if legacy else "", int(legacy[1] or 0) if legacy else 0),
            )

            conn.execute(
                """CREATE TABLE IF NOT EXISTS tool_observations (
                    id TEXT PRIMARY KEY,
                    tool_name TEXT NOT NULL,
                    content TEXT NOT NULL,
                    char_count INTEGER NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )"""
            )
            obs_columns = {row[1] for row in conn.execute("PRAGMA table_info(tool_observations)").fetchall()}
            if "conversation_id" not in obs_columns:
                conn.execute("ALTER TABLE tool_observations ADD COLUMN conversation_id TEXT NOT NULL DEFAULT 'default'")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tool_observations_conversation_id ON tool_observations(conversation_id, created_at)")
            conn.execute("CREATE TABLE IF NOT EXISTS background_tasks (task_name TEXT PRIMARY KEY, status TEXT, output TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)")

            # Historical recall and durable-memory lookup are both lexical hot
            # paths. FTS5 avoids Python table scans and keeps embeddings/model
            # calls out of ordinary recall. External-content indexes add little
            # duplicate storage and stay synchronized through triggers.
            fts_ok = True
            try:
                chat_fts_existed = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='chat_history_fts'"
                ).fetchone() is not None
                memory_fts_existed = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memory_fts'"
                ).fetchone() is not None
                conn.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS chat_history_fts USING fts5(content, role UNINDEXED, conversation_id UNINDEXED, content='chat_history', content_rowid='id', tokenize='unicode61 remove_diacritics 2')"
                )
                conn.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(topic, fact, content='memory', content_rowid='rowid', tokenize='unicode61 remove_diacritics 2')"
                )
                conn.executescript(
                    """
                    CREATE TRIGGER IF NOT EXISTS chat_history_ai AFTER INSERT ON chat_history BEGIN
                      INSERT INTO chat_history_fts(rowid, content, role, conversation_id) VALUES (new.id, new.content, new.role, new.conversation_id);
                    END;
                    CREATE TRIGGER IF NOT EXISTS chat_history_ad AFTER DELETE ON chat_history BEGIN
                      INSERT INTO chat_history_fts(chat_history_fts, rowid, content, role, conversation_id) VALUES ('delete', old.id, old.content, old.role, old.conversation_id);
                    END;
                    CREATE TRIGGER IF NOT EXISTS chat_history_au AFTER UPDATE ON chat_history BEGIN
                      INSERT INTO chat_history_fts(chat_history_fts, rowid, content, role, conversation_id) VALUES ('delete', old.id, old.content, old.role, old.conversation_id);
                      INSERT INTO chat_history_fts(rowid, content, role, conversation_id) VALUES (new.id, new.content, new.role, new.conversation_id);
                    END;
                    CREATE TRIGGER IF NOT EXISTS memory_ai AFTER INSERT ON memory BEGIN
                      INSERT INTO memory_fts(rowid, topic, fact) VALUES (new.rowid, new.topic, new.fact);
                    END;
                    CREATE TRIGGER IF NOT EXISTS memory_ad AFTER DELETE ON memory BEGIN
                      INSERT INTO memory_fts(memory_fts, rowid, topic, fact) VALUES ('delete', old.rowid, old.topic, old.fact);
                    END;
                    CREATE TRIGGER IF NOT EXISTS memory_au AFTER UPDATE ON memory BEGIN
                      INSERT INTO memory_fts(memory_fts, rowid, topic, fact) VALUES ('delete', old.rowid, old.topic, old.fact);
                      INSERT INTO memory_fts(rowid, topic, fact) VALUES (new.rowid, new.topic, new.fact);
                    END;
                    """
                )
                if not chat_fts_existed:
                    conn.execute("INSERT INTO chat_history_fts(chat_history_fts) VALUES('rebuild')")
                if not memory_fts_existed:
                    conn.execute("INSERT INTO memory_fts(memory_fts) VALUES('rebuild')")
            except sqlite3.OperationalError:
                fts_ok = False
            _FTS_AVAILABLE[db] = fts_ok
        _INITIALIZED_DB_PATHS.add(db)
    init_runtime_db()

def _init_chat_db() -> None:
    init_db()


def _init_checkpoint_db() -> None:
    init_runtime_db()


def ensure_conversation(conversation_id: str | None = None, title: str = "") -> str:
    init_db()
    cid = _conversation_id(conversation_id)
    with _connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO conversations(id, title) VALUES (?, ?)",
            (cid, str(title or "").strip()[:120]),
        )
        conn.execute(
            "INSERT OR IGNORE INTO conversation_context(conversation_id) VALUES (?)",
            (cid,),
        )
    return cid


def create_conversation(title: str = "") -> dict[str, Any]:
    cid = uuid.uuid4().hex
    fallback = str(title or "").strip()[:120] or "New conversation"
    ensure_conversation(cid, fallback)
    return {"id": cid, "title": fallback}


def rename_conversation(conversation_id: str, title: str) -> None:
    cid = ensure_conversation(conversation_id)
    with _connect() as conn:
        conn.execute(
            "UPDATE conversations SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (str(title or "").strip()[:120] or "Conversation", cid),
        )


def list_conversations(
    limit: int = 50, *, active_conversation_id: str | None = None
) -> list[dict[str, Any]]:
    init_db()
    limit = max(1, min(int(limit), 200))
    active_id = str(active_conversation_id or "").strip()[:128]
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT c.id, c.title, c.created_at, c.updated_at,
                   (SELECT content FROM chat_history h WHERE h.conversation_id=c.id AND h.role='user' ORDER BY h.id LIMIT 1) AS first_user,
                   (SELECT MAX(id) FROM chat_history h WHERE h.conversation_id=c.id) AS last_message_id,
                   CASE WHEN c.id = ? THEN 0 ELSE 1 END AS active_rank
            FROM conversations c
            ORDER BY active_rank ASC,
                     COALESCE(last_message_id, 0) DESC,
                     c.updated_at DESC,
                     c.created_at DESC,
                     c.id DESC
            LIMIT ?
            """,
            (active_id, limit),
        ).fetchall()
    result = []
    for row in rows:
        title = str(row["title"] or "").strip()
        # Placeholder threads are created eagerly so the next user turn already
        # has a stable conversation_id, but an untouched blank thread is not a
        # saved conversation yet. Keep those placeholders out of the Recent list.
        is_active = bool(active_id and row["id"] == active_id)
        if not is_active and int(row["last_message_id"] or 0) == 0 and (
            row["id"] == DEFAULT_CONVERSATION_ID or title in {"", "New conversation", "Current conversation"}
        ):
            continue
        first = " ".join(str(row["first_user"] or "").split())[:64]
        if not title or title in {"New conversation", "Current conversation"}:
            title = first or title or "Conversation"
        result.append({
            "id": row["id"], "title": title, "created_at": row["created_at"],
            "updated_at": row["updated_at"], "last_message_id": int(row["last_message_id"] or 0),
            "is_active": is_active,
        })
    return result


def delete_conversation(conversation_id: str) -> bool:
    init_db()
    cid = _conversation_id(conversation_id)
    if cid == DEFAULT_CONVERSATION_ID:
        clear_chat_history(cid)
        return True
    with _connect() as conn:
        conn.execute("DELETE FROM chat_history WHERE conversation_id = ?", (cid,))
        conn.execute("DELETE FROM tool_observations WHERE conversation_id = ?", (cid,))
        conn.execute("DELETE FROM conversation_context WHERE conversation_id = ?", (cid,))
        try:
            conn.execute("DELETE FROM working_states WHERE conversation_id = ?", (cid,))
        except sqlite3.Error:
            pass
        cursor = conn.execute("DELETE FROM conversations WHERE id = ?", (cid,))
    _invalidate_history_cache(cid)
    _invalidate_context_cache(cid)
    return bool(cursor.rowcount)


def _save_message_to_db(msg: dict, conversation_id: str | None = None) -> int:
    cid = ensure_conversation(conversation_id)
    role = msg.get("role", "")
    content = msg.get("content", "")
    name = msg.get("name") or msg.get("tool_name")
    extra_data = {}
    for key in ("tool_calls", "tool_call_id", "media", "_runtime"):
        if key in msg:
            extra_data[key] = msg[key]
    extra = json.dumps(extra_data, ensure_ascii=False) if extra_data else None
    with _connect() as conn:
        cursor = conn.execute(
            "INSERT INTO chat_history(role, content, name, extra, conversation_id) VALUES (?, ?, ?, ?, ?)",
            (role, content, name, extra, cid),
        )
        if role == "user":
            first = " ".join(str(content or "").split())[:80]
            if first:
                conn.execute(
                    "UPDATE conversations SET title = CASE WHEN title IN ('', 'New conversation', 'Current conversation') THEN ? ELSE title END, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                    (first, cid),
                )
        else:
            conn.execute("UPDATE conversations SET updated_at=CURRENT_TIMESTAMP WHERE id=?", (cid,))
        message_id = int(cursor.lastrowid)
    _invalidate_history_cache(cid)
    return message_id


def _context_state(conversation_id: str | None = None) -> tuple[str, int]:
    cid = ensure_conversation(conversation_id)
    key = (_db_key(), cid)
    with _CACHE_LOCK:
        cached = _CONTEXT_CACHE.get(key)
        if cached is not None:
            summary, watermark, loaded_at = cached
            if (time.monotonic() - loaded_at) <= _CONTEXT_CACHE_TTL_SECONDS:
                _CONTEXT_CACHE.move_to_end(key)
                return summary, watermark
            _CONTEXT_CACHE.pop(key, None)
    with _connect() as conn:
        row = conn.execute(
            "SELECT summary, compacted_through_id FROM conversation_context WHERE conversation_id = ?", (cid,)
        ).fetchone()
        summary = str(row[0] or "") if row else ""
        watermark = int(row[1] or 0) if row else 0
        if cid == DEFAULT_CONVERSATION_ID:
            legacy = conn.execute("SELECT summary, compacted_through_id FROM conversation_state WHERE id = 1").fetchone()
            if legacy and int(legacy[1] or 0) > watermark:
                watermark = int(legacy[1] or 0)
                if not summary:
                    summary = str(legacy[0] or "")
    value = (summary, watermark, time.monotonic())
    _cache_put(_CONTEXT_CACHE, key, value)
    return summary, watermark


def _load_chat_history_from_db(
    limit: int = 20, *, include_compacted: bool = False, conversation_id: str | None = None
) -> list[dict]:
    """Load bounded timestamped history for one conversation.

    Results are cached in RAM for repeated prompt/UI reads. Raw rows remain in
    SQLite permanently; compaction only advances the prompt-facing watermark.
    """
    cid = ensure_conversation(conversation_id)
    requested_limit = int(limit)
    export_limit = (0 if requested_limit <= 0 else min(requested_limit, 100000)) if include_compacted else max(1, min(requested_limit, 200))
    watermark = 0 if include_compacted else _context_state(cid)[1]
    cache_key = (_db_key(), cid, bool(include_compacted), export_limit, watermark)
    with _CACHE_LOCK:
        cached = _HISTORY_CACHE.get(cache_key)
        if cached is not None:
            _HISTORY_CACHE.move_to_end(cache_key)
            return copy.deepcopy(cached)
    with _connect() as conn:
        if include_compacted:
            if export_limit == 0:
                rows = conn.execute(
                    "SELECT id, role, content, name, extra, created_at FROM chat_history WHERE conversation_id=? ORDER BY id DESC",
                    (cid,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, role, content, name, extra, created_at FROM chat_history WHERE conversation_id=? ORDER BY id DESC LIMIT ?",
                    (cid, export_limit),
                ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, role, content, name, extra, created_at FROM chat_history WHERE conversation_id=? AND id > ? ORDER BY id DESC LIMIT ?",
                (cid, watermark, export_limit),
            ).fetchall()
    rows.reverse()
    result = []
    for message_id, role, content, name, extra, created_at in rows:
        msg = {
            "role": role, "content": content or "", "_db_id": int(message_id),
            "_conversation_id": cid, "_created_at": str(created_at or ""),
        }
        if name:
            msg["name"] = name
        if extra:
            try:
                msg.update(json.loads(extra))
            except json.JSONDecodeError:
                pass
        result.append(msg)
    _cache_put(_HISTORY_CACHE, cache_key, copy.deepcopy(result))
    return result

def clear_chat_history(conversation_id: str | None = None) -> str:
    cid = ensure_conversation(conversation_id)
    with _connect() as conn:
        conn.execute("DELETE FROM chat_history WHERE conversation_id=?", (cid,))
        conn.execute("DELETE FROM tool_observations WHERE conversation_id=?", (cid,))
        # State Tape may be absent in databases created by older revisions.
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='state_tape_entries'").fetchone():
            conn.execute("DELETE FROM state_tape_entries WHERE conversation_id=?", (cid,))
        conn.execute("UPDATE conversation_context SET summary = '', compacted_through_id = 0, updated_at = CURRENT_TIMESTAMP WHERE conversation_id = ?", (cid,))
        if cid == DEFAULT_CONVERSATION_ID:
            conn.execute("UPDATE conversation_state SET summary = '', compacted_through_id = 0, updated_at = CURRENT_TIMESTAMP WHERE id = 1")
    _invalidate_context_cache(cid)
    _invalidate_history_cache(cid)
    try:
        from .working_state import WorkingStateStore
        WorkingStateStore(conversation_id=cid).clear()
    except Exception:
        pass
    return "Conversation history, rolling context summary, and harness working state cleared."


def get_conversation_summary(conversation_id: str | None = None) -> str:
    return _context_state(conversation_id)[0]

def set_conversation_summary(summary: str, conversation_id: str | None = None) -> None:
    cid = ensure_conversation(conversation_id)
    with _connect() as conn:
        conn.execute(
            "INSERT INTO conversation_context(conversation_id, summary, updated_at) VALUES(?, ?, CURRENT_TIMESTAMP) ON CONFLICT(conversation_id) DO UPDATE SET summary=excluded.summary, updated_at=excluded.updated_at",
            (cid, str(summary or "").strip()),
        )
        if cid == DEFAULT_CONVERSATION_ID:
            conn.execute("UPDATE conversation_state SET summary=?, updated_at=CURRENT_TIMESTAMP WHERE id=1", (str(summary or "").strip(),))
    _invalidate_context_cache(cid)


def get_compacted_through_id(conversation_id: str | None = None) -> int:
    return _context_state(conversation_id)[1]

def get_messages_for_compaction(through_id: int, conversation_id: str | None = None) -> list[dict[str, Any]]:
    cid = ensure_conversation(conversation_id)
    through_id = max(0, int(through_id))
    with _connect() as conn:
        state = conn.execute("SELECT compacted_through_id FROM conversation_context WHERE conversation_id = ?", (cid,)).fetchone()
        watermark = int(state[0] or 0) if state else 0
        rows = conn.execute(
            "SELECT id, role, content, name, extra, created_at FROM chat_history WHERE conversation_id=? AND id > ? AND id <= ? ORDER BY id",
            (cid, watermark, through_id),
        ).fetchall()
    messages = []
    for message_id, role, content, name, extra, created_at in rows:
        message: dict[str, Any] = {"_db_id": int(message_id), "role": role, "content": content or "", "_created_at": str(created_at or "")}
        if name:
            message["name"] = name
        if extra:
            try:
                message.update(json.loads(extra))
            except json.JSONDecodeError:
                pass
        messages.append(message)
    return messages


def apply_conversation_compaction(summary: str, through_id: int, conversation_id: str | None = None) -> bool:
    summary = str(summary or "").strip()
    through_id = max(0, int(through_id))
    if not summary or not through_id:
        return False
    cid = ensure_conversation(conversation_id)
    with _connect() as conn:
        cursor = conn.execute(
            """
            UPDATE conversation_context
            SET summary = ?, compacted_through_id = ?, updated_at = CURRENT_TIMESTAMP
            WHERE conversation_id = ? AND compacted_through_id < ?
              AND EXISTS (SELECT 1 FROM chat_history WHERE conversation_id=? AND id = ?)
            """,
            (summary[:8000], through_id, cid, through_id, cid, through_id),
        )
        if cid == DEFAULT_CONVERSATION_ID and cursor.rowcount == 1:
            conn.execute("UPDATE conversation_state SET summary=?, compacted_through_id=?, updated_at=CURRENT_TIMESTAMP WHERE id=1", (summary[:8000], through_id))
    if cursor.rowcount == 1:
        _invalidate_context_cache(cid)
        _invalidate_history_cache(cid)
    return cursor.rowcount == 1


def store_tool_observation(tool_name: str, content: str, conversation_id: str | None = None) -> str:
    cid = ensure_conversation(conversation_id)
    observation_id = uuid.uuid4().hex
    text = str(content)
    with _connect() as conn:
        conn.execute(
            "INSERT INTO tool_observations(id, tool_name, content, char_count, conversation_id) VALUES (?, ?, ?, ?, ?)",
            (observation_id, str(tool_name or "tool"), text, len(text), cid),
        )
    return observation_id


def load_tool_observation_content(observation_id: str = "", conversation_id: str | None = None) -> str:
    """Return the full durable observation content for trusted harness reuse.

    This is an internal fast path used to hydrate deterministic renderers. It is
    deliberately conversation-scoped and should not be exposed as a broad model
    primitive; model-facing reads remain bounded through ``read_observation``.
    """
    observation_id = str(observation_id or "").strip()
    if not observation_id:
        return ""
    cid = ensure_conversation(conversation_id)
    with _connect() as conn:
        row = conn.execute(
            "SELECT content FROM tool_observations WHERE id = ? AND conversation_id = ?",
            (observation_id, cid),
        ).fetchone()
    return str(row[0]) if row else ""


def read_observation(observation_id: str = "", offset: int = 0, length: int = 5000) -> str:
    observation_id = str(observation_id or "").strip()
    if not observation_id:
        return "Error: Missing required 'observation_id' parameter."
    offset = max(0, int(offset))
    length = max(100, min(int(length), 10000))
    cid = ensure_conversation()
    with _connect() as conn:
        row = conn.execute(
            "SELECT tool_name, content, char_count FROM tool_observations WHERE id = ? AND conversation_id = ?",
            (observation_id, cid),
        ).fetchone()
    if not row:
        return f"Observation '{observation_id}' was not found in this conversation."
    tool_name, content, char_count = row
    chunk = str(content)[offset:offset + length]
    return json.dumps(
        {
            "observation_id": observation_id,
            "tool": tool_name,
            "offset": offset,
            "returned_chars": len(chunk),
            "total_chars": int(char_count),
            "has_more": offset + len(chunk) < int(char_count),
            "content": chunk,
        },
        ensure_ascii=False,
        indent=2,
    )

def remember(topic: str = "general_knowledge", fact: str = "Recorded by agent action") -> str:
    """Persist an unchanging user/system fact for later retrieval."""
    topic = str(topic).strip() or "general_knowledge"
    fact = str(fact).strip()
    if not fact:
        return "Error: Missing required 'fact' parameter."
    with _connect() as conn:
        conn.execute(
            "INSERT INTO memory(topic, fact, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP) ON CONFLICT(topic) DO UPDATE SET fact=excluded.fact, updated_at=CURRENT_TIMESTAMP",
            (topic, fact),
        )
    return f"Stored memory under '{topic}'."


_MEMORY_QUERY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "could",
    "did", "do", "does", "earlier", "for", "from", "had", "has", "have", "how", "i",
    "if", "in", "is", "it", "me", "my", "of", "on", "or", "our", "please", "said",
    "tell", "that", "the", "their", "them", "then", "this", "to", "was", "were",
    "what", "when", "where", "which", "who", "why", "with", "would", "you", "your",
}


def _memory_query_terms(query: str) -> list[str]:
    import re
    return [
        token for token in re.findall(r"[a-z0-9][a-z0-9_.:-]*", str(query or "").lower())
        if len(token) >= 3 and token not in _MEMORY_QUERY_STOPWORDS
    ][:24]


def _fts_query(terms: list[str]) -> str:
    clean = []
    for term in terms:
        term = str(term or "").replace('"', '""').strip()
        if term:
            clean.append(f'"{term}"')
    return " OR ".join(clean)


def search_memory(query: str = "", limit: int = 10) -> str:
    """Search durable explicit memories through SQLite FTS5 when available."""
    init_db()
    limit = max(1, min(int(limit), 50))
    raw_query = str(query or "").strip().lower()
    with _connect() as conn:
        if not raw_query or raw_query in {"memory", "all", "everything"}:
            rows = conn.execute(
                "SELECT topic, fact, updated_at FROM memory ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
        else:
            terms = _memory_query_terms(raw_query)
            rows = []
            fts_failed = False
            if terms and _FTS_AVAILABLE.get(_db_key(), False):
                try:
                    rows = conn.execute(
                        """
                        SELECT m.topic, m.fact, m.updated_at
                        FROM memory_fts
                        JOIN memory m ON m.rowid = memory_fts.rowid
                        WHERE memory_fts MATCH ?
                        ORDER BY bm25(memory_fts, 3.0, 1.0), m.updated_at DESC
                        LIMIT ?
                        """,
                        (_fts_query(terms), limit),
                    ).fetchall()
                except sqlite3.OperationalError:
                    fts_failed = True
            if terms and (not _FTS_AVAILABLE.get(_db_key(), False) or fts_failed):
                # Portable fallback for SQLite builds without FTS5. The durable
                # memory table is intentionally small, and this path is bounded.
                candidates = conn.execute(
                    "SELECT topic, fact, updated_at FROM memory ORDER BY updated_at DESC LIMIT 500"
                ).fetchall()
                scored = []
                for row in candidates:
                    topic = str(row[0] or "").lower()
                    fact = str(row[1] or "").lower()
                    score = sum(3 for term in terms if term in topic) + sum(1 for term in terms if term in fact)
                    if score > 0:
                        scored.append((score, str(row[2] or ""), row))
                scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
                rows = [item[2] for item in scored[:limit]]
    return json.dumps(
        [{"topic": row[0], "fact": row[1], "updated_at": row[2]} for row in rows],
        ensure_ascii=False, indent=2,
    ) if rows else "No related memories found."


_HISTORY_QUERY_STOPWORDS = _MEMORY_QUERY_STOPWORDS | {
    "about", "again", "before", "chat", "conversation", "conversations", "covered",
    "discuss", "discussed", "discussion", "history", "last", "night", "recall",
    "remember", "talk", "talked", "talking", "today", "week", "yesterday",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
}


def _history_query_terms(query: str) -> list[str]:
    return [
        token for token in re.findall(r"[a-z0-9][a-z0-9_.:-]*", str(query or "").lower())
        if len(token) >= 3 and token not in _HISTORY_QUERY_STOPWORDS
    ][:24]


def search_conversation_history_records(
    query: str = "", *, start_time: str = "", end_time: str = "",
    conversation_id: str = "", limit: int = 20, before_id: int = 0,
) -> list[dict[str, Any]]:
    """Return timestamped user/assistant rows across saved conversations.

    ``start_time``/``end_time`` use SQLite UTC timestamp form (ISO strings are
    accepted because their leading ``YYYY-MM-DD HH:MM:SS`` sorts identically).
    FTS5 handles lexical recall; timestamp indexes handle date-only recall.
    """
    init_db()
    limit = max(1, min(int(limit), 100))
    terms = _history_query_terms(query)
    clauses = ["h.role IN ('user','assistant')"]
    params: list[Any] = []
    if start_time:
        clauses.append("h.created_at >= ?")
        params.append(str(start_time).replace("T", " ")[:19])
    if end_time:
        clauses.append("h.created_at < ?")
        params.append(str(end_time).replace("T", " ")[:19])
    if conversation_id:
        clauses.append("h.conversation_id = ?")
        params.append(normalize_conversation_id(conversation_id))
    if int(before_id or 0) > 0:
        # Historical recall is built after the current user row is persisted.
        # Excluding that row prevents a generic search from "remembering" the
        # question that asked for the memory instead of the older conversation.
        clauses.append("h.id < ?")
        params.append(int(before_id))
    where = " AND ".join(clauses)
    rows = []
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        fts_failed = False
        if terms and _FTS_AVAILABLE.get(_db_key(), False):
            try:
                rows = conn.execute(
                    f"""
                    SELECT h.id, h.conversation_id, c.title, h.role, h.content, h.created_at,
                           bm25(chat_history_fts) AS rank
                    FROM chat_history_fts
                    JOIN chat_history h ON h.id = chat_history_fts.rowid
                    LEFT JOIN conversations c ON c.id = h.conversation_id
                    WHERE chat_history_fts MATCH ? AND {where}
                    ORDER BY rank ASC, h.created_at DESC, h.id DESC
                    LIMIT ?
                    """,
                    [_fts_query(terms), *params, limit],
                ).fetchall()
            except sqlite3.OperationalError:
                fts_failed = True
        if terms and (not _FTS_AVAILABLE.get(_db_key(), False) or fts_failed):
            like_clauses = []
            like_params: list[Any] = []
            for term in terms:
                like_clauses.append("LOWER(h.content) LIKE ?")
                like_params.append(f"%{term}%")
            rows = conn.execute(
                f"""
                SELECT h.id, h.conversation_id, c.title, h.role, h.content, h.created_at, 0.0 AS rank
                FROM chat_history h LEFT JOIN conversations c ON c.id=h.conversation_id
                WHERE {where} AND ({' OR '.join(like_clauses)})
                ORDER BY h.created_at DESC, h.id DESC LIMIT ?
                """,
                [*params, *like_params, limit],
            ).fetchall()
        elif not terms:
            rows = conn.execute(
                f"""
                SELECT h.id, h.conversation_id, c.title, h.role, h.content, h.created_at, 0.0 AS rank
                FROM chat_history h LEFT JOIN conversations c ON c.id=h.conversation_id
                WHERE {where}
                ORDER BY h.created_at DESC, h.id DESC LIMIT ?
                """,
                [*params, limit],
            ).fetchall()
    # Present selected evidence chronologically so the small model can reconstruct
    # the conversation without having to undo relevance/recency ordering itself.
    result = [
        {
            "id": int(row["id"]), "conversation_id": str(row["conversation_id"] or ""),
            "title": str(row["title"] or "Conversation"), "role": str(row["role"] or ""),
            "content": str(row["content"] or ""), "created_at": str(row["created_at"] or ""),
        }
        for row in rows
    ]
    result.sort(key=lambda item: (item["created_at"], item["id"]))
    return result


def search_conversation_history(
    query: str = "", start_time: str = "", end_time: str = "",
    conversation_id: str = "", limit: int = 20,
) -> str:
    """Search timestamped saved chat history across conversations."""
    rows = search_conversation_history_records(
        query, start_time=start_time, end_time=end_time, conversation_id=conversation_id, limit=limit,
    )
    return json.dumps(rows, ensure_ascii=False, indent=2) if rows else "No matching conversation history found."

def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    return dot / (norm1 * norm2) if norm1 and norm2 else 0.0


def remember_semantic(topic: str = "general_knowledge", fact: str = "") -> str:
    """Compatibility alias: store a fact in lexical memory without another model."""
    return remember(topic=topic, fact=fact)


def search_semantic_memory(query: str = "", limit: int = 5) -> str:
    """Compatibility alias: search persistent memory without embedding inference."""
    return search_memory(query=query, limit=limit)


def get_relevant_memories(query: str, limit: int = 8) -> list[dict[str, Any]]:
    """Fetch only memories likely relevant to the current turn."""
    try:
        text = search_semantic_memory(query, limit=limit)
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        try:
            text = search_memory(query, limit=limit)
            parsed = json.loads(text)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []


def get_all_memories_prompt_summary() -> str:
    """Compatibility helper; returns a short notice rather than dumping the entire memory table."""
    return "\n\n### Memory Policy\nRelevant long-term memories are retrieved per task; the complete memory store is not injected into context."


def start_background_task(task_name: str = "", python_code: str = "", timeout: int = 3600) -> str:
    """Deprecated: arbitrary Python is not launched as a detached thread."""
    return "Background Python execution is disabled by the durable runtime. Use queue_work() for explicit work items."


def check_background_task(task_name: str = "") -> str:
    """Return legacy background task state if one exists."""
    if not task_name:
        return "Error: Missing required 'task_name' parameter."
    with _connect() as conn:
        row = conn.execute("SELECT status, output, timestamp FROM background_tasks WHERE task_name = ?", (task_name,)).fetchone()
    return json.dumps({"task_name": task_name, "status": row[0], "output": row[1], "timestamp": row[2]}) if row else f"No legacy background task named '{task_name}'."


init_db()
