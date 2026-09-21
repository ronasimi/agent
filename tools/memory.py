# ==========================================
# FILE: tools/memory.py
# ==========================================
"""Persistent memories and bounded conversation history."""
from __future__ import annotations

import json
import math
import os
import sqlite3
import uuid
from typing import Any

from .config import load_config
from .runtime import DB_PATH, DB_TIMEOUT, init_runtime_db
from .conversation_context import DEFAULT_CONVERSATION_ID, get_active_conversation_id, normalize_conversation_id

config = load_config()
EMBED_MODEL = config.get("agent", {}).get("embed_model", "nomic-embed-text")


def _connect() -> sqlite3.Connection:
    init_db()
    conn = sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=15000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _conversation_id(value: str | None = None) -> str:
    return normalize_conversation_id(value or get_active_conversation_id())


def init_db() -> None:
    """Initialize persistent memory, conversation, and runtime storage."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT) as conn:
        conn.execute("PRAGMA busy_timeout=15000")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("CREATE TABLE IF NOT EXISTS memory (topic TEXT PRIMARY KEY, fact TEXT, updated_at TEXT DEFAULT CURRENT_TIMESTAMP)")
        conn.execute("CREATE TABLE IF NOT EXISTS semantic_memory (id INTEGER PRIMARY KEY AUTOINCREMENT, topic TEXT, fact TEXT, embedding TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP)")
        conn.execute("CREATE TABLE IF NOT EXISTS chat_history (id INTEGER PRIMARY KEY AUTOINCREMENT, role TEXT, content TEXT, name TEXT, extra TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP)")
        chat_columns = {row[1] for row in conn.execute("PRAGMA table_info(chat_history)").fetchall()}
        if "conversation_id" not in chat_columns:
            conn.execute("ALTER TABLE chat_history ADD COLUMN conversation_id TEXT NOT NULL DEFAULT 'default'")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_history_conversation_id ON chat_history(conversation_id, id)")

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
    init_runtime_db()


def _init_chat_db() -> None:
    init_db()


def _init_checkpoint_db() -> None:
    init_runtime_db()


def ensure_conversation(conversation_id: str | None = None, title: str = "") -> str:
    init_db()
    cid = _conversation_id(conversation_id)
    with sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT) as conn:
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
    with sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT) as conn:
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
    with sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT) as conn:
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
    with sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT) as conn:
        conn.execute("DELETE FROM chat_history WHERE conversation_id = ?", (cid,))
        conn.execute("DELETE FROM tool_observations WHERE conversation_id = ?", (cid,))
        conn.execute("DELETE FROM conversation_context WHERE conversation_id = ?", (cid,))
        try:
            conn.execute("DELETE FROM working_states WHERE conversation_id = ?", (cid,))
        except sqlite3.Error:
            pass
        cursor = conn.execute("DELETE FROM conversations WHERE id = ?", (cid,))
    return bool(cursor.rowcount)


def _save_message_to_db(msg: dict, conversation_id: str | None = None) -> int:
    cid = ensure_conversation(conversation_id)
    role = msg.get("role", "")
    content = msg.get("content", "")
    name = msg.get("name") or msg.get("tool_name")
    extra_data = {}
    for key in ("tool_calls", "tool_call_id", "media"):
        if key in msg:
            extra_data[key] = msg[key]
    extra = json.dumps(extra_data, ensure_ascii=False) if extra_data else None
    with sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT) as conn:
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
        return int(cursor.lastrowid)


def _load_chat_history_from_db(
    limit: int = 20, *, include_compacted: bool = False, conversation_id: str | None = None
) -> list[dict]:
    """Load bounded history for one conversation, or complete rows for export."""
    cid = ensure_conversation(conversation_id)
    requested_limit = int(limit)
    export_limit = (0 if requested_limit <= 0 else min(requested_limit, 100000)) if include_compacted else max(1, min(requested_limit, 200))
    with sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT) as conn:
        if include_compacted:
            if export_limit == 0:
                rows = conn.execute(
                    "SELECT id, role, content, name, extra FROM chat_history WHERE conversation_id=? ORDER BY id DESC",
                    (cid,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, role, content, name, extra FROM chat_history WHERE conversation_id=? ORDER BY id DESC LIMIT ?",
                    (cid, export_limit),
                ).fetchall()
        else:
            row = conn.execute("SELECT compacted_through_id FROM conversation_context WHERE conversation_id = ?", (cid,)).fetchone()
            watermark = int(row[0] or 0) if row else 0
            if cid == DEFAULT_CONVERSATION_ID:
                # Honor legacy callers that still update the singleton state
                # directly during migration; canonical writes keep both in sync.
                legacy = conn.execute("SELECT compacted_through_id FROM conversation_state WHERE id = 1").fetchone()
                watermark = max(watermark, int(legacy[0] or 0) if legacy else 0)
            rows = conn.execute(
                "SELECT id, role, content, name, extra FROM chat_history WHERE conversation_id=? AND id > ? ORDER BY id DESC LIMIT ?",
                (cid, watermark, export_limit),
            ).fetchall()
    rows.reverse()
    result = []
    for message_id, role, content, name, extra in rows:
        msg = {"role": role, "content": content or "", "_db_id": int(message_id), "_conversation_id": cid}
        if name:
            msg["name"] = name
        if extra:
            try:
                msg.update(json.loads(extra))
            except json.JSONDecodeError:
                pass
        result.append(msg)
    return result


def clear_chat_history(conversation_id: str | None = None) -> str:
    cid = ensure_conversation(conversation_id)
    with sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT) as conn:
        conn.execute("DELETE FROM chat_history WHERE conversation_id=?", (cid,))
        conn.execute("DELETE FROM tool_observations WHERE conversation_id=?", (cid,))
        conn.execute("UPDATE conversation_context SET summary = '', compacted_through_id = 0, updated_at = CURRENT_TIMESTAMP WHERE conversation_id = ?", (cid,))
        if cid == DEFAULT_CONVERSATION_ID:
            conn.execute("UPDATE conversation_state SET summary = '', compacted_through_id = 0, updated_at = CURRENT_TIMESTAMP WHERE id = 1")
    try:
        from .working_state import WorkingStateStore
        WorkingStateStore(conversation_id=cid).clear()
    except Exception:
        pass
    return "Conversation history, rolling context summary, and harness working state cleared."


def get_conversation_summary(conversation_id: str | None = None) -> str:
    cid = ensure_conversation(conversation_id)
    with sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT) as conn:
        row = conn.execute("SELECT summary FROM conversation_context WHERE conversation_id = ?", (cid,)).fetchone()
    return row[0] if row else ""


def set_conversation_summary(summary: str, conversation_id: str | None = None) -> None:
    cid = ensure_conversation(conversation_id)
    with sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT) as conn:
        conn.execute(
            "INSERT INTO conversation_context(conversation_id, summary, updated_at) VALUES(?, ?, CURRENT_TIMESTAMP) ON CONFLICT(conversation_id) DO UPDATE SET summary=excluded.summary, updated_at=excluded.updated_at",
            (cid, str(summary or "").strip()),
        )
        if cid == DEFAULT_CONVERSATION_ID:
            conn.execute("UPDATE conversation_state SET summary=?, updated_at=CURRENT_TIMESTAMP WHERE id=1", (str(summary or "").strip(),))


def get_compacted_through_id(conversation_id: str | None = None) -> int:
    cid = ensure_conversation(conversation_id)
    with sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT) as conn:
        row = conn.execute("SELECT compacted_through_id FROM conversation_context WHERE conversation_id = ?", (cid,)).fetchone()
    return int(row[0] or 0) if row else 0


def get_messages_for_compaction(through_id: int, conversation_id: str | None = None) -> list[dict[str, Any]]:
    cid = ensure_conversation(conversation_id)
    through_id = max(0, int(through_id))
    with sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT) as conn:
        state = conn.execute("SELECT compacted_through_id FROM conversation_context WHERE conversation_id = ?", (cid,)).fetchone()
        watermark = int(state[0] or 0) if state else 0
        rows = conn.execute(
            "SELECT id, role, content, name, extra FROM chat_history WHERE conversation_id=? AND id > ? AND id <= ? ORDER BY id",
            (cid, watermark, through_id),
        ).fetchall()
    messages = []
    for message_id, role, content, name, extra in rows:
        message: dict[str, Any] = {"_db_id": int(message_id), "role": role, "content": content or ""}
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
    with sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT) as conn:
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
    return cursor.rowcount == 1


def store_tool_observation(tool_name: str, content: str, conversation_id: str | None = None) -> str:
    cid = ensure_conversation(conversation_id)
    observation_id = uuid.uuid4().hex
    text = str(content)
    with sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT) as conn:
        conn.execute(
            "INSERT INTO tool_observations(id, tool_name, content, char_count, conversation_id) VALUES (?, ?, ?, ?, ?)",
            (observation_id, str(tool_name or "tool"), text, len(text), cid),
        )
    return observation_id


def read_observation(observation_id: str = "", offset: int = 0, length: int = 5000) -> str:
    observation_id = str(observation_id or "").strip()
    if not observation_id:
        return "Error: Missing required 'observation_id' parameter."
    offset = max(0, int(offset))
    length = max(100, min(int(length), 10000))
    cid = ensure_conversation()
    with sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT) as conn:
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
    with sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT) as conn:
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


def search_memory(query: str = "", limit: int = 10) -> str:
    """Search durable memories using relevance-ranked meaningful terms.

    Common conversational words are ignored so a query such as "what did I tell
    you earlier?" does not inject unrelated facts merely because they contain
    words like "my" or "you". Results require at least one meaningful term
    match and are ranked by topic matches, fact matches, then recency.
    """
    limit = max(1, min(int(limit), 50))
    raw_query = str(query or "").strip().lower()
    with sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT) as conn:
        if not raw_query or raw_query in {"memory", "all", "everything"}:
            rows = conn.execute(
                "SELECT topic, fact, updated_at FROM memory ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        else:
            terms = _memory_query_terms(raw_query)
            if not terms:
                rows = []
            else:
                # Bound the scoring pool; memory is user-authored and typically
                # small, but never scan an unbounded table on each turn.
                candidates = conn.execute(
                    "SELECT topic, fact, updated_at FROM memory ORDER BY updated_at DESC LIMIT 500"
                ).fetchall()
                scored = []
                for row in candidates:
                    topic = str(row[0] or "").lower()
                    fact = str(row[1] or "").lower()
                    topic_hits = sum(1 for term in terms if term in topic)
                    fact_hits = sum(1 for term in terms if term in fact)
                    score = topic_hits * 3 + fact_hits
                    if score > 0:
                        scored.append((score, str(row[2] or ""), row))
                scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
                rows = [item[2] for item in scored[:limit]]
    return json.dumps(
        [{"topic": row[0], "fact": row[1], "updated_at": row[2]} for row in rows],
        ensure_ascii=False,
        indent=2,
    ) if rows else "No related memories found."


def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    return dot / (norm1 * norm2) if norm1 and norm2 else 0.0


def remember_semantic(topic: str = "general_knowledge", fact: str = "") -> str:
    """Store a fact with an embedding for semantic retrieval."""
    if not str(fact).strip():
        return "Error: Missing required 'fact' parameter."
    try:
        from ollama import Client
        client = Client(host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
        response = client.embed(model=EMBED_MODEL, input=str(fact), keep_alive=0)
        vectors = response.get("embeddings") or []
        embedding = vectors[0] if vectors else response.get("embedding") or []
        if not embedding:
            return "Error: embedding generation returned no vector."
        embedding_json = json.dumps(embedding)
    except Exception as exc:
        return f"Error: embedding generation failed: {exc}"

    with sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT) as conn:
        conn.execute(
            "INSERT INTO semantic_memory(topic, fact, embedding) VALUES (?, ?, ?)",
            (str(topic), str(fact), embedding_json),
        )
    return f"Stored semantic memory under '{topic}'."


def search_semantic_memory(query: str = "", limit: int = 5) -> str:
    """Retrieve semantically similar memories, falling back to keyword search."""
    if not str(query).strip():
        return search_memory(query, limit=limit)
    try:
        from ollama import Client
        client = Client(host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
        response = client.embed(model=EMBED_MODEL, input=str(query), keep_alive=0)
        vectors = response.get("embeddings") or []
        query_embedding = vectors[0] if vectors else response.get("embedding") or []
        if not query_embedding:
            return search_memory(query, limit=limit)
    except Exception:
        return search_memory(query, limit=limit)

    with sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT) as conn:
        rows = conn.execute("SELECT topic, fact, embedding FROM semantic_memory").fetchall()

    scored = []
    for topic, fact, raw_embedding in rows:
        try:
            score = _cosine_similarity(query_embedding, json.loads(raw_embedding))
            scored.append((score, topic, fact))
        except Exception:
            continue
    scored.sort(key=lambda item: item[0], reverse=True)
    results = [
        {"similarity": round(score, 4), "topic": topic, "fact": fact}
        for score, topic, fact in scored[: max(1, min(int(limit), 20))]
    ]
    return json.dumps(results, ensure_ascii=False, indent=2) if results else search_memory(query, limit=limit)


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
    with sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT) as conn:
        row = conn.execute("SELECT status, output, timestamp FROM background_tasks WHERE task_name = ?", (task_name,)).fetchone()
    return json.dumps({"task_name": task_name, "status": row[0], "output": row[1], "timestamp": row[2]}) if row else f"No legacy background task named '{task_name}'."


init_db()
