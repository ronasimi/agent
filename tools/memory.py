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

config = load_config()
EMBED_MODEL = config.get("agent", {}).get("embed_model", "nomic-embed-text")


def _connect() -> sqlite3.Connection:
    init_db()
    conn = sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=15000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    """Initialize persistent memory and runtime storage."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT) as conn:
        conn.execute("PRAGMA busy_timeout=15000")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("CREATE TABLE IF NOT EXISTS memory (topic TEXT PRIMARY KEY, fact TEXT, updated_at TEXT DEFAULT CURRENT_TIMESTAMP)")
        conn.execute("CREATE TABLE IF NOT EXISTS semantic_memory (id INTEGER PRIMARY KEY AUTOINCREMENT, topic TEXT, fact TEXT, embedding TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP)")
        conn.execute("CREATE TABLE IF NOT EXISTS chat_history (id INTEGER PRIMARY KEY AUTOINCREMENT, role TEXT, content TEXT, name TEXT, extra TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP)")
        conn.execute("CREATE TABLE IF NOT EXISTS conversation_state (id INTEGER PRIMARY KEY CHECK(id = 1), summary TEXT NOT NULL DEFAULT '', compacted_through_id INTEGER NOT NULL DEFAULT 0, updated_at TEXT DEFAULT CURRENT_TIMESTAMP)")
        state_columns = {row[1] for row in conn.execute("PRAGMA table_info(conversation_state)").fetchall()}
        if "compacted_through_id" not in state_columns:
            conn.execute("ALTER TABLE conversation_state ADD COLUMN compacted_through_id INTEGER NOT NULL DEFAULT 0")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tool_observations (
                id TEXT PRIMARY KEY,
                tool_name TEXT NOT NULL,
                content TEXT NOT NULL,
                char_count INTEGER NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("CREATE TABLE IF NOT EXISTS background_tasks (task_name TEXT PRIMARY KEY, status TEXT, output TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)")
        conn.execute("INSERT OR IGNORE INTO conversation_state(id, summary) VALUES (1, '')")
    init_runtime_db()


def _init_chat_db() -> None:
    init_db()


def _init_checkpoint_db() -> None:
    init_runtime_db()


def _save_message_to_db(msg: dict) -> int:
    init_db()
    role = msg.get("role", "")
    content = msg.get("content", "")
    name = msg.get("name")
    extra_data = {}
    for key in ("tool_calls", "tool_call_id"):
        if key in msg:
            extra_data[key] = msg[key]
    extra = json.dumps(extra_data, ensure_ascii=False) if extra_data else None
    with sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT) as conn:
        cursor = conn.execute(
            "INSERT INTO chat_history(role, content, name, extra) VALUES (?, ?, ?, ?)",
            (role, content, name, extra),
        )
        return int(cursor.lastrowid)


def _load_chat_history_from_db(limit: int = 20, *, include_compacted: bool = False) -> list[dict]:
    """Load recent model history or, for export only, the complete stored chat.

    Normal model/UI history remains bounded to uncompacted rows. ``include_compacted``
    is deliberately opt-in so copying/exporting the conversation can include the
    original rows that have already been summarized out of the model context.
    """
    init_db()
    requested_limit = int(limit)
    if include_compacted:
        # Export callers may pass 0 to request every stored row. Ordinary model/UI
        # history remains bounded below and never takes this unbounded path.
        export_limit = 0 if requested_limit <= 0 else min(requested_limit, 100000)
    else:
        export_limit = max(1, min(requested_limit, 200))
    with sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT) as conn:
        if include_compacted:
            if export_limit == 0:
                rows = conn.execute(
                    "SELECT id, role, content, name, extra FROM chat_history ORDER BY id DESC"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, role, content, name, extra FROM chat_history ORDER BY id DESC LIMIT ?",
                    (export_limit,),
                ).fetchall()
        else:
            row = conn.execute("SELECT compacted_through_id FROM conversation_state WHERE id = 1").fetchone()
            watermark = int(row[0] or 0) if row else 0
            rows = conn.execute(
                "SELECT id, role, content, name, extra FROM chat_history WHERE id > ? ORDER BY id DESC LIMIT ?",
                (watermark, export_limit),
            ).fetchall()
    rows.reverse()
    result = []
    for message_id, role, content, name, extra in rows:
        msg = {"role": role, "content": content or "", "_db_id": int(message_id)}
        if name:
            msg["name"] = name
        if extra:
            try:
                msg.update(json.loads(extra))
            except json.JSONDecodeError:
                pass
        result.append(msg)
    return result


def clear_chat_history() -> str:
    with sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT) as conn:
        conn.execute("DELETE FROM chat_history")
        conn.execute("DELETE FROM tool_observations")
        conn.execute("UPDATE conversation_state SET summary = '', compacted_through_id = 0, updated_at = CURRENT_TIMESTAMP WHERE id = 1")
    try:
        from .working_state import WorkingStateStore
        WorkingStateStore().clear()
    except Exception:
        pass
    return "Chat history, rolling context summary, and harness working state cleared."


def get_conversation_summary() -> str:
    init_db()
    with sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT) as conn:
        row = conn.execute("SELECT summary FROM conversation_state WHERE id = 1").fetchone()
    return row[0] if row else ""


def set_conversation_summary(summary: str) -> None:
    init_db()
    with sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT) as conn:
        conn.execute(
            "INSERT INTO conversation_state(id, summary, updated_at) VALUES(1, ?, CURRENT_TIMESTAMP) ON CONFLICT(id) DO UPDATE SET summary=excluded.summary, updated_at=excluded.updated_at",
            (str(summary or "").strip(),),
        )


def get_compacted_through_id() -> int:
    """Return the durable chat-history watermark included in the rolling summary."""
    init_db()
    with sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT) as conn:
        row = conn.execute("SELECT compacted_through_id FROM conversation_state WHERE id = 1").fetchone()
    return int(row[0] or 0) if row else 0


def get_messages_for_compaction(through_id: int) -> list[dict[str, Any]]:
    """Load uncompacted messages up to a fixed id for a background compaction job."""
    init_db()
    through_id = max(0, int(through_id))
    with sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT) as conn:
        state = conn.execute("SELECT compacted_through_id FROM conversation_state WHERE id = 1").fetchone()
        watermark = int(state[0] or 0) if state else 0
        rows = conn.execute(
            "SELECT id, role, content, name, extra FROM chat_history WHERE id > ? AND id <= ? ORDER BY id",
            (watermark, through_id),
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


def apply_conversation_compaction(summary: str, through_id: int) -> bool:
    """Atomically advance the rolling summary and its chat-history watermark."""
    summary = str(summary or "").strip()
    through_id = max(0, int(through_id))
    if not summary or not through_id:
        return False
    init_db()
    with sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT) as conn:
        cursor = conn.execute(
            """
            UPDATE conversation_state
            SET summary = ?, compacted_through_id = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = 1 AND compacted_through_id < ?
              AND EXISTS (SELECT 1 FROM chat_history WHERE id = ?)
            """,
            (summary[:8000], through_id, through_id, through_id),
        )
    return cursor.rowcount == 1


def store_tool_observation(tool_name: str, content: str) -> str:
    """Persist a large tool result and return an opaque retrieval handle."""
    init_db()
    observation_id = uuid.uuid4().hex
    text = str(content)
    with sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT) as conn:
        conn.execute(
            "INSERT INTO tool_observations(id, tool_name, content, char_count) VALUES (?, ?, ?, ?)",
            (observation_id, str(tool_name or "tool"), text, len(text)),
        )
    return observation_id


def read_observation(observation_id: str = "", offset: int = 0, length: int = 5000) -> str:
    """Read a bounded slice of a large tool result previously stored by the harness."""
    observation_id = str(observation_id or "").strip()
    if not observation_id:
        return "Error: Missing required 'observation_id' parameter."
    offset = max(0, int(offset))
    length = max(100, min(int(length), 10000))
    init_db()
    with sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT) as conn:
        row = conn.execute(
            "SELECT tool_name, content, char_count FROM tool_observations WHERE id = ?",
            (observation_id,),
        ).fetchone()
    if not row:
        return f"Observation '{observation_id}' was not found."
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


def search_memory(query: str = "", limit: int = 10) -> str:
    """Search durable memories using simple keyword matching."""
    limit = max(1, min(int(limit), 50))
    query = str(query or "").strip().lower()
    with sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT) as conn:
        if not query or query in {"memory", "all", "everything"}:
            rows = conn.execute(
                "SELECT topic, fact, updated_at FROM memory ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        else:
            keywords = [x for x in query.split() if x]
            clauses = []
            params: list[str] = []
            for word in keywords:
                clauses.append("(lower(topic) LIKE ? OR lower(fact) LIKE ?)")
                params.extend([f"%{word}%", f"%{word}%"])
            rows = conn.execute(
                "SELECT topic, fact, updated_at FROM memory WHERE " + " OR ".join(clauses) + " ORDER BY updated_at DESC LIMIT ?",
                params + [limit],
            ).fetchall()
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
