"""Conversation-scoped persistent goals and definitions of done."""
from __future__ import annotations

import json
import sqlite3
from typing import Any

from .conversation_context import get_active_conversation_id, normalize_conversation_id
from .runtime import DB_PATH, DB_TIMEOUT, utc_now


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=15000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS conversation_goals (
            conversation_id TEXT PRIMARY KEY,
            goal TEXT NOT NULL,
            definition_of_done TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'active',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )"""
    )
    return conn


def _cid() -> str:
    return normalize_conversation_id(get_active_conversation_id())


def get_goal_record(conversation_id: str | None = None) -> dict[str, Any]:
    cid = normalize_conversation_id(conversation_id or _cid())
    with _connect() as conn:
        row = conn.execute("SELECT * FROM conversation_goals WHERE conversation_id=?", (cid,)).fetchone()
    return dict(row) if row else {}


def get_goal() -> str:
    """Return the current conversation's persistent goal and definition of done."""
    return json.dumps(get_goal_record(), ensure_ascii=False, separators=(",", ":"))


def set_goal(goal: str, definition_of_done: str = "", status: str = "active") -> str:
    """Create or update the current conversation's persistent goal."""
    goal = " ".join(str(goal or "").split())[:1600]
    done = " ".join(str(definition_of_done or "").split())[:1200]
    status = str(status or "active").strip().lower()
    if not goal:
        return "Error: goal is required."
    if status not in {"active", "paused", "complete"}:
        return "Error: status must be active, paused, or complete."
    cid = _cid(); now = utc_now()
    with _connect() as conn:
        conn.execute(
            """INSERT INTO conversation_goals(conversation_id,goal,definition_of_done,status,created_at,updated_at)
               VALUES(?,?,?,?,?,?)
               ON CONFLICT(conversation_id) DO UPDATE SET goal=excluded.goal,
                 definition_of_done=excluded.definition_of_done,status=excluded.status,updated_at=excluded.updated_at""",
            (cid, goal, done, status, now, now),
        )
    return get_goal()


def clear_goal() -> str:
    """Clear the current conversation's persistent goal."""
    cid = _cid()
    with _connect() as conn:
        conn.execute("DELETE FROM conversation_goals WHERE conversation_id=?", (cid,))
    return json.dumps({"conversation_id": cid, "cleared": True}, separators=(",", ":"))
