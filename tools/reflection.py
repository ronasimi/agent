"""Durable low-noise reflection notes created by the background fast role."""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from typing import Any

from .runtime import DB_PATH, DB_TIMEOUT, utc_now


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=15000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS rethink_notes (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            trigger_terms_json TEXT NOT NULL DEFAULT '[]',
            note TEXT NOT NULL,
            strength REAL NOT NULL DEFAULT 1.0,
            created_at TEXT NOT NULL,
            last_used_at TEXT
        )"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_rethink_notes_conversation ON rethink_notes(conversation_id, created_at DESC)")
    return conn


def store_reflection_notes(conversation_id: str, notes: list[dict[str, Any]]) -> int:
    stored = 0
    now = utc_now()
    with _connect() as conn:
        for item in notes[:3]:
            if not isinstance(item, dict):
                continue
            note = " ".join(str(item.get("note") or "").split())[:700]
            terms = [" ".join(str(x).lower().split())[:80] for x in item.get("trigger_terms", []) if str(x).strip()][:8]
            if len(note) < 20 or not terms:
                continue
            digest = hashlib.sha256((conversation_id + "|" + note.lower()).encode()).hexdigest()[:24]
            conn.execute(
                "INSERT OR IGNORE INTO rethink_notes(id,conversation_id,trigger_terms_json,note,strength,created_at) VALUES(?,?,?,?,1.0,?)",
                (digest, str(conversation_id), json.dumps(terms), note, now),
            )
            stored += 1
        # bounded per conversation
        conn.execute(
            "DELETE FROM rethink_notes WHERE id IN (SELECT id FROM rethink_notes WHERE conversation_id=? ORDER BY created_at DESC LIMIT -1 OFFSET 64)",
            (str(conversation_id),),
        )
    return stored


def render_relevant_reflections(conversation_id: str, user_text: str, limit: int = 2) -> str:
    hay = " ".join(str(user_text or "").lower().split())
    if not hay:
        return ""
    tokens = set(re.findall(r"[a-z0-9_.-]+", hay))
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM rethink_notes WHERE conversation_id=? ORDER BY created_at DESC LIMIT 64",
            (str(conversation_id),),
        ).fetchall()
    ranked = []
    for row in rows:
        try:
            terms = [str(x) for x in json.loads(row["trigger_terms_json"] or "[]")]
        except Exception:
            terms = []
        hits = 0
        for term in terms:
            tt = set(re.findall(r"[a-z0-9_.-]+", term.lower()))
            if term and (term in hay or (tt and tt.issubset(tokens))):
                hits += 1
        if hits:
            ranked.append((hits * float(row["strength"] or 1.0), row))
    ranked.sort(key=lambda item: (-item[0], str(item[1]["created_at"])), reverse=False)
    chosen = ranked[:max(1, min(int(limit), 4))]
    if not chosen:
        return ""
    ids = [str(row["id"]) for _score, row in chosen]
    with _connect() as conn:
        conn.executemany("UPDATE rethink_notes SET last_used_at=? WHERE id=?", [(utc_now(), x) for x in ids])
    lines = ["[Harness prior reflection: operational guidance from similar completed work; do not treat it as factual evidence.]"]
    lines.extend(f"- {row['note']}" for _score, row in chosen)
    return "\n".join(lines)
