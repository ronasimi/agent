"""Durable deterministic failure lessons inspired by Abacus papercuts.

Recipes capture successful procedures. Failure lessons capture the inverse:
which tool/argument/result patterns failed and what later recovered. Recall is
pure SQLite/string matching so it adds no model call to the foreground path.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from datetime import datetime, timezone
from typing import Any

from .runtime import DB_PATH, DB_TIMEOUT, utc_now

_HALF_LIFE_DAYS = 14.0
_MAX_LESSONS = 256
_GENERIC = {
    "error", "failed", "failure", "tool", "status", "result", "argument",
    "invalid", "unknown", "missing", "cannot", "could", "would", "with",
}


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=15000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS failure_lessons (
            id TEXT PRIMARY KEY,
            tool_name TEXT NOT NULL,
            failure_signature TEXT NOT NULL,
            tripwires_json TEXT NOT NULL DEFAULT '[]',
            failure_summary TEXT NOT NULL DEFAULT '',
            fix_summary TEXT NOT NULL DEFAULT '',
            strength REAL NOT NULL DEFAULT 1.0,
            encounters INTEGER NOT NULL DEFAULT 1,
            successful_reuses INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL
        )"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_failure_lessons_tool ON failure_lessons(tool_name, last_seen_at DESC)")
    return conn


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _signature(tool_name: str, reason: str, result_text: str) -> str:
    basis = f"{tool_name}|{_norm(reason).lower()}|{_norm(result_text).lower()[:240]}"
    return hashlib.sha256(basis.encode("utf-8", errors="replace")).hexdigest()[:24]


def _tripwires(arguments: Any, reason: str, result_text: str) -> list[str]:
    items: list[str] = []
    lower = _norm(result_text).lower()
    for phrase in (
        "outside workspace", "path traversal", "permission denied", "no such file",
        "not found", "tool unavailable", "url validation failed", "unsupported url",
        "missing required", "unknown argument", "invalid arguments", "timed out",
    ):
        if phrase in lower:
            items.append(phrase)
    reason_text = _norm(reason).lower()
    if reason_text and reason_text not in {"tool_error", "error", "failed"}:
        items.append(reason_text[:100])
    if isinstance(arguments, dict):
        for key, value in arguments.items():
            if isinstance(value, (str, int, float, bool)):
                token = f"{key}={_norm(value)}"[:180]
                if len(token) >= 8:
                    items.append(token)
    # Add a bounded distinctive phrase from the error text.
    words = [w for w in re.findall(r"[a-z0-9_./:-]+", lower) if len(w) > 3 and w not in _GENERIC]
    if words:
        items.append(" ".join(words[:8])[:180])
    return list(dict.fromkeys(x for x in items if x))[:6]


def record_failure(tool_name: str, arguments: Any, reason: str, result_text: str) -> str:
    """Persist/update one failure signature without requiring a model call."""
    tool = str(tool_name or "").strip()
    if not tool:
        return ""
    signature = _signature(tool, reason, result_text)
    lesson_id = f"{tool}:{signature}"
    now = utc_now()
    trips = _tripwires(arguments, reason, result_text)
    summary = _norm(result_text)[:360]
    with _connect() as conn:
        existing = conn.execute("SELECT encounters, strength, tripwires_json FROM failure_lessons WHERE id=?", (lesson_id,)).fetchone()
        if existing:
            try:
                old_trips = json.loads(existing["tripwires_json"] or "[]")
            except Exception:
                old_trips = []
            merged = list(dict.fromkeys([*old_trips, *trips]))[:8]
            conn.execute(
                "UPDATE failure_lessons SET encounters=?, strength=?, tripwires_json=?, failure_summary=?, last_seen_at=? WHERE id=?",
                (int(existing["encounters"] or 0) + 1, min(12.0, float(existing["strength"] or 1.0) + 0.25), json.dumps(merged), summary, now, lesson_id),
            )
        else:
            conn.execute(
                "INSERT INTO failure_lessons(id,tool_name,failure_signature,tripwires_json,failure_summary,fix_summary,strength,encounters,successful_reuses,created_at,last_seen_at) VALUES(?,?,?,?,?,'',1.0,1,0,?,?)",
                (lesson_id, tool, signature, json.dumps(trips), summary, now, now),
            )
        # Bounded retention: weakest/oldest lessons are disposable.
        count = conn.execute("SELECT COUNT(*) FROM failure_lessons").fetchone()[0]
        if int(count) > _MAX_LESSONS:
            conn.execute(
                "DELETE FROM failure_lessons WHERE id IN (SELECT id FROM failure_lessons ORDER BY strength ASC, last_seen_at ASC LIMIT ?)",
                (int(count) - _MAX_LESSONS,),
            )
    return lesson_id


def record_recovery(lesson_id: str, failed_arguments: Any, successful_arguments: Any, successful_tool: str = "") -> None:
    """Strengthen a lesson when a later distinct action succeeds."""
    if not str(lesson_id or ""):
        return
    before = json.dumps(failed_arguments, ensure_ascii=False, sort_keys=True, default=str)[:500]
    after = json.dumps(successful_arguments, ensure_ascii=False, sort_keys=True, default=str)[:500]
    tool_note = f" using {successful_tool}" if successful_tool else ""
    fix = f"Recovery{tool_note}: change the failed approach from {before} to {after}; do not repeat the identical failed call."
    with _connect() as conn:
        conn.execute(
            "UPDATE failure_lessons SET fix_summary=?, strength=MIN(12.0,strength+2.0), successful_reuses=successful_reuses+1, last_seen_at=? WHERE id=?",
            (fix[:900], utc_now(), str(lesson_id)),
        )


def _decayed_score(row: sqlite3.Row) -> float:
    try:
        seen = datetime.fromisoformat(str(row["last_seen_at"]))
        if seen.tzinfo is None:
            seen = seen.replace(tzinfo=timezone.utc)
        days = max(0.0, (datetime.now(timezone.utc) - seen).total_seconds() / 86400.0)
    except Exception:
        days = 0.0
    return float(row["strength"] or 1.0) * math.pow(0.5, days / _HALF_LIFE_DAYS)


def relevant_failure_lessons(user_text: str, tool_names: set[str] | list[str], limit: int = 3) -> list[dict[str, Any]]:
    """Return bounded lessons for currently exposed tools, ranked deterministically."""
    tools = {str(x) for x in tool_names if str(x)}
    if not tools:
        return []
    placeholders = ",".join("?" for _ in tools)
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT * FROM failure_lessons WHERE tool_name IN ({placeholders}) ORDER BY last_seen_at DESC LIMIT 64",
            tuple(sorted(tools)),
        ).fetchall()
    haystack = _norm(user_text).lower()
    ranked: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        try:
            trips = [str(x) for x in json.loads(row["tripwires_json"] or "[]") if str(x)]
        except Exception:
            trips = []
        score = _decayed_score(row)
        matched = [trip for trip in trips if trip.lower() in haystack]
        # Recovered lessons are useful for the same tool even if the user's wording
        # does not contain the original low-level error string.
        if matched:
            score += 3.0 + len(matched)
        elif not row["fix_summary"] or score < 1.5:
            continue
        ranked.append((score, {
            "id": row["id"],
            "tool_name": row["tool_name"],
            "failure": row["failure_summary"],
            "fix": row["fix_summary"],
            "tripwires": trips,
            "score": round(score, 3),
        }))
    ranked.sort(key=lambda item: (-item[0], item[1]["tool_name"], item[1]["id"]))
    return [item for _score, item in ranked[:max(1, min(int(limit), 6))]]


def render_failure_lessons(user_text: str, tool_names: set[str] | list[str], limit: int = 3) -> str:
    rows = relevant_failure_lessons(user_text, tool_names, limit=limit)
    if not rows:
        return ""
    lines = ["[Harness learned failure tripwires: avoid repeating these known failed approaches when relevant.]"]
    for row in rows:
        fix = row["fix"] or "Do not repeat the identical failed call; choose a materially different argument/tool approach."
        lines.append(f"- {row['tool_name']}: {fix}")
    return "\n".join(lines)
