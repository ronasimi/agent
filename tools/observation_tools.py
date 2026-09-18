"""Utilities for comparing durable observations without spending LLM context."""
from __future__ import annotations

import difflib
import json
import sqlite3

from .memory import init_db
from .runtime import DB_PATH, DB_TIMEOUT


def _load_observation(observation_id: str) -> tuple[str, str] | None:
    init_db()
    with sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT) as conn:
        row = conn.execute("SELECT tool_name, content FROM tool_observations WHERE id = ?", (str(observation_id),)).fetchone()
    return (str(row[0]), str(row[1])) if row else None


def diff_observations(old_id: str, new_id: str, max_diff_chars: int = 12000) -> str:
    """Return a bounded unified diff between two durable tool observations."""
    old = _load_observation(old_id)
    new = _load_observation(new_id)
    if old is None:
        return f"Error: observation '{old_id}' was not found."
    if new is None:
        return f"Error: observation '{new_id}' was not found."
    max_diff_chars = max(1000, min(int(max_diff_chars), 30000))
    diff = "\n".join(difflib.unified_diff(
        old[1].splitlines(), new[1].splitlines(),
        fromfile=f"{old[0]}:{old_id}", tofile=f"{new[0]}:{new_id}", lineterm="", n=2,
    ))
    return json.dumps({
        "old_id": old_id, "new_id": new_id, "changed": old[1] != new[1],
        "old_chars": len(old[1]), "new_chars": len(new[1]), "diff": diff[:max_diff_chars],
        "truncated": len(diff) > max_diff_chars,
    }, ensure_ascii=False, indent=2)
