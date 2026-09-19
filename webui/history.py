"""Conversation-history serialization for the browser frontend."""
from __future__ import annotations
from typing import Any
from tools import _load_chat_history_from_db

def _history(limit: int = 200) -> list[dict[str, Any]]:
    rows = _load_chat_history_from_db(limit=max(1, min(int(limit), 200)))
    clean = []
    for item in rows:
        entry = {
            "id": item.get("_db_id"),
            "role": item.get("role"),
            "content": item.get("content", ""),
        }
        if item.get("name"):
            entry["name"] = item["name"]
        if item.get("tool_calls"):
            entry["tool_calls"] = item["tool_calls"]
        clean.append(entry)
    return clean
