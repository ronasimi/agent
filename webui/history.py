"""Conversation-history serialization for the browser frontend."""
from __future__ import annotations
import json
from typing import Any
from tools import _load_chat_history_from_db

def _history(limit: int = 200, conversation_id: str | None = None) -> list[dict[str, Any]]:
    kwargs = {"limit": max(1, min(int(limit), 200))}
    if conversation_id is not None:
        kwargs["conversation_id"] = conversation_id
    rows = _load_chat_history_from_db(**kwargs)
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


def _history_export(limit: int = 0, conversation_id: str | None = None) -> str:
    """Serialize the complete stored conversation as readable plain text."""
    kwargs = {"limit": int(limit), "include_compacted": True}
    if conversation_id is not None:
        kwargs["conversation_id"] = conversation_id
    rows = _load_chat_history_from_db(**kwargs)
    parts: list[str] = []
    labels = {"user": "User", "assistant": "Assistant", "tool": "Tool", "system": "System"}
    for item in rows:
        role = str(item.get("role") or "message")
        label = labels.get(role, role.title())
        if role == "tool" and item.get("name"):
            label = f"Tool [{item['name']}]"
        content = str(item.get("content") or "").rstrip()
        tool_calls = item.get("tool_calls")
        if tool_calls:
            rendered = json.dumps(tool_calls, ensure_ascii=False, indent=2)
            content = (content + "\n\n" if content else "") + "Tool calls:\n" + rendered
        parts.append(f"{label}:\n{content}".rstrip())
    return "\n\n".join(parts).strip() + ("\n" if parts else "")
