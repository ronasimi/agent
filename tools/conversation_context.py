"""Conversation-scoped context shared across CLI, Web UI, and background helpers."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

DEFAULT_CONVERSATION_ID = "default"
_ACTIVE_CONVERSATION_ID: ContextVar[str] = ContextVar(
    "agent_conversation_id", default=DEFAULT_CONVERSATION_ID
)


def normalize_conversation_id(value: str | None) -> str:
    text = str(value or "").strip()
    if not text:
        return DEFAULT_CONVERSATION_ID
    # Conversation IDs are opaque local identifiers. Keep them filesystem/SQL/UI
    # friendly without accepting unbounded attacker-controlled strings.
    return text[:128]


def get_active_conversation_id() -> str:
    return normalize_conversation_id(_ACTIVE_CONVERSATION_ID.get())


def set_active_conversation_id(conversation_id: str | None):
    return _ACTIVE_CONVERSATION_ID.set(normalize_conversation_id(conversation_id))


def reset_active_conversation_id(token) -> None:
    _ACTIVE_CONVERSATION_ID.reset(token)


@contextmanager
def conversation_context(conversation_id: str | None) -> Iterator[str]:
    resolved = normalize_conversation_id(conversation_id)
    token = _ACTIVE_CONVERSATION_ID.set(resolved)
    try:
        yield resolved
    finally:
        _ACTIVE_CONVERSATION_ID.reset(token)
