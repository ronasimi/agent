"""Frontend event routing, cancellation, and cross-process inference locking."""
from __future__ import annotations

import contextvars
import fcntl
import hashlib
import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable

from tools.runtime import utc_now
from tools.conversation_context import get_active_conversation_id, normalize_conversation_id
from .state import INFERENCE_LOCK_PATH

_EVENT_SINK: contextvars.ContextVar[Callable[[dict[str, Any]], None] | None] = contextvars.ContextVar("agent_event_sink", default=None)
_CANCEL_EVENT: contextvars.ContextVar[threading.Event | None] = contextvars.ContextVar("agent_cancel_event", default=None)


def emit_event(event_type: str, **payload: Any) -> None:
    sink = _EVENT_SINK.get()
    if sink is None:
        return
    try:
        sink({"type": str(event_type), "timestamp": utc_now(), **payload})
    except Exception:
        pass


@contextmanager
def frontend_event_context(sink=None, cancel_event=None):
    sink_token = _EVENT_SINK.set(sink)
    cancel_token = _CANCEL_EVENT.set(cancel_event)
    try:
        yield
    finally:
        _CANCEL_EVENT.reset(cancel_token)
        _EVENT_SINK.reset(sink_token)


def cancel_requested() -> bool:
    token = _CANCEL_EVENT.get()
    return bool(token and token.is_set())



def _acquire_file_lock(path: str, *, wait_event: str = "queue_wait", acquired_event: str = "queue_acquired"):
    """Acquire one cancellable advisory file lock.

    The helper is shared by the per-conversation turn lock and the global
    inference lock so both queues have identical cancellation semantics.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    handle = open(path, "a+", encoding="utf-8")
    queued = False
    while True:
        if cancel_requested():
            handle.close()
            raise RuntimeError("Turn cancelled while waiting for a harness lock")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            emit_event(acquired_event, queued=queued)
            return handle
        except BlockingIOError:
            if not queued:
                queued = True
                emit_event(wait_event)
            time.sleep(0.10)


def acquire_turn_lock():
    """Serialize turns only within the active conversation.

    Historically the global inference lock also serialized history/state work.
    That made unrelated network/file preflight block Ollama for every chat. A
    per-conversation lock preserves ordering without occupying the model queue.
    """
    conversation_id = normalize_conversation_id(get_active_conversation_id())
    digest = hashlib.sha256(conversation_id.encode("utf-8", errors="replace")).hexdigest()[:20]
    base = os.path.dirname(INFERENCE_LOCK_PATH) or "."
    return _acquire_file_lock(
        os.path.join(base, f".agent_turn_{digest}.lock"),
        wait_event="turn_queue_wait",
        acquired_event="turn_queue_acquired",
    )


def release_turn_lock(handle) -> None:
    release_inference_lock(handle)


def acquire_inference_lock():
    return _acquire_file_lock(INFERENCE_LOCK_PATH)


def release_inference_lock(handle) -> None:
    if handle is None:
        return
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()
