"""Frontend event routing, cancellation, and cross-process inference locking."""
from __future__ import annotations

import contextvars
import fcntl
import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable

from tools.runtime import utc_now
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


def acquire_inference_lock():
    os.makedirs(os.path.dirname(INFERENCE_LOCK_PATH), exist_ok=True)
    handle = open(INFERENCE_LOCK_PATH, "a+", encoding="utf-8")
    queued = False
    while True:
        if cancel_requested():
            handle.close()
            raise RuntimeError("Turn cancelled while waiting for the inference queue")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            emit_event("queue_acquired", queued=queued)
            return handle
        except BlockingIOError:
            if not queued:
                queued = True
                emit_event("queue_wait")
            time.sleep(0.10)


def release_inference_lock(handle) -> None:
    if handle is None:
        return
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()
