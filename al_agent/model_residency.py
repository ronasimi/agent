"""Ollama model residency helpers for memory-bounded background stages.

The interactive model and fast validator normally share the server.  A larger
research report model may not fit alongside them, so the research worker uses
this module to temporarily swap model residency around synthesis.  All swaps
are explicit and best-effort; the normal Ollama request path remains the final
source of truth if a warm-up call is evicted under pressure.
"""
from __future__ import annotations

import fcntl
import os
from contextlib import contextmanager
from typing import Any, Iterator

from ollama import Client

from tools.runtime import get_monitor_state, record_monitor_state, utc_now

from .background.config import (
    FAST_MODEL,
    MAIN_OPTIONS,
    MODEL,
    OLLAMA_HOST,
    REPORT_MODEL,
    REPORT_MODEL_KEEP_ALIVE,
    REPORT_OPTIONS,
    REPORT_RESTORE_MODELS,
)

INFERENCE_LOCK_PATH = os.environ.get("AGENT_INFERENCE_LOCK", "/app/workspace/.agent_inference.lock")
_REPORT_STATE_KEY = "agent.report_model_active"


def _client() -> Client:
    return Client(host=os.environ.get("OLLAMA_HOST", OLLAMA_HOST))


def _unload(client: Any, model: str) -> bool:
    """Explicitly unload one Ollama model without generating text."""
    model = str(model or "").strip()
    if not model:
        return False
    try:
        client.chat(model=model, messages=[], keep_alive=0, stream=False)
        return True
    except Exception:
        return False


def _warm(client: Any, model: str, options: dict[str, Any], keep_alive: Any) -> bool:
    """Load one model with the same context/options its real requests use."""
    model = str(model or "").strip()
    if not model:
        return False
    try:
        client.chat(model=model, messages=[], options=dict(options or {}), keep_alive=keep_alive, stream=False)
        return True
    except Exception:
        return False


@contextmanager
def background_inference_slot() -> Iterator[None]:
    """Take the shared inference lock without ever queueing ahead of a user.

    Background work fails fast when the interactive path owns the lock.  The
    worker catches the resulting InferenceDeferred and retries after its normal
    cooldown, preserving foreground priority while preventing simultaneous
    model swaps/generations from exceeding the memory budget.
    """
    from .background.resources import InferenceDeferred, _ensure_interactive_idle

    _ensure_interactive_idle()
    os.makedirs(os.path.dirname(INFERENCE_LOCK_PATH), exist_ok=True)
    handle = open(INFERENCE_LOCK_PATH, "a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise InferenceDeferred("Interactive inference owns the model queue; report synthesis deferred.") from exc
        # Close the race between the pre-check and lock acquisition.
        _ensure_interactive_idle()
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def enter_report_model_stage(job_id: str = "") -> dict[str, Any]:
    """Evict smaller models and preload the configured report model."""
    client = _client()
    with background_inference_slot():
        for model in dict.fromkeys([MODEL, FAST_MODEL]):
            if model and model != REPORT_MODEL:
                _unload(client, model)
        loaded = _warm(client, REPORT_MODEL, REPORT_OPTIONS, REPORT_MODEL_KEEP_ALIVE)
        if not loaded:
            raise RuntimeError(f"Unable to load report model '{REPORT_MODEL}'.")
        # Validate the *resident* footprint after normal models have been
        # evicted. If the 9B runner/context still exceeds the configured worker
        # budget, fail explicitly instead of allowing Ollama to thrash models or
        # push the host into memory pressure.
        from .background.resources import resources_available
        ok, reason = resources_available()
        if not ok:
            _unload(client, REPORT_MODEL)
            record_monitor_state(_REPORT_STATE_KEY, False)
            raise RuntimeError(f"Report model exceeds the configured memory budget: {reason}")
        state = {
            "job_id": str(job_id or ""),
            "model": REPORT_MODEL,
            "started_at": utc_now(),
            "pid": os.getpid(),
        }
        record_monitor_state(_REPORT_STATE_KEY, state)
        return state


def exit_report_model_stage(job_id: str = "", *, restore: bool = True) -> dict[str, Any]:
    """Unload the report writer and restore only the foreground model.

    The fast role is deliberately *not* restored here. Report teardown owns the
    shared inference lock, so warming a validator/research runner that may never
    be used would make an arriving user wait behind avoidable background work.
    The 2B fast model instead lazy-loads on its next real request.
    """
    client = _client()
    result = {
        "report_unloaded": False,
        "main_restored": False,
        "fast_restored": False,
        "fast_restore_deferred": bool(restore and REPORT_RESTORE_MODELS and FAST_MODEL and FAST_MODEL != MODEL),
    }
    with background_inference_slot():
        result["report_unloaded"] = _unload(client, REPORT_MODEL)
        record_monitor_state(_REPORT_STATE_KEY, False)
        if restore and REPORT_RESTORE_MODELS:
            result["main_restored"] = _warm(client, MODEL, MAIN_OPTIONS, -1)
    return result


def evict_report_model_for_interactive() -> bool:
    """Called with the interactive inference lock held before main-model work.

    If a background report stage left the large writer resident, evict it before
    Ollama loads the interactive model.  This makes foreground recovery safe
    even when a research job was deferred between report calls.
    """
    active = get_monitor_state(_REPORT_STATE_KEY, False)
    if not active:
        return False
    client = _client()
    unloaded = _unload(client, str((active or {}).get("model") or REPORT_MODEL) if isinstance(active, dict) else REPORT_MODEL)
    record_monitor_state(_REPORT_STATE_KEY, False)
    return unloaded
