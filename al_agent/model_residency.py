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
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator

from ollama import Client

from tools.runtime import get_monitor_state, record_monitor_state, utc_now

from .background.config import (
    FAST_MODEL,
    FAST_MODEL_KEEP_ALIVE,
    FAST_OPTIONS,
    MAIN_OPTIONS,
    MODEL,
    OLLAMA_HOST,
    REPORT_MODEL,
    REPORT_MODEL_KEEP_ALIVE,
    REPORT_OPTIONS,
    REPORT_RESTORE_MODELS,
    VISION_MODEL,
)

INFERENCE_LOCK_PATH = os.environ.get("AGENT_INFERENCE_LOCK", "/app/workspace/.agent_inference.lock")
MODEL_MAINTENANCE_LOCK_PATH = os.environ.get(
    "AGENT_MODEL_MAINTENANCE_LOCK", "/app/workspace/.agent_model_maintenance.lock"
)
_REPORT_STATE_KEY = "agent.report_model_active"
_MODEL_MAINTENANCE_LOCK = threading.Lock()
_PREWARM_STATE_LOCK = threading.Lock()
_fast_prewarm_thread: threading.Thread | None = None


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


def _model_name(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("model") or value.get("name") or "")
    return str(getattr(value, "model", None) or getattr(value, "name", None) or "")


def _fast_runner_resident(client: Any) -> bool:
    """Return True only for the canonical fast runner/context combination."""
    try:
        response = client.ps()
        models = response.get("models", []) if isinstance(response, dict) else getattr(response, "models", [])
        expected_ctx = int((FAST_OPTIONS or {}).get("num_ctx", 0) or 0)
        for item in models or []:
            if _model_name(item) != FAST_MODEL:
                continue
            context = getattr(item, "context_length", None)
            if context is None and isinstance(item, dict):
                context = item.get("context_length")
            return not expected_ctx or int(context or 0) == expected_ctx
    except Exception:
        return False
    return False


@contextmanager
def model_maintenance_slot(*, blocking: bool = True) -> Iterator[None]:
    """Serialize model loads/swaps across frontend and worker processes.

    This is deliberately distinct from the foreground inference lock. A user
    turn never acquires this lock and therefore never queues behind prewarming.
    """
    if not _MODEL_MAINTENANCE_LOCK.acquire(blocking=blocking):
        raise BlockingIOError("model maintenance is already active")
    handle = None
    try:
        lock_dir = os.path.dirname(MODEL_MAINTENANCE_LOCK_PATH)
        if lock_dir:
            os.makedirs(lock_dir, exist_ok=True)
        handle = open(MODEL_MAINTENANCE_LOCK_PATH, "a+", encoding="utf-8")
        flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        fcntl.flock(handle.fileno(), flags)
        yield
    finally:
        if handle is not None:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()
        _MODEL_MAINTENANCE_LOCK.release()


def _prewarm_fast_when_idle(reason: str) -> None:
    """Warm fast without taking the foreground inference lock.

    The idle checks protect startup and post-report responsiveness. A separate
    maintenance lock prevents this best-effort load from colliding with report
    model swaps. Foreground turns never wait on this lock.
    """
    global _fast_prewarm_thread
    try:
        from .background.resources import _interactive_busy

        while True:
            while _interactive_busy():
                time.sleep(0.25)
            try:
                with model_maintenance_slot(blocking=False):
                    # Close the race between the idle check and maintenance-lock
                    # acquisition without abandoning the scheduled warm-up.
                    if _interactive_busy():
                        continue
                    client = _client()
                    if not _fast_runner_resident(client):
                        _warm(client, FAST_MODEL, FAST_OPTIONS, FAST_MODEL_KEEP_ALIVE)
                    break
            except BlockingIOError:
                time.sleep(0.1)
    finally:
        with _PREWARM_STATE_LOCK:
            _fast_prewarm_thread = None


def schedule_fast_model_prewarm(reason: str = "idle") -> bool:
    """Schedule one deduplicated, foreground-lock-free fast-model warm-up."""
    global _fast_prewarm_thread
    if not FAST_MODEL or FAST_MODEL == MODEL:
        return False
    with _PREWARM_STATE_LOCK:
        if _fast_prewarm_thread is not None and _fast_prewarm_thread.is_alive():
            return False
        thread = threading.Thread(
            target=_prewarm_fast_when_idle,
            args=(str(reason or "idle"),),
            name="fast-model-prewarm",
            daemon=True,
        )
        _fast_prewarm_thread = thread
        thread.start()
        return True


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
    with model_maintenance_slot():
        with background_inference_slot():
            for model in dict.fromkeys([MODEL, FAST_MODEL, VISION_MODEL]):
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
    """Unload the report model and optionally restore normal model residency."""
    client = _client()
    result = {"report_unloaded": False, "main_restored": False, "fast_prewarm_scheduled": False}
    with model_maintenance_slot():
        with background_inference_slot():
            result["report_unloaded"] = _unload(client, REPORT_MODEL)
            record_monitor_state(_REPORT_STATE_KEY, False)
            if restore and REPORT_RESTORE_MODELS:
                result["main_restored"] = _warm(client, MODEL, MAIN_OPTIONS, -1)
    # This happens only after the foreground inference lock and maintenance
    # lock have both been released. A user turn can therefore take priority.
    if restore and REPORT_RESTORE_MODELS and result["main_restored"]:
        result["fast_prewarm_scheduled"] = schedule_fast_model_prewarm("post-report")
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
