"""Ollama model residency helpers for memory-bounded background stages.

The executor and decision model normally share the server. A larger reasoning
or research-report model may not fit alongside them, so the runtime uses
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
    DECISION_MODEL,
    DECISION_MODEL_KEEP_ALIVE,
    DECISION_OPTIONS,
    FAST_MODEL,
    FAST_MODEL_KEEP_ALIVE,
    FAST_OPTIONS,
    MAIN_OPTIONS,
    MODEL,
    OLLAMA_HOST,
    REASONING_MODEL,
    REASONING_MODEL_KEEP_ALIVE,
    REASONING_OPTIONS,
    REPORT_MODEL,
    REPORT_MODEL_KEEP_ALIVE,
    REPORT_OPTIONS,
    REPORT_RESTORE_MODELS,
    VISION_MODEL,
    VISION_MODEL_KEEP_ALIVE,
    VISION_OPTIONS,
)

INFERENCE_LOCK_PATH = os.environ.get("AGENT_INFERENCE_LOCK", "/app/workspace/.agent_inference.lock")
MODEL_MAINTENANCE_LOCK_PATH = os.environ.get(
    "AGENT_MODEL_MAINTENANCE_LOCK", "/app/workspace/.agent_model_maintenance.lock"
)
_REPORT_STATE_KEY = "agent.report_model_active"
_MODEL_MAINTENANCE_LOCK = threading.Lock()
_PREWARM_STATE_LOCK = threading.Lock()
_fast_prewarm_thread: threading.Thread | None = None
_decision_prewarm_thread: threading.Thread | None = None


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


def _runner_resident(client: Any, model: str, options: dict[str, Any]) -> bool:
    """Return True only when a model is resident at the requested context size."""
    try:
        response = client.ps()
        models = response.get("models", []) if isinstance(response, dict) else getattr(response, "models", [])
        expected_ctx = int((options or {}).get("num_ctx", 0) or 0)
        for item in models or []:
            if _model_name(item) != model:
                continue
            context = getattr(item, "context_length", None)
            if context is None and isinstance(item, dict):
                context = item.get("context_length")
            return not expected_ctx or int(context or 0) == expected_ctx
    except Exception:
        return False
    return False


def _decision_runner_resident(client: Any) -> bool:
    return _runner_resident(client, DECISION_MODEL, DECISION_OPTIONS)


def _fast_runner_resident(client: Any) -> bool:
    """Return True only for the canonical support/extraction runner."""
    return _runner_resident(client, FAST_MODEL, FAST_OPTIONS)


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
            try:
                while _interactive_busy():
                    time.sleep(0.25)
            except Exception:
                # Startup/tests may briefly observe an uninitialized runtime DB.
                # Prewarm is best-effort and must never surface as a background
                # thread failure or interfere with foreground startup.
                return
            try:
                with model_maintenance_slot(blocking=False):
                    # Close the race between the idle check and maintenance-lock
                    # acquisition without abandoning the scheduled warm-up.
                    if _interactive_busy():
                        continue
                    client = _client()
                    if not _fast_runner_resident(client):
                        _warm(client, FAST_MODEL, FAST_OPTIONS, FAST_MODEL_KEEP_ALIVE)
                    # Probe a distinct fast model only while it is already
                    # resident and the system is idle. This adds no foreground
                    # queueing and avoids an extra model swap on constrained GPUs.
                    try:
                        from . import state as _state
                        if _state.MODEL_CAPABILITY_PROBE_FAST and FAST_MODEL != MODEL:
                            from .model_capabilities import probe_model_capabilities
                            probe_model_capabilities(
                                client, FAST_MODEL, options=dict(FAST_OPTIONS or {}),
                                keep_alive=FAST_MODEL_KEEP_ALIVE,
                                cache_path=_state.MODEL_CAPABILITY_CACHE_PATH,
                                force=_state.MODEL_CAPABILITY_FORCE_PROBE,
                            )
                    except Exception:
                        # Best effort only; validator behavior remains unchanged
                        # when probing is unavailable or inconclusive.
                        pass
                    break
            except BlockingIOError:
                time.sleep(0.1)
    finally:
        with _PREWARM_STATE_LOCK:
            _fast_prewarm_thread = None


def _prewarm_decision_when_idle(reason: str) -> None:
    """Restore the tiny decision model without ever queueing ahead of a user."""
    global _decision_prewarm_thread
    try:
        from .background.resources import _interactive_busy
        while True:
            try:
                while _interactive_busy():
                    time.sleep(0.25)
            except Exception:
                # Startup/tests may briefly observe an uninitialized runtime DB.
                # Prewarm is best-effort and must never surface as a background
                # thread failure or interfere with foreground startup.
                return
            try:
                with model_maintenance_slot(blocking=False):
                    if _interactive_busy():
                        continue
                    client = _client()
                    if not _decision_runner_resident(client):
                        if REASONING_MODEL and REASONING_MODEL not in {MODEL, DECISION_MODEL}:
                            if _runner_resident(client, REASONING_MODEL, REASONING_OPTIONS):
                                _unload(client, REASONING_MODEL)
                        if VISION_MODEL and VISION_MODEL not in {MODEL, DECISION_MODEL, REASONING_MODEL}:
                            if _runner_resident(client, VISION_MODEL, VISION_OPTIONS):
                                _unload(client, VISION_MODEL)
                        _warm(client, DECISION_MODEL, DECISION_OPTIONS, DECISION_MODEL_KEEP_ALIVE)
                    try:
                        from . import state as _state
                        if _state.MODEL_CAPABILITY_PROBE_DECISION and DECISION_MODEL != MODEL:
                            from .model_capabilities import probe_model_capabilities
                            probe_model_capabilities(
                                client, DECISION_MODEL, options=dict(DECISION_OPTIONS or {}),
                                keep_alive=DECISION_MODEL_KEEP_ALIVE,
                                cache_path=_state.MODEL_CAPABILITY_CACHE_PATH,
                                force=_state.MODEL_CAPABILITY_FORCE_PROBE,
                            )
                    except Exception:
                        pass
                    break
            except BlockingIOError:
                time.sleep(0.1)
    finally:
        with _PREWARM_STATE_LOCK:
            _decision_prewarm_thread = None


def schedule_decision_model_prewarm(reason: str = "idle") -> bool:
    """Schedule one deduplicated decision-model warm-up."""
    global _decision_prewarm_thread
    if not DECISION_MODEL or DECISION_MODEL == MODEL:
        return False
    with _PREWARM_STATE_LOCK:
        if _decision_prewarm_thread is not None and _decision_prewarm_thread.is_alive():
            return False
        thread = threading.Thread(
            target=_prewarm_decision_when_idle,
            args=(str(reason or "idle"),),
            name="decision-model-prewarm",
            daemon=True,
        )
        _decision_prewarm_thread = thread
        thread.start()
        return True



def prepare_decision_model_for_interactive() -> bool:
    """Free a lingering reasoning slot before foreground decision inference.

    With ``OLLAMA_MAX_LOADED_MODELS=2`` the desired steady state is executor +
    decision. A previous 4B escalation must never force Ollama to choose which
    of those two models to evict when the next validator/compiler call arrives.
    Called only while the foreground inference lock is held.
    """
    if not DECISION_MODEL or DECISION_MODEL == MODEL:
        return False
    client = _client()
    if _decision_runner_resident(client):
        return False
    changed = False
    if REASONING_MODEL and REASONING_MODEL not in {MODEL, DECISION_MODEL}:
        if _runner_resident(client, REASONING_MODEL, REASONING_OPTIONS):
            changed = _unload(client, REASONING_MODEL) or changed
    if VISION_MODEL and VISION_MODEL not in {MODEL, DECISION_MODEL, REASONING_MODEL}:
        if _runner_resident(client, VISION_MODEL, VISION_OPTIONS):
            changed = _unload(client, VISION_MODEL) or changed
    return changed

def prepare_reasoning_model_for_interactive() -> bool:
    """Free the decision-model slot before a lazy 4B reasoning escalation.

    Called only while the foreground inference lock is held. The executor is
    kept resident so normal turns resume cheaply; the micro model is restored
    asynchronously after the reasoning call/turn.
    """
    if not REASONING_MODEL or REASONING_MODEL in {MODEL, DECISION_MODEL}:
        return False
    client = _client()
    if _runner_resident(client, REASONING_MODEL, REASONING_OPTIONS):
        return False
    changed = _unload(client, DECISION_MODEL)
    if VISION_MODEL and VISION_MODEL not in {MODEL, DECISION_MODEL, REASONING_MODEL}:
        if _runner_resident(client, VISION_MODEL, VISION_OPTIONS):
            changed = _unload(client, VISION_MODEL) or changed
    return changed


def prepare_vision_model_for_interactive() -> bool:
    """Reserve the second Ollama runner for the distinct multimodal model.

    The steady state is executor + micro. A vision request temporarily replaces
    the micro decision runner with the multimodal 4B while preserving the warm
    executor. A lingering text reasoner is also removed so Ollama never has to
    choose an eviction victim under ``OLLAMA_MAX_LOADED_MODELS=2``.
    """
    if not VISION_MODEL or VISION_MODEL == MODEL:
        return False
    client = _client()
    if _runner_resident(client, VISION_MODEL, VISION_OPTIONS):
        return False
    changed = _unload(client, DECISION_MODEL)
    if REASONING_MODEL and REASONING_MODEL not in {MODEL, DECISION_MODEL, VISION_MODEL}:
        if _runner_resident(client, REASONING_MODEL, REASONING_OPTIONS):
            changed = _unload(client, REASONING_MODEL) or changed
    return changed


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
            for model in dict.fromkeys([MODEL, FAST_MODEL, DECISION_MODEL, REASONING_MODEL, VISION_MODEL]):
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
    result = {"report_unloaded": False, "main_restored": False, "fast_prewarm_scheduled": False, "decision_prewarm_scheduled": False}
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
        result["decision_prewarm_scheduled"] = schedule_decision_model_prewarm("post-report")
    return result


def evict_report_model_for_interactive() -> bool:
    """Called with the interactive inference lock held before interactive-model work.

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
