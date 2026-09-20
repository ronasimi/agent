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
import time
from contextlib import contextmanager
from typing import Any, Iterator

from ollama import Client

from tools.runtime import get_monitor_state, record_monitor_state, utc_now
from .decision_engine import DecisionEngineClient

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
    REPORT_RESTORE_FAST_MODEL,
    REPORT_RESTORE_MODELS,
    AGENT_CFG,
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
    """Evict smaller models and Laya, then preload the report model."""
    client = _client()
    decision = DecisionEngineClient(dict(AGENT_CFG.get("decision_engine") or {}))
    # The Laya sidecar is a single shared process, so this HTTP unload actually
    # releases its encoder RAM for every frontend before the 9B writer loads.
    decision_cfg = dict(AGENT_CFG.get("decision_engine") or {})
    if bool(decision_cfg.get("suspend_during_report", True)):
        decision.unload(timeout=2.0)
        # An encoder cold-load may have been in flight when /research asked it
        # to unload. Wait briefly for that cancelled loader to drop its temporary
        # weights before admitting the 9B writer. If it is still winding down,
        # defer the background job rather than creating a transient RAM spike.
        deadline = time.monotonic() + max(0.5, float(decision_cfg.get("report_unload_wait_seconds", 5.0)))
        while time.monotonic() < deadline:
            health = decision.health()
            if not bool(health.get("loaded")) and not bool(health.get("loading")):
                break
            time.sleep(0.1)
        else:
            from .background.resources import InferenceDeferred
            raise InferenceDeferred("Decision engine is still releasing memory; report synthesis deferred.")
    with background_inference_slot():
        for model in dict.fromkeys([MODEL, FAST_MODEL]):
            if model and model != REPORT_MODEL:
                _unload(client, model)
        loaded = _warm(client, REPORT_MODEL, REPORT_OPTIONS, REPORT_MODEL_KEEP_ALIVE)
        if not loaded:
            # Restore the decision service best-effort if report load fails.
            decision.preload(timeout=0.5)
            raise RuntimeError(f"Unable to load report model '{REPORT_MODEL}'.")
        from .background.resources import resources_available
        ok, reason = resources_available()
        if not ok:
            _unload(client, REPORT_MODEL)
            record_monitor_state(_REPORT_STATE_KEY, False)
            decision.preload(timeout=0.5)
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
    """Unload report model, restore normal models, and asynchronously warm Laya."""
    client = _client()
    decision = DecisionEngineClient(dict(AGENT_CFG.get("decision_engine") or {}))
    result = {"report_unloaded": False, "main_restored": False, "fast_restored": False, "decision_preload_requested": False}
    with background_inference_slot():
        result["report_unloaded"] = _unload(client, REPORT_MODEL)
        record_monitor_state(_REPORT_STATE_KEY, False)
        if restore and REPORT_RESTORE_MODELS:
            result["main_restored"] = _warm(client, MODEL, MAIN_OPTIONS, -1)
            if REPORT_RESTORE_FAST_MODEL and FAST_MODEL and FAST_MODEL != MODEL:
                result["fast_restored"] = _warm(client, FAST_MODEL, FAST_OPTIONS, FAST_MODEL_KEEP_ALIVE)
    result["decision_preload_requested"] = decision.preload(timeout=0.5)
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
    DecisionEngineClient(dict(AGENT_CFG.get("decision_engine") or {})).preload(timeout=0.5)
    return unloaded
