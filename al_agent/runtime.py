# ruff: noqa: F401
"""Stable Web UI facade over the autonomous single-model runtime."""

from __future__ import annotations

from tools.runtime import record_monitor_state, utc_now

from . import state as _state
from . import turn_engine
from .events import (
    acquire_inference_lock as _acquire_inference_lock,
)
from .events import (
    acquire_turn_lock as _acquire_turn_lock,
)
from .events import (
    emit_event,
    frontend_event_context,
)
from .events import (
    release_inference_lock as _release_inference_lock,
)
from .events import (
    release_turn_lock as _release_turn_lock,
)
from .prompts import append_and_save, build_system_prompt
from .state import *

OLLAMA = _state.OLLAMA
ROUTER_OLLAMA = _state.ROUTER_OLLAMA


def handle_user_turn(
    messages: list[dict],
    user_input: str,
    thinking_enabled: bool,
    *,
    refresh_history: bool = False,
) -> None:
    """Refresh persisted history under the conversation lock when called by Web UI."""
    return turn_engine.handle_user_turn(
        messages,
        user_input,
        thinking_enabled,
        refresh_history=refresh_history,
        runtime_overrides={
            "OLLAMA": OLLAMA,
            "ROUTER_OLLAMA": ROUTER_OLLAMA,
            "append_and_save": append_and_save,
            "acquire_inference_lock": _acquire_inference_lock,
            "release_inference_lock": _release_inference_lock,
            "acquire_turn_lock": _acquire_turn_lock,
            "release_turn_lock": _release_turn_lock,
        },
    )
