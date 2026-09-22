# ruff: noqa: F401
"""Web application runtime facade for Al Agent.

The browser frontend imports this module as its stable composition surface.
Implementation remains split across focused modules:

- :mod:`al_agent.state`: configuration and long-lived services
- :mod:`al_agent.events`: frontend events, cancellation, inference locking
- :mod:`al_agent.prompts`: stable policy, memory, and media prompt construction
- :mod:`al_agent.turn_support`: deterministic tool-loop helpers
- :mod:`al_agent.turn_engine`: the interactive model/tool state machine

Policy invariants retained here for discoverability/backward source checks:
"Never infer the current clock from uptime".
"Their absence from the currently supplied schemas" does not mean a withheld
capability is unavailable to the harness.
"""
from __future__ import annotations

from tools.runtime import record_monitor_state, utc_now
from tools.task_requirements import TaskRequirementLedger

from . import state as _state
from . import turn_engine as _turn_engine
from . import turn_support as _turn_support
from .events import (
    acquire_inference_lock as _default_acquire_inference_lock,
)
from .events import (
    cancel_requested as _cancel_requested,
)
from .events import (
    emit_event,
    frontend_event_context,
)
from .events import (
    release_inference_lock as _default_release_inference_lock,
)
from .prompts import (
    IMAGE_REGEX,
    SYSTEM_POLICY,
    append_and_save,
    build_memory_context,
    build_system_prompt,
    encode_image,
)
from .prompts import (
    clean_thinking as _clean_thinking,
)
from .state import *  # stable Web UI runtime surface
from .turn_support import (
    _adaptive_iteration_limit,
    _add_recovery_schema,
    _bounded_tool_result,
    _bounded_tool_result_with_ref,
    _ensure_tool_schemas,
    _execute_registered_tool,
    _extract_tool_calls,
    _parse_tool_calls,
    _prune_compacted_history,
    _refresh_requirement_tool_schemas,
    _sanitize_tool_call_batch,
    _suppress_completed_requirement_calls,
    _tool_status_prefix,
    _user_requests_recheck,
)

# Monkeypatch-friendly aliases used by tests and embedders.
OLLAMA = _state.OLLAMA
LOOP_VALIDATOR_CLIENT = _state.LOOP_VALIDATOR_CLIENT
_acquire_inference_lock = _default_acquire_inference_lock
_release_inference_lock = _default_release_inference_lock
_queue_compaction_if_needed = _turn_support._queue_compaction_if_needed


def _runtime_overrides() -> dict[str, object]:
    """Return request-local facade overrides without mutating module globals.

    Tests and embedders historically monkeypatch this facade.  Passing those
    values explicitly preserves that contract while avoiding cross-request
    writes into :mod:`turn_engine` / :mod:`turn_support` in threaded WebUI
    deployments.
    """
    return {
        "OLLAMA": globals().get("OLLAMA", _state.OLLAMA),
        "LOOP_VALIDATOR_CLIENT": globals().get("LOOP_VALIDATOR_CLIENT", _state.LOOP_VALIDATOR_CLIENT),
        "RECIPE_MATCH_THRESHOLD": globals().get("RECIPE_MATCH_THRESHOLD", _state.RECIPE_MATCH_THRESHOLD),
        "TaskRequirementLedger": globals().get("TaskRequirementLedger", TaskRequirementLedger),
        "record_monitor_state": globals().get("record_monitor_state", record_monitor_state),
        "append_and_save": globals().get("append_and_save", append_and_save),
        "acquire_inference_lock": globals().get("_acquire_inference_lock", _default_acquire_inference_lock),
        "release_inference_lock": globals().get("_release_inference_lock", _default_release_inference_lock),
        "queue_compaction_if_needed": globals().get("_queue_compaction_if_needed", _turn_support._queue_compaction_if_needed),
    }


def _finalize_after_limit(messages, turn_tail=None, reason="The tool-call safety limit was reached."):
    overrides = _runtime_overrides()
    return _turn_support._finalize_after_limit(
        messages,
        turn_tail,
        reason,
        client=overrides["OLLAMA"],
        append_fn=overrides["append_and_save"],
    )


def handle_user_turn(messages: list[dict], user_input: str, thinking_enabled: bool, **kwargs) -> None:
    return _turn_engine.handle_user_turn(
        messages,
        user_input,
        thinking_enabled,
        runtime_overrides=_runtime_overrides(),
        **kwargs,
    )
