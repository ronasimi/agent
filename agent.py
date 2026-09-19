"""Compatibility facade and executable entry point for Al Agent.

The implementation is modularized under :mod:`al_agent`:

- ``al_agent.state``: configuration and long-lived services
- ``al_agent.events``: frontend events, cancellation, inference locking
- ``al_agent.prompts``: stable policy, memory/media prompt construction
- ``al_agent.turn_support``: deterministic tool-loop helpers
- ``al_agent.turn_engine``: the interactive model/tool state machine
- ``al_agent.cli``: terminal frontend

Existing integrations may continue importing ``agent`` unchanged.

Policy invariants retained here for discoverability/backward source checks:
"Never infer the current clock from uptime".
"Their absence from the currently supplied schemas" does not mean a withheld
capability is unavailable to the harness.
"""
from __future__ import annotations

from tools.runtime import record_monitor_state, utc_now
from tools.task_requirements import TaskRequirementLedger

from al_agent import state as _state
from al_agent.state import *  # noqa: F401,F403 - compatibility surface
from al_agent.events import (
    acquire_inference_lock as _default_acquire_inference_lock,
    cancel_requested as _cancel_requested,
    emit_event,
    frontend_event_context,
    release_inference_lock as _default_release_inference_lock,
)
from al_agent.prompts import (
    IMAGE_REGEX,
    SYSTEM_POLICY,
    append_and_save,
    build_memory_context,
    build_system_prompt,
    clean_thinking as _clean_thinking,
    encode_image,
)
from al_agent.console import Spinner, get_bottom_toolbar, print_perf_stats as _print_perf_stats
from al_agent import turn_support as _turn_support
from al_agent.turn_support import (
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
from al_agent import turn_engine as _turn_engine

# Monkeypatch-friendly compatibility aliases used by tests and external callers.
OLLAMA = _state.OLLAMA
LOOP_VALIDATOR_CLIENT = _state.LOOP_VALIDATOR_CLIENT
_acquire_inference_lock = _default_acquire_inference_lock
_release_inference_lock = _default_release_inference_lock
_queue_compaction_if_needed = _turn_support._queue_compaction_if_needed


def _sync_compat_overrides() -> None:
    """Propagate facade monkeypatches into the modular turn engine.

    Production code normally never needs this.  It preserves the historical
    ``agent.<name>`` testing/integration surface while allowing implementation
    modules to stay focused.
    """
    for name in (
        "OLLAMA", "LOOP_VALIDATOR_CLIENT", "RECIPE_MATCH_THRESHOLD",
        "TaskRequirementLedger", "record_monitor_state", "append_and_save",
        "_acquire_inference_lock", "_release_inference_lock",
        "_queue_compaction_if_needed",
    ):
        if name in globals():
            setattr(_turn_engine, name, globals()[name])
    _turn_support.OLLAMA = globals().get("OLLAMA", _state.OLLAMA)
    _turn_support.append_and_save = globals().get("append_and_save", append_and_save)


def _finalize_after_limit(messages, turn_tail=None, reason="The tool-call safety limit was reached."):
    _sync_compat_overrides()
    return _turn_support._finalize_after_limit(messages, turn_tail, reason)


def handle_user_turn(messages: list[dict], user_input: str, thinking_enabled: bool, **kwargs) -> None:
    _sync_compat_overrides()
    return _turn_engine.handle_user_turn(messages, user_input, thinking_enabled, **kwargs)


def print_jobs() -> None:
    from al_agent.cli import print_jobs as _impl
    return _impl()


def print_job(job_id: str) -> None:
    from al_agent.cli import print_job as _impl
    return _impl(job_id)


def main() -> None:
    from al_agent.cli import main as _main
    return _main()


if __name__ == "__main__":
    main()
