"""Lightweight in-process lifecycle hooks for tool execution.

Hooks are deliberately synchronous and best-effort. They must perform only
bounded local work and must never call an LLM. Custom hook failures are isolated
from the foreground turn so extension bugs cannot take down the agent loop.
"""
from __future__ import annotations

import inspect
import threading
from dataclasses import dataclass
from typing import Any, Callable

_ALLOWED_EVENTS = {"before_tool", "after_tool"}
_LOCK = threading.RLock()
_HOOK_TIMEOUT_SECONDS = 0.05
_DISABLED: set[tuple[str, str, str]] = set()


@dataclass(frozen=True)
class HookRegistration:
    event: str
    priority: int
    source: str
    callback: Callable[[dict[str, Any]], Any]


_HOOKS: dict[str, list[HookRegistration]] = {event: [] for event in _ALLOWED_EVENTS}


def agent_hook(event: str, *, priority: int = 0):
    """Mark a custom function as a lifecycle hook.

    Hook signature: ``hook(payload: dict) -> None | dict``.
    ``before_tool`` hooks may return ``{"deny": "reason"}`` to fail closed.
    Other return values are ignored. Hooks must not mutate external state unless
    that side effect is itself the purpose of an explicitly installed extension.
    """
    event = str(event or "").strip()
    if event not in _ALLOWED_EVENTS:
        raise ValueError(f"Unsupported hook event: {event}")

    def decorate(func: Callable[[dict[str, Any]], Any]):
        func._agent_hook_event = event
        func._agent_hook_priority = int(priority)
        return func

    return decorate


def register_hook(event: str, callback: Callable[[dict[str, Any]], Any], *, priority: int = 0, source: str = "runtime") -> None:
    event = str(event or "").strip()
    if event not in _ALLOWED_EVENTS:
        raise ValueError(f"Unsupported hook event: {event}")
    reg = HookRegistration(event=event, priority=int(priority), source=str(source or "runtime"), callback=callback)
    with _LOCK:
        rows = [item for item in _HOOKS[event] if not (item.source == reg.source and item.callback is callback)]
        rows.append(reg)
        rows.sort(key=lambda item: (-item.priority, item.source, getattr(item.callback, "__name__", "")))
        _HOOKS[event] = rows


def clear_hooks(*, source_prefix: str | None = None) -> None:
    """Clear all hooks, or only hooks whose source begins with a prefix."""
    with _LOCK:
        for event in _ALLOWED_EVENTS:
            if source_prefix is None:
                _HOOKS[event] = []
            else:
                _HOOKS[event] = [item for item in _HOOKS[event] if not item.source.startswith(source_prefix)]
        if source_prefix is None:
            _DISABLED.clear()
        else:
            _DISABLED.difference_update({key for key in _DISABLED if key[1].startswith(source_prefix)})


def register_module_hooks(module: Any, *, source: str) -> int:
    """Register functions decorated with :func:`agent_hook` from a module."""
    count = 0
    for _, func in inspect.getmembers(module, inspect.isfunction):
        event = str(getattr(func, "_agent_hook_event", "") or "")
        if not event:
            continue
        register_hook(event, func, priority=int(getattr(func, "_agent_hook_priority", 0)), source=source)
        count += 1
    return count


def _hook_key(item: HookRegistration) -> tuple[str, str, str]:
    return (item.event, item.source, getattr(item.callback, "__name__", "hook"))


def _invoke_bounded(item: HookRegistration, payload: dict[str, Any]) -> Any:
    key = _hook_key(item)
    with _LOCK:
        if key in _DISABLED:
            return {"hook_error": f"{item.source}:{key[2]} is disabled after a prior timeout"}
    result_box: list[Any] = []
    error_box: list[BaseException] = []

    def runner() -> None:
        try:
            result_box.append(item.callback(dict(payload)))
        except BaseException as exc:  # extension failures must not escape the harness
            error_box.append(exc)

    thread = threading.Thread(target=runner, name=f"agent-hook-{key[2]}", daemon=True)
    thread.start()
    thread.join(_HOOK_TIMEOUT_SECONDS)
    if thread.is_alive():
        with _LOCK:
            _DISABLED.add(key)
        return {"hook_error": f"{item.source}:{key[2]} exceeded {_HOOK_TIMEOUT_SECONDS * 1000:.0f}ms and was disabled"}
    if error_box:
        return {"hook_error": f"{item.source}:{key[2]}: {error_box[0]}"}
    return result_box[0] if result_box else None


def emit_hooks(event: str, payload: dict[str, Any]) -> list[Any]:
    """Run hooks in deterministic priority order with a tiny foreground budget."""
    if event not in _ALLOWED_EVENTS:
        return []
    with _LOCK:
        callbacks = list(_HOOKS[event])
    base = dict(payload or {})
    return [_invoke_bounded(item, base) for item in callbacks]


def before_tool_decision(tool_name: str, arguments: dict[str, Any]) -> str:
    """Return a denial reason from the first hook that explicitly blocks a call."""
    payload = {"event": "before_tool", "tool": str(tool_name), "arguments": dict(arguments or {})}
    for result in emit_hooks("before_tool", payload):
        if isinstance(result, dict) and result.get("deny"):
            return str(result.get("deny"))[:500]
    return ""


def after_tool_notify(tool_name: str, arguments: dict[str, Any], result: Any, *, error: bool = False) -> None:
    """Publish one bounded post-tool event without affecting the tool result."""
    text = str(result)
    emit_hooks("after_tool", {
        "event": "after_tool",
        "tool": str(tool_name),
        "arguments": dict(arguments or {}),
        "result_preview": text[:4000],
        "result_chars": len(text),
        "error": bool(error),
    })


def hook_inventory() -> list[dict[str, Any]]:
    with _LOCK:
        return [
            {"event": event, "source": item.source, "priority": item.priority, "name": getattr(item.callback, "__name__", "hook")}
            for event in sorted(_ALLOWED_EVENTS)
            for item in _HOOKS[event]
        ]
