"""Single execution path for registered tools.

Every caller (interactive loop, recipes, pipelines) goes through this module so
timeouts and future execution policy cannot drift between orchestration paths.
"""
from __future__ import annotations

import queue
import signal
import threading
from typing import Any

_TIMEOUT_LOCK = threading.Lock()
_TIMED_OUT_THREADS: dict[str, list[threading.Thread]] = {}
_MAX_ORPHANED_TOOL_THREADS = 8

def _prune_timed_out_threads() -> int:
    with _TIMEOUT_LOCK:
        total = 0
        for key in list(_TIMED_OUT_THREADS):
            alive = [thread for thread in _TIMED_OUT_THREADS[key] if thread.is_alive()]
            if alive:
                _TIMED_OUT_THREADS[key] = alive
                total += len(alive)
            else:
                _TIMED_OUT_THREADS.pop(key, None)
        return total

def _guard_timed_out_tool(name: str) -> None:
    total = _prune_timed_out_threads()
    with _TIMEOUT_LOCK:
        if any(thread.is_alive() for thread in _TIMED_OUT_THREADS.get(name, [])):
            raise TimeoutError(f"Tool '{name}' still has a previous timed-out invocation running; refusing to start another copy.")
        if total >= _MAX_ORPHANED_TOOL_THREADS:
            raise TimeoutError("Harness tool-timeout circuit breaker is open because too many timed-out tool invocations are still running.")


def execute_registered_tool(name: str, args: dict[str, Any]) -> Any:
    # Import lazily to avoid a catalog -> executor -> catalog cycle at startup.
    from .catalog import AVAILABLE_TOOLS_MAP, TOOL_METADATA

    _guard_timed_out_tool(name)
    func = AVAILABLE_TOOLS_MAP[name]
    timeout = TOOL_METADATA.get(name, {}).get("timeout")
    if not timeout:
        return func(**args)
    seconds = max(1, min(int(timeout), 300))

    # SIGALRM cleanly interrupts Python/native calls from the process main
    # thread.  Web turns run on worker threads, so use a daemon invocation there
    # to keep a buggy extension from wedging the turn forever.  The daemon may
    # finish later, but it cannot hold the interactive loop or process shutdown.
    if threading.current_thread() is threading.main_thread() and hasattr(signal, "SIGALRM"):
        previous_handler = signal.getsignal(signal.SIGALRM)

        def _raise_timeout(_signum, _frame):
            raise TimeoutError(f"Tool '{name}' exceeded its {seconds}-second harness timeout.")

        signal.signal(signal.SIGALRM, _raise_timeout)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            return func(**args)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)

    result_queue: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def invoke() -> None:
        try:
            result_queue.put((True, func(**args)))
        except BaseException as exc:  # propagate the original tool exception
            result_queue.put((False, exc))

    thread = threading.Thread(target=invoke, name=f"tool-{name}", daemon=True)
    thread.start()
    try:
        ok, value = result_queue.get(timeout=seconds)
    except queue.Empty as exc:
        with _TIMEOUT_LOCK:
            _TIMED_OUT_THREADS.setdefault(name, []).append(thread)
        raise TimeoutError(f"Tool '{name}' exceeded its {seconds}-second harness timeout; repeated invocations are blocked until the timed-out call exits.") from exc
    if ok:
        return value
    raise value
