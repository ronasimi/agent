"""Single execution path for registered tools.

Every caller (interactive loop, recipes, pipelines) goes through this module so
timeouts and future execution policy cannot drift between orchestration paths.
"""
from __future__ import annotations

import queue
import signal
import threading
from typing import Any


def execute_registered_tool(name: str, args: dict[str, Any]) -> Any:
    # Import lazily to avoid a catalog -> executor -> catalog cycle at startup.
    from .catalog import AVAILABLE_TOOLS_MAP, TOOL_METADATA

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
        raise TimeoutError(f"Tool '{name}' exceeded its {seconds}-second harness timeout.") from exc
    if ok:
        return value
    raise value
