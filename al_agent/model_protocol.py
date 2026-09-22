"""Small-model/Ollama protocol helpers.

Keep wire-format quirks out of the turn state machine.  The helpers here are
pure (apart from the optional sleep in the retry iterator), so they can be
unit-tested without importing the Ollama client or initializing agent state.
"""
from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from typing import Any


def _tool_call_parts(call: Any) -> tuple[str, str, Any]:
    """Return ``(id, name, arguments)`` for dict or Ollama ToolCall objects."""
    if isinstance(call, dict):
        function = call.get("function") or {}
        call_id = str(call.get("id") or "")
        name = str(function.get("name") or "") if isinstance(function, dict) else str(getattr(function, "name", "") or "")
        args = function.get("arguments", {}) if isinstance(function, dict) else getattr(function, "arguments", {})
        return call_id, name, args
    function = getattr(call, "function", None)
    return (
        str(getattr(call, "id", "") or ""),
        str(getattr(function, "name", "") or "") if function is not None else "",
        getattr(function, "arguments", {}) if function is not None else {},
    )


def _tool_call_key(call: Any) -> str:
    call_id, name, args = _tool_call_parts(call)
    if call_id:
        return "id:" + call_id
    try:
        encoded = json.dumps(args, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        encoded = str(args)
    return f"sig:{name}:{encoded}"


def merge_stream_tool_calls(accumulated: list[Any], incoming: Any) -> list[Any]:
    """Accumulate tool calls emitted across streamed Ollama chunks.

    Ollama streams complete tool-call objects, but multiple calls can arrive in
    different chunks.  Some servers/clients may also repeat a call in a later
    chunk.  Preserve arrival order while replacing an existing call with the
    same id/signature so the turn engine neither drops nor duplicates actions.
    """
    result = list(accumulated or [])
    if isinstance(incoming, dict):
        items = [incoming]
    else:
        try:
            items = list(incoming or [])
        except TypeError:
            items = [incoming] if incoming else []
    positions = {_tool_call_key(call): index for index, call in enumerate(result)}
    for call in items:
        key = _tool_call_key(call)
        if key in positions:
            result[positions[key]] = call
        else:
            positions[key] = len(result)
            result.append(call)
    return result


def tool_result_message(tool_name: str, content: str, *, tool_call_id: str = "") -> dict[str, Any]:
    """Build a native Ollama tool-result message plus a local correlation id.

    ``tool_name`` is the Ollama-native field.  ``tool_call_id`` is retained for
    local history/compaction pairing and for servers/clients that expose it.
    """
    message: dict[str, Any] = {
        "role": "tool",
        "content": str(content),
        "tool_name": str(tool_name),
    }
    if tool_call_id:
        message["tool_call_id"] = str(tool_call_id)
    return message


def ollama_wire_messages(messages: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return messages containing only fields accepted by Ollama's chat API.

    The harness keeps ``tool_call_id`` for local transaction pairing, but the
    native Ollama Python ``Message`` type currently identifies tool results via
    ``tool_name`` and has no ``tool_call_id`` field.
    """
    allowed = {"role", "content", "thinking", "images", "tool_name", "tool_calls"}
    return [{key: value for key, value in message.items() if key in allowed} for message in messages]


def is_retryable_transport_error(exc: Exception) -> bool:
    """Return whether a pre-stream model failure is plausibly transient.

    Ollama raises the built-in ``ConnectionError`` for connect failures and a
    ``ResponseError`` carrying ``status_code`` for HTTP failures. Avoid
    replaying deterministic 4xx request/schema errors; retry timeouts, rate
    limits, and server-side failures.
    """
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    status = getattr(exc, "status_code", None)
    try:
        code = int(status)
    except (TypeError, ValueError):
        return False
    return code in {408, 425, 429} or 500 <= code <= 599


def stream_with_preflight_retry(
    factory: Callable[[], Iterable[Any]],
    *,
    retries: int = 1,
    base_delay: float = 0.15,
    max_delay: float = 0.75,
    sleep_fn: Callable[[float], None] = time.sleep,
    on_retry: Callable[[int, Exception, float], None] | None = None,
    retry_if: Callable[[Exception], bool] = is_retryable_transport_error,
) -> Iterator[Any]:
    """Retry transient model transport failures only before the first chunk.

    Once any chunk has been yielded, replaying the request can duplicate visible
    output or tool calls.  Before the first chunk, however, a retry is safe and
    should not consume the agent's semantic/tool-loop iteration budget.
    """
    retries = max(0, int(retries))
    base_delay = max(0.0, float(base_delay))
    max_delay = max(base_delay, float(max_delay))
    attempt = 0
    while True:
        yielded = False
        try:
            for item in factory():
                yielded = True
                yield item
            return
        except Exception as exc:
            if yielded or attempt >= retries or not retry_if(exc):
                raise
            attempt += 1
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            if on_retry is not None:
                on_retry(attempt, exc, delay)
            if delay:
                sleep_fn(delay)


def warm_model(
    client: Any,
    model: str,
    *,
    options: dict[str, Any] | None = None,
    keep_alive: Any = -1,
    system_prompt: str = "",
    on_error: Callable[[Exception], None] | None = None,
) -> bool:
    """Load the model and optionally attempt to prime a stable prompt prefix.

    The empty-message request is the reliable part: it asks Ollama to load and
    keep the model resident. Prefix priming is deliberately optional because a
    tool-capable chat template may serialize dynamic tool schemas before normal
    messages; in that case a system-only request is not the common byte prefix
    of a real turn and may provide little or no KV-cache reuse. Measure
    ``prompt_eval_cached_count`` with ``scripts/benchmark_warmup.py`` before
    enabling it.

    ``options`` must match interactive turns, especially ``num_ctx``, otherwise
    Ollama can select/reload a different runner and invalidate the measurement.
    """
    base_options = dict(options or {})
    try:
        client.chat(model=model, messages=[], options=base_options, keep_alive=keep_alive)
        if system_prompt:
            prime_options = {**base_options, "num_predict": 1}
            client.chat(
                model=model,
                messages=[{"role": "system", "content": str(system_prompt)}],
                options=prime_options,
                keep_alive=keep_alive,
                think=False,
                stream=False,
            )
        return True
    except Exception as exc:
        if on_error is not None:
            on_error(exc)
        return False


def warm_model_async(
    client: Any,
    model: str,
    *,
    options: dict[str, Any] | None = None,
    keep_alive: Any = -1,
    system_prompt: str = "",
    on_error: Callable[[Exception], None] | None = None,
    on_success: Callable[[], None] | None = None,
) -> threading.Thread:
    """Warm the model on a daemon thread so a frontend can start immediately."""

    def _run() -> None:
        if warm_model(
            client, model, options=options, keep_alive=keep_alive,
            system_prompt=system_prompt, on_error=on_error,
        ) and on_success is not None:
            on_success()

    thread = threading.Thread(target=_run, name="model-warmup", daemon=True)
    thread.start()
    return thread
