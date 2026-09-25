"""Small-model/Ollama protocol helpers.

Keep wire-format quirks out of the turn state machine.  The helpers here are
pure (apart from the optional sleep in the retry iterator), so they can be
unit-tested without importing the Ollama client or initializing agent state.
"""
from __future__ import annotations

import json
import queue
import re
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any



class StreamTimeoutError(TimeoutError):
    """Raised when a streamed model response misses a harness deadline."""


def _iter_stream_with_timeouts(
    stream: Iterable[Any],
    *,
    first_chunk_timeout_seconds: float | None,
    idle_timeout_seconds: float | None,
) -> Iterator[Any]:
    """Yield stream chunks with separate prefill/first-chunk and idle deadlines.

    Ollama/httpx exposes one read timeout for the whole response.  A local model
    may legitimately spend longer on prompt prefill than we want to tolerate
    between already-started stream chunks.  Pumping the iterator on a daemon
    thread lets the harness distinguish those two phases without buffering the
    response or blocking UI streaming.  The caller closes the underlying stream
    on timeout, which normally interrupts the producer thread immediately.
    """

    first_timeout = (
        float(first_chunk_timeout_seconds)
        if first_chunk_timeout_seconds is not None and float(first_chunk_timeout_seconds) > 0
        else None
    )
    idle_timeout = (
        float(idle_timeout_seconds)
        if idle_timeout_seconds is not None and float(idle_timeout_seconds) > 0
        else None
    )
    if first_timeout is None and idle_timeout is None:
        yield from stream
        return

    events: queue.Queue[tuple[str, Any]] = queue.Queue()

    def pump() -> None:
        try:
            for item in stream:
                events.put(("item", item))
            events.put(("done", None))
        except BaseException as exc:  # provider iterator boundary
            events.put(("error", exc))

    threading.Thread(target=pump, name="ollama-stream-pump", daemon=True).start()
    first = True
    while True:
        timeout = first_timeout if first else idle_timeout
        try:
            kind, payload = events.get(timeout=timeout) if timeout is not None else events.get()
        except queue.Empty as exc:
            phase = "first response chunk" if first else "stream activity"
            raise StreamTimeoutError(
                f"Timed out waiting for {phase} after {timeout:.1f}s"
            ) from exc
        if kind == "done":
            return
        if kind == "error":
            raise payload
        first = False
        yield payload

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
    function = call.get("function", {}) if isinstance(call, dict) else getattr(call, "function", None)
    index = function.get("index") if isinstance(function, dict) else getattr(function, "index", None)
    return f"index:{index}" if index is not None else ""


def merge_stream_tool_calls(accumulated: list[Any], incoming: Any) -> list[Any]:
    """Accumulate tool calls emitted across streamed Ollama chunks.

    Ollama streams complete tool-call objects, but multiple calls can arrive in
    different chunks.  Some servers/clients may also repeat a call in a later
    chunk.  Preserve arrival order while replacing an existing call with the
    same explicit id/index so distinct equal-argument calls retain their identity. Calls without an
    explicit identity are preserved instead of guessed to be duplicates.
    """
    result = list(accumulated or [])
    if isinstance(incoming, dict):
        items = [incoming]
    else:
        try:
            items = list(incoming or [])
        except TypeError:
            items = [incoming] if incoming else []
    positions = {_tool_call_key(call): index for index, call in enumerate(result) if _tool_call_key(call)}
    for call in items:
        key = _tool_call_key(call)
        if key and key in positions:
            result[positions[key]] = call
        else:
            if key:
                positions[key] = len(result)
            result.append(call)
    return result


_QWEN_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=([^>\s]+)>\s*(.*?)\s*</function>\s*</tool_call>",
    flags=re.I | re.S,
)
_QWEN_PARAMETER_RE = re.compile(
    r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>",
    flags=re.I | re.S,
)


def _qwen_parameter_value(raw: str) -> str:
    """Remove only the template's structural boundary newline(s).

    ``str.strip()`` is intentionally avoided: leading/trailing spaces can be
    meaningful for shell snippets, source text, or other string parameters.
    """
    value = str(raw or "")
    if value.startswith("\r\n"):
        value = value[2:]
    elif value.startswith("\n"):
        value = value[1:]
    if value.endswith("\r\n"):
        value = value[:-2]
    elif value.endswith("\n"):
        value = value[:-1]
    return value


def extract_qwen_xml_tool_calls(content: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse Qwen3.x's native textual XML tool-call envelope.

    Qwen3.8 GGUF chat templates serialize *definitions* as JSON under ``<tools>``
    but instruct the model to emit invocations as::

        <tool_call>
        <function=tool_name>
        <parameter=arg>value</parameter>
        </function>
        </tool_call>

    Parameter bodies are intentionally kept as strings here; the canonical
    registry normalizer later coerces integers, booleans, arrays, and objects
    according to the supplied tool schema.  Parsing is strict enough that normal
    prose or arbitrary JSON can never become executable by accident.
    """
    text = str(content or "")
    if "<tool_call>" not in text.lower():
        return [], []
    calls: list[dict[str, Any]] = []
    errors: list[str] = []
    matches = list(_QWEN_TOOL_CALL_RE.finditer(text))
    if not matches:
        return [], ["malformed Qwen XML tool call envelope"]
    opening_count = len(re.findall(r"<tool_call\s*>", text, flags=re.I))
    if opening_count != len(matches):
        return [], ["malformed or incomplete Qwen XML tool call envelope"]
    for left, right in zip(matches, matches[1:]):
        if text[left.end() : right.start()].strip():
            return [], ["unexpected text between Qwen XML tool calls"]
    if text[matches[-1].end() :].strip():
        return [], ["unexpected text after final Qwen XML tool call"]
    for index, match in enumerate(matches, start=1):
        name = match.group(1).strip().strip('"\'')
        body = match.group(2)
        params: dict[str, Any] = {}
        spans: list[tuple[int, int]] = []
        duplicate = ""
        for param in _QWEN_PARAMETER_RE.finditer(body):
            key = param.group(1).strip().strip('"\'')
            if key in params:
                duplicate = key
                break
            params[key] = _qwen_parameter_value(param.group(2))
            spans.append(param.span())
        if duplicate:
            errors.append(f"call {index} ({name or '[missing]'}): duplicate parameter '{duplicate}'")
            continue
        residue_parts: list[str] = []
        cursor = 0
        for start, end in spans:
            residue_parts.append(body[cursor:start])
            cursor = end
        residue_parts.append(body[cursor:])
        residue = "".join(residue_parts).strip()
        if residue:
            errors.append(f"call {index} ({name or '[missing]'}): unexpected text inside function envelope")
            continue
        if not name:
            errors.append(f"call {index}: missing function name")
            continue
        calls.append({
            "type": "function",
            "function": {"name": name, "arguments": params},
        })
    return calls, errors


def qwen_xml_tool_prelude(content: str) -> str:
    """Return optional prose before the first XML tool call, never the XML itself."""
    text = str(content or "")
    match = _QWEN_TOOL_CALL_RE.search(text)
    return text[: match.start()].strip() if match else text.strip()


@dataclass
class StreamCapture:
    """Normalized result from one streamed Ollama chat response."""

    content: str
    thinking: str
    tool_calls: list[Any]
    perf_stats: dict[str, Any]
    first_token_at: float | None
    first_visible_at: float | None
    policy_leak_detected: bool
    cancelled: bool = False


def consume_chat_stream(
    stream: Iterable[Any],
    *,
    content_stream_allowed: bool,
    leak_detector: Callable[[str], bool],
    cancel_requested: Callable[[], bool] | None = None,
    on_thinking: Callable[[str], None] | None = None,
    on_visible_content: Callable[[str], None] | None = None,
    now: Callable[[], float] = time.monotonic,
    guard_chars: int = 96,
    guard_line_chars: int = 32,
    control_prefixes: tuple[str, ...] = (),
    first_chunk_timeout_seconds: float | None = None,
    idle_timeout_seconds: float | None = None,
) -> StreamCapture:
    """Consume Ollama chunks while preserving native streaming semantics.

    Tool calls are accumulated across chunks, model performance fields are kept
    from the latest chunk, and visible prose is guarded until a short prefix is
    available for prompt/policy-leak detection.  The function contains no agent
    policy beyond that guard; callers decide what to do with the captured result.
    """
    tool_calls: list[Any] = []
    full_content = ""
    full_thinking = ""
    perf_stats: dict[str, Any] = {}
    first_token_at: float | None = None
    first_visible_at: float | None = None
    guard_buffer = ""
    guard_released = False
    policy_leak_detected = False
    control_tail = ""
    normalized_control_prefixes = tuple(
        prefix.lower() for prefix in control_prefixes if str(prefix or "")
    )

    def _ambiguous_control_prefix(text: str) -> bool:
        """Return True while the visible prefix could still become a control token.

        Qwen tool XML can be split at arbitrary token/chunk boundaries.  Never
        release ``<tool_`` merely because the initial latency guard expired; wait
        until the prefix is either a complete control marker or normal prose.
        """
        if not normalized_control_prefixes:
            return False
        candidate = str(text or "").lstrip().lower()
        if not candidate:
            return True
        return any(prefix.startswith(candidate) for prefix in normalized_control_prefixes)

    def _safe_visible_piece(text: str) -> str:
        """Return streamable prose while retaining possible split control suffixes.

        This guard remains active *after* ordinary prose has started streaming.
        A late/malformed tool envelope therefore cannot leak parser XML to the UI.
        Only the shortest suffix that might become a control prefix is withheld.
        """
        nonlocal control_tail, policy_leak_detected
        if not normalized_control_prefixes or policy_leak_detected:
            return text if not policy_leak_detected else ""
        combined = control_tail + str(text or "")
        lowered = combined.lower()
        marker_positions = [
            lowered.find(prefix) for prefix in normalized_control_prefixes
            if lowered.find(prefix) >= 0
        ]
        if marker_positions:
            cut = min(marker_positions)
            safe = combined[:cut]
            control_tail = ""
            policy_leak_detected = True
            return safe

        keep = 0
        max_keep = min(
            len(combined),
            max((len(prefix) - 1 for prefix in normalized_control_prefixes), default=0),
        )
        lower_combined = combined.lower()
        for size in range(1, max_keep + 1):
            suffix = lower_combined[-size:]
            if any(prefix.startswith(suffix) for prefix in normalized_control_prefixes):
                keep = size
        if keep:
            safe = combined[:-keep]
            control_tail = combined[-keep:]
            return safe
        control_tail = ""
        return combined

    for chunk in _iter_stream_with_timeouts(
        stream,
        first_chunk_timeout_seconds=first_chunk_timeout_seconds,
        idle_timeout_seconds=idle_timeout_seconds,
    ):
        if cancel_requested is not None and cancel_requested():
            return StreamCapture(
                content=full_content,
                thinking=full_thinking,
                tool_calls=tool_calls,
                perf_stats=perf_stats,
                first_token_at=first_token_at,
                first_visible_at=first_visible_at,
                policy_leak_detected=policy_leak_detected,
                cancelled=True,
            )

        if isinstance(chunk, dict):
            perf_stats = chunk
            chunk_msg = chunk.get("message", {})
        else:
            perf_stats = {
                "done": getattr(chunk, "done", False),
                "prompt_eval_count": getattr(chunk, "prompt_eval_count", None),
                "prompt_eval_cached_count": getattr(chunk, "prompt_eval_cached_count", None),
                "prompt_eval_duration": getattr(chunk, "prompt_eval_duration", None),
                "eval_count": getattr(chunk, "eval_count", None),
                "eval_duration": getattr(chunk, "eval_duration", None),
                "load_duration": getattr(chunk, "load_duration", None),
                "total_duration": getattr(chunk, "total_duration", None),
                "done_reason": getattr(chunk, "done_reason", None),
            }
            chunk_msg = getattr(chunk, "message", {})

        if isinstance(chunk_msg, dict):
            thinking = chunk_msg.get("thinking", "")
            content = chunk_msg.get("content", "")
            calls = chunk_msg.get("tool_calls", [])
        else:
            thinking = getattr(chunk_msg, "thinking", "")
            content = getattr(chunk_msg, "content", "")
            calls = getattr(chunk_msg, "tool_calls", [])

        if first_token_at is None and (thinking or content or calls):
            first_token_at = now()
        if calls:
            tool_calls = merge_stream_tool_calls(tool_calls, calls)
        if thinking:
            full_thinking += str(thinking)
            if on_thinking is not None:
                on_thinking(str(thinking))
        if not content:
            continue

        text = str(content)
        full_content += text
        if not content_stream_allowed or policy_leak_detected:
            continue
        if guard_released:
            safe = _safe_visible_piece(text)
            if safe:
                if first_visible_at is None:
                    first_visible_at = now()
                if on_visible_content is not None:
                    on_visible_content(safe)
            continue

        guard_buffer += text
        if leak_detector(guard_buffer):
            policy_leak_detected = True
            continue
        ready = len(guard_buffer) >= guard_chars or (
            "\n" in guard_buffer and len(guard_buffer) >= guard_line_chars
        )
        if ready and not _ambiguous_control_prefix(guard_buffer):
            safe = _safe_visible_piece(guard_buffer)
            if safe:
                if first_visible_at is None:
                    first_visible_at = now()
                if on_visible_content is not None:
                    on_visible_content(safe)
            guard_buffer = ""
            guard_released = True

    if content_stream_allowed and not policy_leak_detected:
        final_piece = guard_buffer if not guard_released else control_tail
        if final_piece:
            if first_visible_at is None:
                first_visible_at = now()
            if on_visible_content is not None:
                on_visible_content(final_piece)

    return StreamCapture(
        content=full_content,
        thinking=full_thinking,
        tool_calls=tool_calls,
        perf_stats=perf_stats,
        first_token_at=first_token_at,
        first_visible_at=first_visible_at,
        policy_leak_detected=policy_leak_detected,
    )


def tool_result_message(tool_name: str, content: str, *, tool_call_id: str = "") -> dict[str, Any]:
    """Build the logical tool-result record plus a local correlation id.

    Storage keeps a real ``tool`` role for transaction pairing.  At the provider
    boundary :func:`ollama_wire_messages` converts consecutive tool records into
    the exact Qwen user-message ``<tool_response>`` envelope required by the
    supplied Jinja template. ``tool_call_id`` never crosses that boundary.
    """
    message: dict[str, Any] = {
        "role": "tool",
        "content": str(content),
        "tool_name": str(tool_name),
    }
    if tool_call_id:
        message["tool_call_id"] = str(tool_call_id)
    return message


def canonicalize_system_messages(messages: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a provider-safe chat sequence with at most one leading system message.

    Some Ollama chat templates (including the configured Qwen3.8 template)
    reject any ``system`` message that appears after the
    beginning of the conversation.  Harness-owned context used to be emitted as
    a second system message after the current user turn for KV-prefix reuse,
    which made those requests fail before inference.

    Preserve the semantic priority of every harness/system block by merging
    their textual contents, in original order, into one leading system message.
    Non-system messages retain their relative order exactly, so native
    assistant/tool transactions are not disturbed.
    """
    normalized = [dict(message) for message in messages if isinstance(message, dict)]
    system_contents: list[str] = []
    non_system: list[dict[str, Any]] = []
    for message in normalized:
        if str(message.get("role") or "") == "system":
            content = str(message.get("content") or "").strip()
            if content:
                system_contents.append(content)
            continue
        non_system.append(message)

    if not system_contents:
        return non_system
    return [
        {"role": "system", "content": "\n\n".join(system_contents)},
        *non_system,
    ]


def validate_system_message_order(messages: Iterable[dict[str, Any]]) -> None:
    """Raise when a chat sequence contains a non-leading/multiple system role."""
    system_indexes = [
        index
        for index, message in enumerate(messages)
        if isinstance(message, dict) and str(message.get("role") or "") == "system"
    ]
    if system_indexes and system_indexes != [0]:
        raise ValueError(
            "invalid chat protocol: exactly one system message is allowed and it must be first; "
            f"system_indexes={system_indexes}"
        )


def ollama_wire_messages(messages: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return messages containing only fields accepted by Ollama's chat API.

    The harness keeps logical ``tool`` records and ``tool_call_id`` values for
    local transaction pairing.  The Qwen3.8 template, however, consumes tool
    feedback as user content wrapped in ``<tool_response>`` tags.  Translate
    that representation here so every provider-bound request follows the model
    contract regardless of how history is stored. System-role placement is also
    canonicalized here as a final provider-boundary invariant.
    """
    allowed = {"role", "content", "thinking", "images", "tool_calls"}
    wire: list[dict[str, Any]] = []
    pending_tool_results: list[str] = []

    def flush_tool_results() -> None:
        if not pending_tool_results:
            return
        # This is the exact user-message envelope recognized by the supplied
        # Qwen3.8 Jinja template. Group consecutive results exactly as the
        # template's native ``role == 'tool'`` branch would do.
        content = "\n".join(
            f"<tool_response>\n{value}\n</tool_response>"
            for value in pending_tool_results
        )
        wire.append({"role": "user", "content": content})
        pending_tool_results.clear()

    for message in canonicalize_system_messages(messages):
        if str(message.get("role") or "") == "tool":
            pending_tool_results.append(str(message.get("content", "")))
            continue
        flush_tool_results()
        wire.append({key: value for key, value in message.items() if key in allowed})
    flush_tool_results()
    validate_system_message_order(wire)
    return wire


def is_prompt_protocol_error(exc: Exception) -> bool:
    """Return whether an Ollama error is a deterministic prompt/template failure.

    Ollama can surface chat-template/Jinja failures as HTTP 500 responses.  They
    are not transient transport failures and replaying the identical request
    only burns latency and the no-progress budget.
    """
    text = str(exc or "").lower()
    markers = (
        "system message must be at the beginning",
        "system message must be first",
        "chat template",
        "jinja exception",
        "invalid chat protocol",
        "invalid role",
        "roles must alternate",
    )
    return any(marker in text for marker in markers)


def is_retryable_transport_error(exc: Exception) -> bool:
    """Return whether a pre-stream model failure is plausibly transient.

    Ollama raises the built-in ``ConnectionError`` for connect failures and a
    ``ResponseError`` carrying ``status_code`` for HTTP failures. Avoid
    replaying deterministic 4xx request/schema errors; retry timeouts, rate
    limits, and server-side failures.
    """
    if is_prompt_protocol_error(exc):
        return False
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
    tools: list[dict[str, Any]] | None = None,
    think: bool = False,
    on_error: Callable[[Exception], None] | None = None,
) -> bool:
    """Load the model and prime the same Qwen chat-template prefix as real turns.

    The supplied Qwen3.8 template rejects an empty ``messages`` list, so warmup
    is itself a tiny valid chat turn.  When ``tools`` are provided, they are sent
    through Ollama's native ``tools`` request field; the model template then
    renders those definitions into its own ``<tools>`` block exactly as it does
    for foreground requests.

    ``options`` must match interactive turns, especially ``num_ctx``, otherwise
    Ollama can select/reload a different runner and invalidate the measurement.
    """
    base_options = dict(options or {})
    try:
        prime_options = {**base_options, "num_predict": 1}
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": str(system_prompt)})
        messages.append(
            {
                "role": "user",
                "content": "Protocol warmup. Reply with OK and do not call a tool.",
            }
        )
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "options": prime_options,
            "keep_alive": keep_alive,
            "stream": False,
            "think": bool(think),
        }
        if tools:
            kwargs["tools"] = list(tools)
        client.chat(**kwargs)
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
    tools: list[dict[str, Any]] | None = None,
    think: bool = False,
    on_error: Callable[[Exception], None] | None = None,
    on_success: Callable[[], None] | None = None,
) -> threading.Thread:
    """Warm the model on a daemon thread so a frontend can start immediately."""

    def _run() -> None:
        if warm_model(
            client, model, options=options, keep_alive=keep_alive,
            system_prompt=system_prompt, tools=tools, think=think, on_error=on_error,
        ) and on_success is not None:
            on_success()

    thread = threading.Thread(target=_run, name="model-warmup", daemon=True)
    thread.start()
    return thread
