"""A bounded, single-model action/result loop, independent of Web UI and storage.

The model alone selects tools, their order, arguments, and when to answer.
The harness enforces protocol, types, cancellation, and resource limits.
"""

from __future__ import annotations

import copy
import json
import time
import uuid
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

from tools.context import estimate_messages_tokens, estimate_tokens

from .model_protocol import (
    consume_chat_stream,
    extract_qwen_xml_tool_calls,
    ollama_wire_messages,
    qwen_xml_tool_prelude,
    tool_result_message,
)
from .tool_session import ToolSession


class LoopStopped(RuntimeError):
    pass


@dataclass(frozen=True)
class LoopConfig:
    model: str
    options: dict
    protocol: str = "qwen_xml"
    keep_alive: Any = -1
    max_model_calls: int = 24
    max_tool_calls: int = 48
    max_calls_per_response: int = 6
    max_no_progress: int = 3
    timeout_seconds: float = 600
    max_output_chars: int = 6000
    reserve_tokens: int = 2048
    soft_prompt_tokens: int = 8192
    hard_prompt_tokens: int = 0
    first_byte_timeout_seconds: float = 120
    stream_idle_timeout_seconds: float = 60

    def __post_init__(self):
        if self.protocol not in {"qwen_xml", "native", "json"}:
            raise ValueError("tool_protocol must be qwen_xml, native, or legacy json")
        if not self.model.strip():
            raise ValueError("model must be nonempty")
        for value in (
            self.max_model_calls,
            self.max_tool_calls,
            self.max_calls_per_response,
            self.max_no_progress,
            self.timeout_seconds,
            self.max_output_chars,
            self.first_byte_timeout_seconds,
            self.stream_idle_timeout_seconds,
        ):
            if value <= 0:
                raise ValueError("Loop budgets must be positive")


def action_schema(schemas: list[dict]) -> dict:
    """Schema-constrained actions work even with text-only distilled templates."""
    variants = [
        {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["final"]},
                "answer": {"type": "string", "minLength": 1},
            },
            "required": ["action", "answer"],
            "additionalProperties": False,
        }
    ]
    for schema in schemas:
        fn = schema["function"]
        variants.append(
            {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["tool"]},
                    "name": {"type": "string", "enum": [fn["name"]]},
                    "arguments": fn.get("parameters", {"type": "object"}),
                },
                "required": ["action", "name", "arguments"],
                "additionalProperties": False,
            }
        )
    return {"oneOf": variants}


def _strict_json(text: str) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"Non-finite JSON number: {value}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def decode_response(
    content: str, native_calls: list, protocol: str
) -> tuple[str, list[dict]]:
    if protocol == "json":
        obj = _strict_json(content)
        if not isinstance(obj, dict):
            raise ValueError("Expected one JSON action object")
        if obj.get("action") == "final" and set(obj) == {"action", "answer"}:
            answer = obj["answer"]
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("Final answer must be a nonempty string")
            return answer, []
        if obj.get("action") != "tool" or set(obj) != {"action", "name", "arguments"}:
            raise ValueError(
                "Expected a final answer or a tool action with name and arguments"
            )
        if not isinstance(obj["name"], str) or not isinstance(obj["arguments"], dict):
            raise ValueError("Tool action needs a string name and object arguments")
        return "", [{"name": obj["name"], "arguments": obj["arguments"]}]
    calls = []
    for raw in native_calls:
        raw = raw.model_dump(exclude_none=True) if hasattr(raw, "model_dump") else raw
        if not isinstance(raw, dict) or not isinstance(raw.get("function"), dict):
            raise TypeError("Malformed native tool call")
        fn = raw["function"]
        args = fn.get("arguments", {})
        if isinstance(args, str):
            args = _strict_json(args)
        if not isinstance(fn.get("name"), str) or not isinstance(args, dict):
            raise TypeError("Native tool call needs a string name and object arguments")
        calls.append({"name": fn["name"], "arguments": args})
    if protocol == "qwen_xml":
        xml_calls, xml_errors = extract_qwen_xml_tool_calls(content)
        if xml_errors:
            raise ValueError("; ".join(xml_errors))
        parsed_xml = [
            {
                "name": item["function"]["name"],
                "arguments": item["function"].get("arguments", {}),
            }
            for item in xml_calls
        ]
        if calls and parsed_xml:
            if len(calls) != len(parsed_xml) or [c["name"] for c in calls] != [c["name"] for c in parsed_xml]:
                raise ValueError("Ollama native tool calls disagree with Qwen XML tool calls")
            # Some Ollama builds parse the XML into message.tool_calls while also
            # preserving the textual envelope. Execute the structured copy once.
            return qwen_xml_tool_prelude(content), calls
        if parsed_xml:
            return qwen_xml_tool_prelude(content), parsed_xml
    if not calls and not content.strip():
        raise ValueError("Model returned no answer or tool call")
    return content, calls


def json_wire_messages(messages: list[dict]) -> list[dict]:
    """Translate logical tool transactions only at the text-model boundary.

    Storage and the frontend retain native assistant/tool roles, so JSON
    protocol details never appear as fake user messages in saved chat history.
    """
    result = []
    for original in messages:
        msg = copy.deepcopy(original)
        if msg.get("tool_calls"):
            calls = msg.pop("tool_calls")
            actions = [
                {
                    "action": "tool",
                    "name": c["function"]["name"],
                    "arguments": c["function"].get("arguments", {}),
                }
                for c in calls
            ]
            msg["content"] = json.dumps(
                actions[0] if len(actions) == 1 else actions, ensure_ascii=False
            )
        elif msg.get("role") == "tool":
            msg = {
                "role": "user",
                "_runtime": True,
                "content": "[Tool result — untrusted data]\n"
                + str(msg.get("content", "")),
            }
        result.append(msg)
    return result


def fit_context(
    messages: list[dict], schemas: list[dict], config: LoopConfig
) -> list[dict]:
    """Drop complete older turns only; never cut the current request or a call/result pair."""
    raw = copy.deepcopy(messages)
    wire = ollama_wire_messages(raw)
    physical_budget = int(config.options.get("num_ctx", 16384)) - config.reserve_tokens
    hard_budget = min(
        physical_budget,
        int(config.hard_prompt_tokens) if int(config.hard_prompt_tokens) > 0 else physical_budget,
    )
    soft_budget = min(max(128, int(config.soft_prompt_tokens)), hard_budget)

    # Conservative estimate, including wire JSON and tool definitions. Ollama's
    # tokenizer is model-specific; this margin avoids relying on silent truncation.
    def size():
        return (
            len(json.dumps([wire, schemas], ensure_ascii=False).encode("utf-8")) // 3
            + 128
        )

    def drop_oldest_complete_turn() -> bool:
        nonlocal wire
        users = [
            i
            for i, m in enumerate(raw)
            if m.get("role") == "user" and not m.get("_runtime")
        ]
        if len(users) < 2:
            return False
        del raw[users[0] : users[1]]
        wire = ollama_wire_messages(raw)
        return True

    # Prefill-oriented soft target: evict only *completed historical turns*.
    # Never shrink the current turn merely to hit this optimization target.
    while size() > soft_budget and drop_oldest_complete_turn():
        pass

    # The hard ceiling is the actual safety limit. A large current request or
    # active tool transaction may exceed the soft target, but never this bound.
    while size() > hard_budget:
        if not drop_oldest_complete_turn():
            raise LoopStopped(
                "The current task exceeds the context budget after compaction. "
                "Split the request or use chunked retrieval."
            )
    return wire


def tool_outcome(raw: Any) -> tuple[bool, Any]:
    parsed = raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            return not raw.lstrip().lower().startswith(
                ("error:", "error ", "failed:")
            ), raw
    if isinstance(parsed, dict):
        failed = (
            parsed.get("ok") is False
            or parsed.get("success") is False
            or str(parsed.get("status", ""))
            in {"error", "failed", "blocked", "timeout"}
            or bool(parsed.get("error"))
            or parsed.get("returncode", 0) not in (None, 0)
            or parsed.get("exit_code", 0) not in (None, 0)
        )
        return not failed, parsed
    return True, parsed


def _prompt_telemetry(messages: list[dict], schemas: list[dict]) -> dict[str, Any]:
    """Return cheap model-call telemetry before bytes are sent to Ollama."""

    schema_json = json.dumps(schemas, ensure_ascii=False, separators=(",", ":"))
    message_chars = sum(len(str(message.get("content") or "")) for message in messages)
    tool_call_chars = sum(
        len(json.dumps(message.get("tool_calls"), ensure_ascii=False, default=str))
        for message in messages
        if message.get("tool_calls")
    )
    message_tokens = estimate_messages_tokens(messages)
    schema_tokens = estimate_tokens(schema_json) if schema_json else 0
    return {
        "message_count": len(messages),
        "message_chars": message_chars,
        "tool_call_chars": tool_call_chars,
        "schema_count": len(schemas),
        "schema_chars": len(schema_json),
        "estimated_message_tokens": message_tokens,
        "estimated_schema_tokens": schema_tokens,
        "estimated_input_tokens": message_tokens + schema_tokens,
        "historical_tool_messages": sum(1 for message in messages if message.get("role") == "tool"),
        "historical_tool_call_messages": sum(1 for message in messages if message.get("tool_calls")),
    }


def run_loop(
    messages: list[dict],
    *,
    client: Any,
    tools: ToolSession,
    config: LoopConfig,
    append: Callable[[dict], None],
    emit: Callable[..., None],
    cancel: Callable[[], bool] = lambda: False,
    inference_slot: Callable = nullcontext,
    store_observation: Callable[[str, str], str] = lambda name, text: "",
    trace: Callable[..., None] = lambda **kwargs: None,
    thinking: bool = False,
    think_supported: bool = False,
    clock: Callable[[], float] = time.monotonic,
) -> str:
    deadline = clock() + config.timeout_seconds
    tool_count = 0
    no_progress = 0
    # Work on a separate model transcript; storage keeps the unabridged record.
    transcript = copy.deepcopy(messages)
    exchanges: list[list[dict]] = []

    def check():
        if cancel():
            raise LoopStopped("Turn cancelled")
        if clock() >= deadline:
            raise LoopStopped("The turn time budget was reached")

    def record(message):
        append(message)
        transcript.append(copy.deepcopy(message))

    def feedback(text):
        # A protocol observation is data. Never add late system messages.
        transcript.append(
            {
                "role": "user",
                "content": "[Runtime protocol feedback] " + text,
                "_runtime": True,
            }
        )

    for call_index in range(1, config.max_model_calls + 1):
        check()
        schemas = copy.deepcopy(tools.schemas)
        current = (
            json_wire_messages(transcript)
            if config.protocol == "json"
            else copy.deepcopy(transcript)
        )
        if config.protocol == "json":
            current[0]["content"] += (
                '\nReturn exactly one JSON object: {"action":"final","answer":"..."} or '
                '{"action":"tool","name":"...","arguments":{...}}. '
                "Only use the following loaded schemas. Do not put actions inside answer text.\n"
                + json.dumps(schemas, ensure_ascii=False, separators=(",", ":"))
            )
        active_schemas = schemas if config.protocol in {"native", "qwen_xml"} else []
        before_telemetry = _prompt_telemetry(current, active_schemas)
        wire = fit_context(current, active_schemas, config)
        telemetry = _prompt_telemetry(wire, active_schemas)
        telemetry.update({
            "soft_prompt_tokens": int(config.soft_prompt_tokens),
            "hard_prompt_tokens": (
                int(config.hard_prompt_tokens)
                if int(config.hard_prompt_tokens) > 0
                else int(config.options.get("num_ctx", 16384)) - int(config.reserve_tokens)
            ),
            "prefill_compaction_removed_messages": max(
                0, before_telemetry["message_count"] - telemetry["message_count"]
            ),
            "prefill_compaction_removed_estimated_tokens": max(
                0, before_telemetry["estimated_input_tokens"] - telemetry["estimated_input_tokens"]
            ),
        })
        request = {
            "model": config.model,
            "messages": wire,
            "options": dict(config.options),
            "keep_alive": config.keep_alive,
            "stream": True,
        }
        if think_supported:
            request["think"] = bool(thinking)
        if config.protocol == "json":
            request["format"] = action_schema(schemas)
        else:
            request["tools"] = schemas
        emit(
            "model_start", model=config.model, call_index=call_index,
            prompt_telemetry=telemetry,
        )
        capture = None
        request_started_at = clock()
        trace_request = {**request, "prompt_telemetry": telemetry}
        try:
            with inference_slot():
                check()
                stream = client.chat(**request)
                try:
                    capture = consume_chat_stream(
                        stream,
                        # Final-answer prose should reach the Web UI as soon as it
                        # is generated.  The protocol gate keeps Qwen control XML
                        # off the visible stream even when <tool_call> is split
                        # across chunks or appears late in malformed output.
                        # Thinking uses its own event stream and is never delayed by
                        # this content gate.
                        content_stream_allowed=True,
                        leak_detector=(
                            (lambda text: "<tool_call" in str(text).lower())
                            if config.protocol == "qwen_xml"
                            else (lambda _: False)
                        ),
                        cancel_requested=lambda: cancel() or clock() >= deadline,
                        on_thinking=(lambda text: emit("thinking_delta", content=text))
                        if thinking
                        else None,
                        on_visible_content=lambda text: emit(
                            "assistant_delta", content=text
                        ),
                        # Qwen's opening <tool_call> marker is only 11 chars.
                        # A 16-char guard prevents XML leakage while keeping the
                        # first visible prose latency close to native streaming.
                        guard_chars=16,
                        guard_line_chars=8,
                        control_prefixes=("<tool_call>",)
                        if config.protocol == "qwen_xml"
                        else (),
                        first_chunk_timeout_seconds=config.first_byte_timeout_seconds,
                        idle_timeout_seconds=config.stream_idle_timeout_seconds,
                    )
                finally:
                    close = getattr(stream, "close", None)
                    if close:
                        close()
            if capture is not None:
                if capture.first_token_at is not None:
                    capture.perf_stats["harness_first_token_ms"] = round(
                        (capture.first_token_at - request_started_at) * 1000, 3
                    )
                if capture.first_visible_at is not None:
                    capture.perf_stats["harness_first_visible_ms"] = round(
                        (capture.first_visible_at - request_started_at) * 1000, 3
                    )
                capture.perf_stats["harness_prompt_estimated_tokens"] = telemetry["estimated_input_tokens"]
                capture.perf_stats["harness_schema_chars"] = telemetry["schema_chars"]
            check()
            if capture.cancelled:
                raise LoopStopped("Turn cancelled")
            if not capture.perf_stats.get("done"):
                raise RuntimeError(
                    "Model stream ended before its completion marker; no tools were executed"
                )
            if capture.perf_stats.get("done_reason") == "length":
                raise ValueError(
                    "Model reached its output token limit; return a shorter action or answer"
                )
            answer, calls = decode_response(
                capture.content, capture.tool_calls, config.protocol
            )
        except LoopStopped:
            raise
        except (ValueError, TypeError) as exc:
            no_progress += 1
            trace(
                call_index=call_index, request=trace_request, capture=capture, error=str(exc)
            )
            if no_progress >= config.max_no_progress:
                raise LoopStopped(
                    f"Model could not produce a valid action: {exc}"
                ) from exc
            feedback(str(exc))
            continue
        except Exception as exc:
            # Never replay a partial stream, promote thinking to an answer, or
            # infer a tool invocation from arbitrary prose/XML.
            trace(
                call_index=call_index, request=trace_request, capture=capture, error=str(exc)
            )
            raise LoopStopped(f"Model request failed: {exc}") from exc
        trace(call_index=call_index, request=trace_request, capture=capture, error="")
        if calls and capture is not None and capture.first_visible_at is not None:
            # A strict Qwen tool turn should contain only XML tool-call blocks,
            # so normally nothing was made visible.  If the model violated that
            # rule and emitted prose before a tool call, retract the provisional
            # stream before showing tool activity.
            emit("assistant_reset")
        if not calls:
            blocker = tools.finalization_blocker()
            if blocker:
                no_progress += 1
                if no_progress >= config.max_no_progress:
                    raise LoopStopped(blocker)
                feedback(blocker)
                continue
            record({"role": "assistant", "content": answer})
            emit("assistant_final", content=answer, finalization=False)
            return answer
        if (
            len(calls) > config.max_calls_per_response
            or tool_count + len(calls) > config.max_tool_calls
        ):
            raise LoopStopped(
                "The tool-call budget was reached; this batch was not executed"
            )
        # Reject a malformed batch before any side effects. The model may repair
        # it, but the harness never substitutes another tool or guesses arguments.
        try:
            for call in calls:
                call["arguments"] = tools.validate(call["name"], call["arguments"])
        except (ValueError, TypeError) as exc:
            no_progress += 1
            if no_progress >= config.max_no_progress:
                raise LoopStopped(f"Repeated invalid tool calls: {exc}") from exc
            feedback(str(exc))
            continue

        ids = [uuid.uuid4().hex for _ in calls]
        assistant = {
            "role": "assistant",
            "content": answer,
            "tool_calls": [
                {
                    "id": cid,
                    "type": "function",
                    "function": {"name": c["name"], "arguments": c["arguments"]},
                }
                for cid, c in zip(ids, calls)
            ],
        }
        record(assistant)
        exchange = [transcript[-1]]
        failed_batch = False
        for cid, call in zip(ids, calls):
            name, arguments = call["name"], call["arguments"]
            stop = cancel() or clock() >= deadline
            if failed_batch or stop:
                raw = {
                    "ok": False,
                    "error": "Not executed: preceding call failed or turn stopped",
                    "skipped": True,
                }
            else:
                tool_count += 1
                emit(
                    "tool_start",
                    tool=name,
                    name=name,
                    arguments=arguments,
                    tool_call_id=cid,
                )
                try:
                    raw = tools.invoke(name, arguments)
                except Exception as exc:  # noqa: BLE001 - tool boundary returns errors to the model
                    raw = {
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "outcome_unknown": not tools.metadata.get(name, {}).get(
                            "readonly", False
                        ),
                    }
            media = []
            if isinstance(raw, dict) and raw.get("__agent_media_result__") is True:
                media = list(raw.get("images") or [])
                raw = str(raw.get("content", ""))
            ok, value = tool_outcome(raw)
            tools.record_outcome(name, ok)
            failed_batch = failed_batch or not ok
            text = (
                value
                if isinstance(value, str)
                else json.dumps(value, ensure_ascii=False, default=str)
            )
            observation = store_observation(name, text)
            bounded = text
            if len(text) > config.max_output_chars:
                bounded = (
                    text[: config.max_output_chars]
                    + f"\n[Truncated. Full observation: {observation or 'unavailable'}]"
                )
            packet = {
                "tool": name,
                "ok": ok,
                "result": bounded,
                "observation_id": observation,
            }
            encoded = json.dumps(packet, ensure_ascii=False)
            result_msg = tool_result_message(name, encoded, tool_call_id=cid)
            if media:
                result_msg["media"] = media
            record(result_msg)
            exchange.append(transcript[-1])
            emit(
                "tool_result",
                tool=name,
                name=name,
                arguments=arguments,
                content=bounded,
                result=bounded,
                success=ok,
                status="ok" if ok else "error",
                observation_id=observation,
                tool_call_id=cid,
                media=media,
            )
        exchanges.append(exchange)
        # Bound older results while retaining call/result pairing and a durable
        # retrieval handle. Keep the latest two exchanges in full.
        for old in exchanges[:-2]:
            for msg in old[1:]:
                if len(msg.get("content", "")) > 900:
                    packet = json.loads(msg["content"])
                    packet["result"] = (
                        packet["result"][:300]
                        + " [Earlier result compacted; use observation_id to retrieve.]"
                    )
                    msg["content"] = json.dumps(packet, ensure_ascii=False)
        no_progress = no_progress + 1 if failed_batch else 0
        check()
        if no_progress >= config.max_no_progress:
            raise LoopStopped("Repeated tool failures exhausted the recovery budget")
    raise LoopStopped(
        "The model-call budget was reached; completed tool results remain in history"
    )
