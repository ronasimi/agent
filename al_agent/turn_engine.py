"""Persistence, UI events, and locking around the autonomous action/result loop."""

from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager

from tools.catalog import catalog_snapshot
from tools.conversation_context import get_active_conversation_id
from tools.executor import execute_registered_tool
from tools.memory import _load_chat_history_from_db, store_tool_observation
from tools.runtime import record_monitor_state, set_foreground_turn, utc_now
from tools.working_state import WorkingStateStore

from . import state
from .agent_loop import LoopConfig, LoopStopped, run_loop
from .events import (
    acquire_inference_lock,
    acquire_turn_lock,
    cancel_requested,
    emit_event,
    release_inference_lock,
    release_turn_lock,
)
from .model_traces import record_model_trace, router_trace_callback
from .system1_router import SystemOneRouter
from .prompts import append_and_save, build_system_prompt
from .tool_session import ToolSession


def _is_transport_error_message(message: dict) -> bool:
    if message.get("role") != "assistant":
        return False
    text = str(message.get("content") or "").strip().lower()
    return (
        text.startswith("model request failed:")
        or "model request failed: timed out" in text
        or (text.startswith("turn failed:") and "timed out" in text)
    )


def _history_for_protocol(messages: list[dict], protocol: str) -> list[dict]:
    """Repair incomplete stored transactions without replaying any tool action."""
    start = next(
        (
            i
            for i, m in enumerate(messages)
            if m.get("role") == "user" and not m.get("_runtime")
        ),
        len(messages),
    )
    output = []
    index = start
    while index < len(messages):
        msg = dict(messages[index])
        msg.pop("images", None)
        if _is_transport_error_message(msg):
            index += 1
            continue
        if msg.get("role") not in {"system", "tool"}:
            output.append(msg)
        index += 1
        if msg.get("tool_calls"):
            results = []
            while index < len(messages) and messages[index].get("role") == "tool":
                results.append(dict(messages[index]))
                index += 1
            for call in msg["tool_calls"]:
                fn = call.get("function", {})
                call_id = call.get("id")
                match = next(
                    (
                        i
                        for i, result in enumerate(results)
                        if (call_id and result.get("tool_call_id") == call_id)
                        or (
                            not result.get("tool_call_id")
                            and result.get("tool_name") == fn.get("name")
                        )
                    ),
                    None,
                )
                if match is not None:
                    result = results.pop(match)
                    result["tool_call_id"] = call_id or ""
                    output.append(result)
                else:
                    output.append(
                        {
                            "role": "tool",
                            "tool_name": fn.get("name", "unknown"),
                            "tool_call_id": call.get("id", ""),
                            "content": json.dumps(
                                {
                                    "ok": False,
                                    "outcome_unknown": True,
                                    "error": "Missing result after interrupted turn. Inspect target state before retrying.",
                                }
                            ),
                        }
                    )
    return output


def build_session_system_prompt(session: ToolSession) -> str:
    """Build the exact stable system prefix used for foreground and warmup turns."""
    prompt = build_system_prompt() + "\n" + session.inventory()
    if not state.VISION_SUPPORTS_IMAGES:
        prompt += (
            "\nThis model cannot inspect image pixels. Use file metadata or available text extraction when useful, and state this limit."
        )
    return prompt


def handle_user_turn(
    messages: list[dict],
    user_input: str,
    thinking_enabled: bool,
    *,
    refresh_history: bool = False,
    runtime_overrides: dict | None = None,
) -> None:
    overrides = runtime_overrides or {}
    append_fn = overrides.get("append_and_save", append_and_save)
    get_turn = overrides.get("acquire_turn_lock", acquire_turn_lock)
    free_turn = overrides.get("release_turn_lock", release_turn_lock)
    get_inference = overrides.get("acquire_inference_lock", acquire_inference_lock)
    free_inference = overrides.get("release_inference_lock", release_inference_lock)
    started = time.monotonic()
    token = uuid.uuid4().hex
    turn_lock = None
    work_state = None
    cid = get_active_conversation_id()
    success = False
    routing_engine = None
    session = None
    routing_context_key = "_general"
    routing_tool_status: dict[str, list[bool]] = {}
    cfg = state.AGENT_CFG
    protocol = str(cfg.get("tool_protocol", "qwen_xml"))
    set_foreground_turn(
        token, {"pid": os.getpid(), "conversation_id": cid, "started_at": utc_now()}
    )
    try:
        turn_lock = get_turn()
        if cancel_requested():
            raise LoopStopped("Turn cancelled")
        emit_event(
            "turn_start",
            content=user_input,
            thinking=bool(thinking_enabled),
            queue_wait_ms=(time.monotonic() - started) * 1000,
        )
        if refresh_history:
            # The autonomous loop fits complete recent turns itself. A legacy
            # compaction watermark must not hide history without its summary.
            messages[:] = _load_chat_history_from_db(
                limit=state.RECENT_MESSAGES, include_compacted=True
            )
        messages[:] = [
            {"role": "system", "content": build_system_prompt()},
            *_history_for_protocol(messages, protocol),
        ]
        append_fn(messages, {"role": "user", "content": user_input})
        turn_id = int(messages[-1].get("_db_id") or 0)
        # Registry access is a turn-local snapshot; active tool lists never live
        # in shared module globals or leak into other conversations.
        schemas, functions, metadata = catalog_snapshot()

        def execute(name, arguments):
            return execute_registered_tool(
                name, arguments, binding=(functions[name], metadata[name])
            )

        @contextmanager
        def router_slot():
            handle = get_inference()
            try:
                yield
            finally:
                free_inference(handle)

        routing_engine = SystemOneRouter(
            overrides.get("ROUTER_OLLAMA", state.ROUTER_OLLAMA),
            model=state.ROUTER_MODEL,
            options=state.ROUTER_OPTIONS,
            keep_alive=state.ROUTER_KEEP_ALIVE,
            candidate_count=state.ROUTER_CANDIDATES,
            route_threshold=state.ROUTER_ROUTE_THRESHOLD,
            inference_slot=router_slot,
            prefix_max_bytes=state.ROUTER_PREFIX_MAX_BYTES,
            description_chars=state.ROUTER_DESCRIPTION_CHARS,
            on_metrics=router_trace_callback(
                path=state.MODEL_TRACE_PATH, enabled=state.MODEL_TRACE_ENABLED,
                max_bytes=state.MODEL_TRACE_MAX_BYTES, model=state.ROUTER_MODEL,
                options=state.ROUTER_OPTIONS, conversation_id=cid, turn_id=turn_id,
            ),
        )
        decision = routing_engine.decide(user_input, schemas, metadata)
        routing_context_key = decision.context_key
        for selected_name in decision.selected:
            routing_engine.record(
                selected_name,
                routing_context_key,
                None,
                event_type="route_selected",
                detail=(
                    f"tier={decision.tier}; confidence={decision.confidence:.3f}; "
                    f"choice={decision.raw_choice[:24]}"
                ),
            )
        session = ToolSession(
            schemas,
            execute,
            metadata,
            max_active=int(cfg.get("max_active_tools", 16)),
            max_schema_chars=int(cfg.get("max_tool_schema_chars", 20000)),
            router=routing_engine,
            initial_active=list(decision.selected),
        )
        messages[0]["content"] = build_session_system_prompt(session)
        work_state = WorkingStateStore(
            limits=state.WORKING_STATE_CFG, conversation_id=cid
        )
        work_state.begin_turn(
            turn_id=turn_id,
            objective=user_input,
            rolling_summary="",
            recalled_context="",
            recent_messages=messages[-12:],
            policy_note="Autonomous single-model tool selection",
            tool_schemas=session.schemas,
            requirements=[],
        )

        def append(msg):
            append_fn(messages, msg)

        def emit(kind, **payload):
            if kind == "tool_result":
                work_state.record_tool_result(
                    tool_name=payload["tool"],
                    arguments=payload["arguments"],
                    status=payload["status"],
                    reason="model_selected",
                    result_text=payload["content"],
                    observation_id=payload.get("observation_id", ""),
                )
                work_state.update_tools(session.schemas)
                tool_name = str(payload["tool"])
                tool_ok = str(payload.get("status")) == "ok"
                routing_tool_status.setdefault(tool_name, []).append(tool_ok)
                # Route learning is task-outcome based. A tool can execute
                # successfully and still be the wrong capability, so do not
                # train confidence merely because the primitive returned OK.
            if kind in {"tool_start", "tool_result"}:
                payload["name"] = payload["tool"]
            emit_event(kind, **payload)

        @contextmanager
        def slot():
            handle = get_inference()
            try:
                yield
            finally:
                free_inference(handle)

        def trace(*, call_index, request, capture, error):
            record_model_trace(
                path=state.MODEL_TRACE_PATH,
                enabled=state.MODEL_TRACE_ENABLED,
                max_bytes=state.MODEL_TRACE_MAX_BYTES,
                conversation_id=cid,
                turn_id=turn_id,
                call_index=call_index,
                model=state.MODEL,
                role="all-purpose",
                purpose="autonomous_action",
                thinking_enabled=thinking_enabled,
                messages=request["messages"],
                tools=request.get("tools", session.schemas),
                options=request["options"],
                completion={
                    "content": capture.content,
                    "tool_calls": [
                        x.model_dump() if hasattr(x, "model_dump") else x
                        for x in capture.tool_calls
                    ],
                }
                if capture
                else {},
                metrics=capture.perf_stats if capture else {},
                error=error,
            )

        run_loop(
            messages,
            client=overrides.get("OLLAMA", state.OLLAMA),
            tools=session,
            config=LoopConfig(
                model=state.MODEL,
                options=state.MAIN_OPTIONS,
                protocol=protocol,
                keep_alive=cfg.get("keep_alive", -1),
                max_model_calls=state.MAX_MODEL_CALLS_PER_TURN,
                max_tool_calls=int(cfg.get("max_tool_calls_per_turn", 48)),
                max_calls_per_response=state.MAX_TOOL_CALLS_PER_ITERATION,
                max_no_progress=state.MODEL_NO_PROGRESS_MAX_RETRIES,
                timeout_seconds=state.TURN_HARD_TIMEOUT_SECONDS,
                max_output_chars=state.MAX_TOOL_OUTPUT,
                reserve_tokens=state.RESERVE_TOKENS,
            ),
            append=append,
            emit=emit,
            cancel=cancel_requested,
            inference_slot=slot,
            store_observation=lambda name, text: store_tool_observation(
                name, text, conversation_id=cid
            ),
            trace=trace,
            thinking=thinking_enabled,
            think_supported=bool(cfg.get("supports_thinking", False)),
        )
        success = True
        if routing_engine is not None:
            for tool_name in session.routed_tools:
                outcomes = routing_tool_status.get(tool_name, [])
                if any(outcomes):
                    routing_engine.record(
                        tool_name,
                        routing_context_key,
                        1.0,
                        event_type="task_success",
                    )
    except Exception as exc:  # noqa: BLE001 - UI/persistence boundary must finalize failed turns
        # Persist an honest termination record. Never report a deadline/error as
        # task completion, and never synthesize success from earlier side effects.
        content = (
            str(exc)
            if isinstance(exc, LoopStopped)
            else f"Turn failed: {type(exc).__name__}: {exc}"
        )
        # Transport/runtime failures are UI diagnostics, not conversation. Do
        # not persist them as assistant turns because they poison retries and
        # grow the next model prompt. Existing stored transport errors are also
        # filtered by _history_for_protocol above.
        infrastructure_failure = (
            "model request failed:" in content.lower()
            or "timed out" in content.lower()
            or "timeout" in content.lower()
        )
        if routing_engine is not None:
            if infrastructure_failure:
                for tool_name in getattr(session, "routed_tools", ()):
                    routing_engine.record(
                        tool_name, routing_context_key, None,
                        event_type="infrastructure_failure", detail=content[:240]
                    )
            else:
                routing_related_failure = any(
                    marker in content.lower()
                    for marker in (
                        "repeated tool failures",
                        "tool-call budget",
                        "model-call budget",
                    )
                )
                for tool_name in getattr(session, "routed_tools", ()):
                    outcomes = routing_tool_status.get(tool_name, [])
                    if outcomes and not any(outcomes):
                        routing_engine.record(
                            tool_name, routing_context_key, 0.0,
                            event_type="task_failure", detail=content[:240]
                        )
                    elif outcomes and routing_related_failure:
                        routing_engine.record(
                            tool_name, routing_context_key, 0.35,
                            event_type="task_failure", detail=content[:240]
                        )
        if cancel_requested():
            emit_event("turn_cancelled", reason=content)
        else:
            emit_event("assistant_final", content=content, finalization=True)
            emit_event("error", message=content)
    finally:
        try:
            if work_state is not None:
                work_state.complete_turn(blocked=not success)
        finally:
            try:
                if routing_engine is not None:
                    routing_engine.reset_turn()
            finally:
                try:
                    if turn_lock is not None:
                        free_turn(turn_lock)
                finally:
                    set_foreground_turn(token, None)
                    record_monitor_state("agent.last_interaction", utc_now())
                    emit_event(
                        "turn_end",
                        success=success,
                        elapsed_ms=(time.monotonic() - started) * 1000,
                    )
