"""WebSocket chat orchestration for the Web UI."""
from __future__ import annotations

import asyncio
import inspect
import threading
import uuid
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from al_agent import runtime as agent_runtime
from al_agent.slash_commands import execute_slash_command
from tools import conversation_context, ensure_conversation, normalize_conversation_id

from .config import ARTIFACT_MAX_PER_TURN
from .workspace_ops import _build_user_content, _new_artifacts, _workspace_file_snapshot

RUNS: dict[str, threading.Event] = {}
RUNS_LOCK = threading.Lock()

# Thinking is opt-in per turn. When the WebUI Think checkbox is disabled, keep
# any accidental/empty reasoning fragments off the wire. When it is enabled,
# forward thinking_delta events so the browser can render the model's reasoning
# live in a dedicated, non-persistent stream.
def _should_forward_event(event: dict[str, Any], *, thinking_enabled: bool = False) -> bool:
    event_type = str(event.get("type") or "")
    return event_type != "thinking_delta" or bool(thinking_enabled)

async def _run_turn(websocket: WebSocket, payload: dict[str, Any]) -> None:
    turn_id = str(payload.get("turn_id") or uuid.uuid4().hex)
    conversation_id = normalize_conversation_id(payload.get("conversation_id"))
    ensure_conversation(conversation_id)
    raw_content = str(payload.get("content") or "").strip()
    text = _build_user_content(raw_content, list(payload.get("attachments") or []))
    if not text:
        await websocket.send_json({"type": "error", "turn_id": turn_id, "message": "Message is empty"})
        return
    thinking = bool(payload.get("thinking", agent_runtime.THINKING_DEFAULT))

    # Slash commands are harness/frontend control operations, never model input.
    # Route them before allocating a cancellation slot, inference lock, history
    # snapshot, or Ollama request. Unknown slash commands are also consumed here
    # so malformed control input cannot leak into the model prompt.
    if raw_content.startswith("/"):
        result = await asyncio.to_thread(
            execute_slash_command,
            raw_content,
            conversation_id=conversation_id,
            thinking_enabled=thinking,
        )
        await websocket.send_json({
            "type": "accepted", "turn_id": turn_id,
            "conversation_id": conversation_id, "content": raw_content,
            "command": True,
        })
        event = result.event()
        event["turn_id"] = turn_id
        await websocket.send_json(event)
        await websocket.send_json({"type": "history_refresh", "turn_id": turn_id})
        await websocket.send_json({"type": "turn_end", "turn_id": turn_id})
        return

    cancel_event = threading.Event()
    with RUNS_LOCK:
        RUNS[turn_id] = cancel_event

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    emitted_artifacts: set[str] = set()

    # Acknowledge the turn before any filesystem scan or model work. This keeps
    # the composer/Stop control responsive even when the workspace contains many
    # files or the local model is queued behind another inference request.
    await websocket.send_json({"type": "accepted", "turn_id": turn_id, "conversation_id": conversation_id, "content": text})
    artifact_snapshot = await asyncio.to_thread(_workspace_file_snapshot)

    def collect_new_artifacts() -> list[dict[str, Any]]:
        nonlocal artifact_snapshot
        events: list[dict[str, Any]] = []
        if len(emitted_artifacts) >= ARTIFACT_MAX_PER_TURN:
            return events
        current = _workspace_file_snapshot()
        remaining = ARTIFACT_MAX_PER_TURN - len(emitted_artifacts)
        for artifact in _new_artifacts(artifact_snapshot, current, limit=remaining):
            relative = str(artifact.get("relative") or "")
            if not relative or relative in emitted_artifacts:
                continue
            emitted_artifacts.add(relative)
            events.append({
                "type": "artifact_created",
                "timestamp": agent_runtime.utc_now(),
                "turn_id": turn_id,
                "artifact": artifact,
            })
        artifact_snapshot = current
        return events

    def queue_new_artifacts() -> None:
        for event in collect_new_artifacts():
            loop.call_soon_threadsafe(queue.put_nowait, event)

    def sink(event: dict[str, Any]) -> None:
        event = {**event, "turn_id": turn_id}
        if _should_forward_event(event, thinking_enabled=thinking):
            loop.call_soon_threadsafe(queue.put_nowait, event)
        if event.get("type") == "tool_result":
            queue_new_artifacts()

    def work() -> None:
        # The history snapshot is refreshed after the per-conversation turn lock
        # is acquired. That preserves same-chat ordering without occupying the
        # global Ollama queue during history/preflight/tool I/O.
        messages = [{"role": "system", "content": agent_runtime.build_system_prompt()}]
        with conversation_context(conversation_id), agent_runtime.frontend_event_context(sink, cancel_event):
            handler = agent_runtime.handle_user_turn
            if "refresh_history" in inspect.signature(handler).parameters:
                handler(messages, text, thinking, refresh_history=True)
            else:  # compatibility for embedders/tests with the legacy callback shape
                handler(messages, text, thinking)

    if cancel_event.is_set():
        await websocket.send_json({"type": "turn_cancelled", "turn_id": turn_id})
        await websocket.send_json({"type": "turn_end", "turn_id": turn_id})
        with RUNS_LOCK:
            RUNS.pop(turn_id, None)
        return

    task = asyncio.create_task(asyncio.to_thread(work))
    try:
        while True:
            if task.done() and queue.empty():
                break
            try:
                event = await asyncio.wait_for(queue.get(), timeout=0.25)
                await websocket.send_json(event)
            except TimeoutError:
                pass
        exc = task.exception() if task.done() else None
        if exc:
            await websocket.send_json({"type": "error", "turn_id": turn_id, "message": str(exc)})
        # The final scan can touch many directory entries; keep it off the
        # event loop just like inference. No frontend navigation/input work waits
        # on this scan.
        for event in await asyncio.to_thread(collect_new_artifacts):
            queue.put_nowait(event)
        while not queue.empty():
            await websocket.send_json(queue.get_nowait())
        await websocket.send_json({"type": "history_refresh", "turn_id": turn_id})
    finally:
        with RUNS_LOCK:
            RUNS.pop(turn_id, None)

async def chat_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    active: asyncio.Task | None = None
    try:
        while True:
            payload = await websocket.receive_json()
            action = str(payload.get("type") or "message")
            if action == "cancel":
                turn_id = str(payload.get("turn_id") or "")
                with RUNS_LOCK:
                    event = RUNS.get(turn_id)
                if event:
                    event.set()
                continue
            if action != "message":
                continue
            if active and not active.done():
                if bool(payload.get("defer_until_idle", False)):
                    await asyncio.gather(active, return_exceptions=True)
                else:
                    await websocket.send_json({"type": "error", "message": "This browser session already has an active turn"})
                    continue
            active = asyncio.create_task(_run_turn(websocket, payload))
    except WebSocketDisconnect:
        if active and not active.done():
            # The turn may continue safely unless the browser explicitly stopped it.
            pass
