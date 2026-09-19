"""WebSocket chat orchestration for the Web UI."""
from __future__ import annotations
import asyncio, threading, uuid
from typing import Any
from fastapi import WebSocket, WebSocketDisconnect
import agent as agent_runtime
from tools import _load_chat_history_from_db
from .config import ARTIFACT_MAX_PER_TURN
from .workspace_ops import _build_user_content, _new_artifacts, _workspace_file_snapshot

RUNS: dict[str, threading.Event] = {}
RUNS_LOCK = threading.Lock()

async def _run_turn(websocket: WebSocket, payload: dict[str, Any]) -> None:
    turn_id = str(payload.get("turn_id") or uuid.uuid4().hex)
    text = _build_user_content(payload.get("content", ""), list(payload.get("attachments") or []))
    if not text:
        await websocket.send_json({"type": "error", "turn_id": turn_id, "message": "Message is empty"})
        return
    thinking = bool(payload.get("thinking", agent_runtime.THINKING_DEFAULT))
    cancel_event = threading.Event()
    with RUNS_LOCK:
        RUNS[turn_id] = cancel_event

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    artifact_snapshot = _workspace_file_snapshot()
    emitted_artifacts: set[str] = set()

    def queue_new_artifacts() -> None:
        nonlocal artifact_snapshot
        if len(emitted_artifacts) >= ARTIFACT_MAX_PER_TURN:
            return
        current = _workspace_file_snapshot()
        remaining = ARTIFACT_MAX_PER_TURN - len(emitted_artifacts)
        for artifact in _new_artifacts(artifact_snapshot, current, limit=remaining):
            relative = str(artifact.get("relative") or "")
            if not relative or relative in emitted_artifacts:
                continue
            emitted_artifacts.add(relative)
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {"type": "artifact_created", "timestamp": agent_runtime.utc_now(), "turn_id": turn_id, "artifact": artifact},
            )
        artifact_snapshot = current

    def sink(event: dict[str, Any]) -> None:
        event = {**event, "turn_id": turn_id}
        loop.call_soon_threadsafe(queue.put_nowait, event)
        if event.get("type") == "tool_result":
            queue_new_artifacts()

    def work() -> None:
        messages = [{"role": "system", "content": agent_runtime.build_system_prompt()}] + _load_chat_history_from_db(limit=100)
        with agent_runtime.frontend_event_context(sink, cancel_event):
            agent_runtime.handle_user_turn(messages, text, thinking)

    task = asyncio.create_task(asyncio.to_thread(work))
    await websocket.send_json({"type": "accepted", "turn_id": turn_id, "content": text})
    try:
        while True:
            if task.done() and queue.empty():
                break
            try:
                event = await asyncio.wait_for(queue.get(), timeout=0.25)
                await websocket.send_json(event)
            except asyncio.TimeoutError:
                pass
        exc = task.exception() if task.done() else None
        if exc:
            await websocket.send_json({"type": "error", "turn_id": turn_id, "message": str(exc)})
        queue_new_artifacts()
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
                await websocket.send_json({"type": "error", "message": "This browser session already has an active turn"})
                continue
            active = asyncio.create_task(_run_turn(websocket, payload))
    except WebSocketDisconnect:
        if active and not active.done():
            # The turn may continue safely unless the browser explicitly stopped it.
            pass
