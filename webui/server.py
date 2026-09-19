"""FastAPI composition root for the optional Al Agent Web UI.

Route mechanics live here; filesystem/artifact behavior, theme loading, chat
streaming, and history serialization live in focused sibling modules.
"""
from __future__ import annotations

import hashlib
import json
import mimetypes
import subprocess
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import agent as agent_runtime
from tools import _load_chat_history_from_db, clear_chat_history
from tools.reminders import list_reminders
from tools.runtime import list_jobs
from tools.working_state import WorkingStateStore

from . import workspace_ops as _workspace_ops
from . import chat as _chat_module
from .chat import RUNS, RUNS_LOCK, _run_turn as _chat_run_turn, chat_socket as _chat_socket
from .config import (
    PDF_PREVIEW_DIR,
    PREVIEW_TEXT_BYTES,
    STATIC,
    UPLOAD_DIR,
    WORKSPACE as _DEFAULT_WORKSPACE,
    XRESOURCES_PATH,
)
from .history import _history
from .theme import DEFAULT_THEME, read_xresources_theme

# Mutable compatibility alias: tests/integrations historically monkeypatch
# ``webui.server.WORKSPACE``.  Wrappers synchronize it into workspace_ops.
WORKSPACE = _DEFAULT_WORKSPACE

app = FastAPI(title="Al Agent Web UI", docs_url=None, redoc_url=None)
STATE = WorkingStateStore(limits=agent_runtime.WORKING_STATE_CFG)


@app.middleware("http")
async def security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; connect-src 'self' ws: wss:; img-src 'self' data:; "
        "style-src 'self'; script-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'"
    )
    # This is a localhost-first development UI.  Avoid stale browser assets after
    # rebuilding the sidecar; otherwise CSS/JS changes can appear to be missing.
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, max-age=0"
        response.headers["Pragma"] = "no-cache"
    return response


app.mount("/static", StaticFiles(directory=STATIC), name="static")


def _sync_workspace_root() -> None:
    _workspace_ops.WORKSPACE = Path(WORKSPACE).resolve()
    _workspace_ops.UPLOAD_DIR = (_workspace_ops.WORKSPACE / "uploads").resolve()


def _safe_workspace_path(value: str) -> Path:
    _sync_workspace_root(); return _workspace_ops._safe_workspace_path(value)

def _safe_upload_name(filename: str | None) -> str:
    return _workspace_ops._safe_upload_name(filename)

def _unique_upload_target(directory: Path, filename: str | None, *, prefix: str = "") -> Path:
    _sync_workspace_root(); return _workspace_ops._unique_upload_target(directory, filename, prefix=prefix)

async def _save_upload(file: UploadFile, target: Path) -> int:
    return await _workspace_ops._save_upload(file, target)

def _upload_payload(target: Path, original_name: str | None, size: int) -> dict[str, Any]:
    _sync_workspace_root(); return _workspace_ops._upload_payload(target, original_name, size)

def _preview_kind(path: Path) -> str:
    return _workspace_ops._preview_kind(path)

def _artifact_payload(path: Path) -> dict[str, Any]:
    _sync_workspace_root(); return _workspace_ops._artifact_payload(path)

def _workspace_file_snapshot(limit: int | None = None):
    _sync_workspace_root()
    return _workspace_ops._workspace_file_snapshot() if limit is None else _workspace_ops._workspace_file_snapshot(limit)

def _new_artifacts(previous, current, *, limit=None):
    _sync_workspace_root()
    if limit is None: return _workspace_ops._new_artifacts(previous, current)
    return _workspace_ops._new_artifacts(previous, current, limit=limit)

def _workspace_listing(relative: str = "") -> dict[str, Any]:
    _sync_workspace_root(); return _workspace_ops._workspace_listing(relative)

def _build_user_content(text: str, attachments: list[str]) -> str:
    _sync_workspace_root(); return _workspace_ops._build_user_content(text, attachments)

def _read_xresources_theme(path: Path | None = None) -> dict[str, str]:
    return read_xresources_theme(path)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "main_model": agent_runtime.MODEL,
        "fast_model": agent_runtime.FAST_MODEL,
        "context": agent_runtime.MAX_CTX,
        "working_state": agent_runtime.WORKING_STATE_ENABLED,
    }


@app.get("/api/history")
def history(limit: int = 200) -> list[dict[str, Any]]:
    return _history(limit)


@app.get("/api/theme")
def theme() -> dict[str, Any]:
    return {"colors": _read_xresources_theme(), "source": str(XRESOURCES_PATH)}


@app.get("/api/workspace")
def workspace(path: str = Query(default="", max_length=512)) -> dict[str, Any]:
    return _workspace_listing(path)


@app.delete("/api/history")
def forget() -> dict[str, Any]:
    return {"ok": True, "message": clear_chat_history()}


@app.get("/api/state")
def state() -> dict[str, Any]:
    return STATE.load()


@app.get("/api/jobs")
def jobs(limit: int = 25) -> list[dict[str, Any]]:
    return list_jobs(limit=max(1, min(int(limit), 100)))


@app.get("/api/reminders")
def reminders() -> Any:
    raw = list_reminders()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return []


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
    _sync_workspace_root()
    upload_dir = (_workspace_ops.WORKSPACE / "uploads").resolve()
    upload_dir.mkdir(parents=True, exist_ok=True)
    import uuid
    target = _unique_upload_target(upload_dir, file.filename, prefix=f"{uuid.uuid4().hex[:10]}_")
    size = await _save_upload(file, target)
    return _upload_payload(target, file.filename, size)


@app.post("/api/workspace/upload")
async def workspace_upload(file: UploadFile = File(...), path: str = Query(default="", max_length=512)) -> dict[str, Any]:
    directory = _safe_workspace_path(path or ".")
    if not directory.exists(): raise HTTPException(status_code=404, detail="Workspace destination not found")
    if not directory.is_dir(): raise HTTPException(status_code=400, detail="Workspace destination is not a directory")
    target = _unique_upload_target(directory, file.filename)
    size = await _save_upload(file, target)
    return _upload_payload(target, file.filename, size)


@app.get("/api/files/{path:path}")
def workspace_file(path: str, download: bool = False) -> FileResponse:
    target = _safe_workspace_path(path)
    if not target.is_file(): raise HTTPException(status_code=404, detail="File not found")
    media_type, _ = mimetypes.guess_type(target.name)
    kwargs: dict[str, Any] = {"media_type": media_type or "application/octet-stream"}
    if download:
        kwargs["filename"] = target.name; kwargs["content_disposition_type"] = "attachment"
    return FileResponse(target, **kwargs)


@app.get("/api/pdf-preview/{path:path}")
def workspace_pdf_preview(path: str) -> FileResponse:
    target = _safe_workspace_path(path)
    if not target.is_file() or target.suffix.lower() != ".pdf": raise HTTPException(status_code=404, detail="PDF file not found")
    try:
        stat = target.stat(); fingerprint = hashlib.sha256(f"{target}:{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()[:24]
        PDF_PREVIEW_DIR.mkdir(parents=True, exist_ok=True); output_prefix = PDF_PREVIEW_DIR / fingerprint; output_path = output_prefix.with_suffix(".png")
        if not output_path.is_file():
            result = subprocess.run(["pdftoppm","-f","1","-l","1","-singlefile","-png","-scale-to","1200",str(target),str(output_prefix)],capture_output=True,text=True,timeout=20,check=False)
            if result.returncode != 0 or not output_path.is_file():
                detail=(result.stderr or result.stdout or "pdftoppm failed").strip()[:500]
                raise HTTPException(status_code=500, detail=f"Unable to render PDF preview: {detail}")
        return FileResponse(output_path, media_type="image/png")
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail="PDF preview rendering timed out") from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Unable to render PDF preview: {exc}") from exc


@app.get("/api/preview/{path:path}")
def workspace_preview(path: str) -> dict[str, Any]:
    target = _safe_workspace_path(path)
    if not target.is_file(): raise HTTPException(status_code=404, detail="File not found")
    artifact = _artifact_payload(target)
    if artifact["preview_kind"] not in {"text", "markdown"}: return {**artifact, "content": None, "truncated": False}
    try:
        with target.open("rb") as handle: raw = handle.read(PREVIEW_TEXT_BYTES + 1)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Unable to preview file: {exc}") from exc
    truncated = len(raw) > PREVIEW_TEXT_BYTES; raw = raw[:PREVIEW_TEXT_BYTES]
    return {**artifact, "content": raw.decode("utf-8", errors="replace"), "truncated": truncated}


@app.post("/api/cancel/{turn_id}")
def cancel(turn_id: str) -> dict[str, Any]:
    with RUNS_LOCK: event = RUNS.get(turn_id)
    if not event: return {"ok": False, "message": "Turn is not active"}
    event.set(); return {"ok": True}


async def _run_turn(websocket: WebSocket, payload: dict[str, Any]) -> None:
    # Keep the historical test/integration hook while delegating to chat.py.
    _sync_workspace_root()
    _chat_module._load_chat_history_from_db = _load_chat_history_from_db
    # artifact_created events are emitted by webui.chat after tool-created files.
    return await _chat_run_turn(websocket, payload)


@app.websocket("/ws/chat")
async def chat_socket(websocket: WebSocket) -> None:
    _sync_workspace_root()
    return await _chat_socket(websocket)
