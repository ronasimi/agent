"""FastAPI composition root for the Al Agent Web UI.

Route mechanics live here; filesystem/artifact behavior, theme loading, chat
streaming, and history serialization live in focused sibling modules.
"""
from __future__ import annotations

import hashlib
import json
import mimetypes
import re
import subprocess
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlencode, urlparse

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile, WebSocket
from fastapi.responses import FileResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from al_agent import runtime as agent_runtime
from al_agent.slash_commands import list_slash_commands
from tools import (
    _load_chat_history_from_db,
    clear_chat_history,
    conversation_context,
    create_conversation,
    delete_conversation,
    ensure_conversation,
    list_conversations,
    rename_conversation,
)
from tools.reminders import list_reminders
from tools.browser_benchmark import browsergym_available, regression_dashboard
from tools.runtime import cancel_job, get_job, list_jobs
from tools.user_profile import (
    complete_onboarding_profile,
    get_onboarding_state,
    get_profile_image_path,
    reset_onboarding_profile,
)
from tools.working_state import WorkingStateStore

from . import chat as _chat_module
from . import workspace_ops as _workspace_ops
from .chat import RUNS, RUNS_LOCK
from .chat import _run_turn as _chat_run_turn
from .chat import chat_socket as _chat_socket
from .config import (
    PDF_PREVIEW_DIR,
    PREVIEW_TEXT_BYTES,
    STATIC,
    XRESOURCES_PATH,
)
from .config import (
    WORKSPACE as _DEFAULT_WORKSPACE,
)
from .history import _history, _history_export
from .theme import DEFAULT_THEME as DEFAULT_THEME
from .theme import read_xresources_theme

# Mutable compatibility alias: tests/integrations historically monkeypatch
# ``webui.server.WORKSPACE``.  Wrappers synchronize it into workspace_ops.
WORKSPACE = _DEFAULT_WORKSPACE

@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Load and prefix-prime the main model so the first browser turn is fast.

    Best effort and non-blocking: the UI serves normally when the Ollama server
    is unreachable, and the first real turn simply pays the load itself.
    """
    if agent_runtime.WARMUP_ENABLED:
        from al_agent.model_protocol import warm_model_async
        from al_agent.model_residency import schedule_fast_model_prewarm

        def _main_warm_complete() -> None:
            if agent_runtime.WARMUP_FAST_MODEL:
                schedule_fast_model_prewarm("startup")

        warm_model_async(
            agent_runtime.OLLAMA,
            agent_runtime.MODEL,
            options=agent_runtime.MAIN_OPTIONS,
            keep_alive=-1,
            system_prompt=(
                agent_runtime.build_system_prompt() if agent_runtime.WARMUP_PRIME_PREFIX else ""
            ),
            on_success=_main_warm_complete,
            on_error=lambda exc: print(f"[webui]: main-model warm-up skipped: {exc}"),
        )
    yield


app = FastAPI(title="Al Agent Web UI", docs_url=None, redoc_url=None, lifespan=_lifespan)
STATE = WorkingStateStore(limits=agent_runtime.WORKING_STATE_CFG)


@app.middleware("http")
async def security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; connect-src 'self' ws: wss:; img-src 'self' data:; "
        "style-src 'self' https://cdn.jsdelivr.net; font-src 'self' https://cdn.jsdelivr.net data:; "
        "script-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'"
    )
    # This is a localhost-first development UI.  Avoid stale browser assets after
    # rebuilding the application; otherwise CSS/JS changes can appear to be missing.
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

def _document_preview_text(path: Path, *, max_chars: int = PREVIEW_TEXT_BYTES):
    return _workspace_ops._document_preview_text(path, max_chars=max_chars)

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


@app.get("/api/commands")
def commands() -> list[dict[str, Any]]:
    """Return the shared slash-command catalog for Web UI autocomplete."""
    return list_slash_commands()


@app.get("/api/health")
def health() -> dict[str, Any]:
    """Return the configured model roles used by the browser status footer.

    Keep this endpoint limited to stable runtime/configuration fields.  The old
    ``MICRO_MODEL`` role was removed from the harness, but a stale reference
    here made the entire endpoint return HTTP 500 and left the sidebar model
    summary at its placeholder values.
    """
    return {
        "ok": True,
        "main_model": agent_runtime.MODEL,
        "fast_model": agent_runtime.FAST_MODEL,
        "vision_model": agent_runtime.VISION_MODEL,
        "report_model": str(agent_runtime.AGENT_CFG.get("report_model") or ""),
        "context": agent_runtime.MAX_CTX,
        "working_state": agent_runtime.WORKING_STATE_ENABLED,
    }


@app.get("/api/browser-benchmarks")
def browser_benchmarks(limit: int = Query(default=100, ge=1, le=1000)) -> dict[str, Any]:
    """Return P2 UI/browser regression metrics for the local dashboard."""
    data = regression_dashboard(limit=limit)
    data["browsergym"] = browsergym_available()
    return data


@app.get("/api/history")
def history(limit: int = 200, conversation_id: str = "default") -> list[dict[str, Any]]:
    cid = ensure_conversation(conversation_id)
    return _history(limit, cid)


@app.get("/api/history/export", response_class=PlainTextResponse)
def history_export(limit: int = 0, conversation_id: str = "default") -> str:
    cid = ensure_conversation(conversation_id)
    return _history_export(limit, cid)


@app.get("/api/conversations")
def conversations(
    limit: int = Query(default=50, ge=1, le=200),
    active_conversation_id: str | None = Query(default=None, max_length=128),
) -> list[dict[str, Any]]:
    return list_conversations(
        limit=limit,
        active_conversation_id=active_conversation_id,
    )


@app.post("/api/conversations")
def conversation_create(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = payload or {}
    return create_conversation(str(payload.get("title") or ""))


@app.patch("/api/conversations/{conversation_id}")
def conversation_rename(conversation_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    rename_conversation(conversation_id, str(payload.get("title") or ""))
    return {"ok": True}


@app.delete("/api/conversations/{conversation_id}")
def conversation_delete(conversation_id: str) -> dict[str, Any]:
    return {"ok": delete_conversation(conversation_id)}




@app.get("/api/onboarding")
def onboarding_state() -> dict[str, Any]:
    return get_onboarding_state()


@app.post("/api/onboarding")
def onboarding_complete(payload: dict[str, Any]) -> dict[str, Any]:
    interests = payload.get("interests") or []
    if isinstance(interests, str):
        interests = [item.strip() for item in interests.split(",") if item.strip()]
    return complete_onboarding_profile(
        name=str(payload.get("name") or ""),
        role=str(payload.get("role") or ""),
        timezone=str(payload.get("timezone") or agent_runtime.AGENT_CFG.get("timezone", "UTC")),
        location=str(payload.get("location") or ""),
        email=str(payload.get("email") or ""),
        interests=list(interests)[:20],
        response_style=str(payload.get("response_style") or "concise"),
        research_depth=str(payload.get("research_depth") or "balanced"),
        profile_image_path=str(payload.get("profile_image_path") or ""),
        reset=bool(payload.get("reset", False)),
    )


@app.delete("/api/onboarding")
def onboarding_reset() -> dict[str, Any]:
    return reset_onboarding_profile()


def _google_oauth_service():
    # Keep cryptography/OAuth imports off the first-paint path unless the user
    # opens Connections or invokes a Workspace tool.
    from tools.google_workspace_auth import get_google_workspace_oauth

    return get_google_workspace_oauth()


def _raise_google_integration_error(exc: Exception) -> None:
    from tools.credential_store import CredentialStoreError
    from tools.google_workspace_auth import GoogleWorkspaceAuthError

    if isinstance(exc, GoogleWorkspaceAuthError):
        status = 409 if exc.code in {"client_not_configured", "not_connected"} else 400
        raise HTTPException(status_code=status, detail={"code": exc.code, "message": str(exc)}) from exc
    if isinstance(exc, CredentialStoreError):
        raise HTTPException(
            status_code=503,
            detail={"code": "credential_store_unavailable", "message": "The local credential store is unavailable."},
        ) from exc
    raise HTTPException(
        status_code=500,
        detail={"code": "integration_error", "message": "Google Workspace setup could not be completed."},
    ) from exc


def _require_same_origin(request: Request) -> None:
    """Reject browser cross-site writes to the localhost integration API."""
    fetch_site = str(request.headers.get("sec-fetch-site") or "").lower()
    if fetch_site == "cross-site":
        raise HTTPException(status_code=403, detail="Cross-site integration requests are not allowed.")
    origin = str(request.headers.get("origin") or "").strip()
    if not origin:
        return
    parsed = urlparse(origin)
    expected_host = str(request.headers.get("host") or "").lower()
    if parsed.scheme not in {"http", "https"} or parsed.netloc.lower() != expected_host:
        raise HTTPException(status_code=403, detail="Integration request origin does not match this Web UI.")


def _google_callback_redirect(result: str, reason: str = "") -> RedirectResponse:
    safe_result = result if result in {"connected", "error"} else "error"
    safe_reason = re.sub(r"[^a-z0-9_-]", "", str(reason or "").lower())[:64]
    query = {"google_oauth": safe_result}
    if safe_reason:
        query["reason"] = safe_reason
    return RedirectResponse(url=f"/?{urlencode(query)}", status_code=303)


@app.get("/api/integrations/google")
def google_integration_status() -> dict[str, Any]:
    try:
        return _google_oauth_service().status()
    except RuntimeError as exc:
        _raise_google_integration_error(exc)
        raise AssertionError("unreachable")


@app.post("/api/integrations/google/client")
async def google_integration_client(
    request: Request,
    file: Annotated[UploadFile, File()],
) -> dict[str, Any]:
    _require_same_origin(request)
    try:
        raw = await file.read(65_537)
        if len(raw) > 65_536:
            raise HTTPException(status_code=413, detail="OAuth client JSON must be 64 KiB or smaller.")
        return _google_oauth_service().store_client_config(raw)
    except HTTPException:
        raise
    except RuntimeError as exc:
        _raise_google_integration_error(exc)
        raise AssertionError("unreachable")
    finally:
        await file.close()


@app.post("/api/integrations/google/authorize")
def google_integration_authorize(request: Request) -> dict[str, str]:
    _require_same_origin(request)
    try:
        return {"authorization_url": _google_oauth_service().authorization_url()}
    except RuntimeError as exc:
        _raise_google_integration_error(exc)
        raise AssertionError("unreachable")


@app.get("/api/integrations/google/callback")
def google_integration_callback(
    code: str = Query(default="", max_length=8192),
    state: str = Query(default="", max_length=512),
    error: str = Query(default="", max_length=256),
) -> RedirectResponse:
    try:
        service = _google_oauth_service()
        if error:
            service.cancel_authorization(state)
            return _google_callback_redirect("error", "access_denied" if error == "access_denied" else "oauth_error")
        service.complete_authorization(state, code)
        return _google_callback_redirect("connected")
    except RuntimeError as exc:
        reason = str(getattr(exc, "code", "oauth_error"))
        return _google_callback_redirect("error", reason)


@app.delete("/api/integrations/google/connection")
def google_integration_disconnect(request: Request) -> dict[str, Any]:
    _require_same_origin(request)
    try:
        return _google_oauth_service().disconnect()
    except RuntimeError as exc:
        _raise_google_integration_error(exc)
        raise AssertionError("unreachable")


@app.delete("/api/integrations/google/client")
def google_integration_remove_client(request: Request) -> dict[str, Any]:
    _require_same_origin(request)
    try:
        return _google_oauth_service().remove_client_config()
    except RuntimeError as exc:
        _raise_google_integration_error(exc)
        raise AssertionError("unreachable")


@app.get("/api/profile-image")
def profile_image() -> FileResponse:
    path = get_profile_image_path(migrate_legacy=True)
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="No profile image configured")
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "no-store"})


@app.get("/api/theme")
def theme() -> dict[str, Any]:
    return {"colors": _read_xresources_theme(), "source": str(XRESOURCES_PATH)}


@app.get("/api/workspace")
def workspace(path: str = Query(default="", max_length=512)) -> dict[str, Any]:
    return _workspace_listing(path)


@app.delete("/api/history")
def forget(conversation_id: str = "default") -> dict[str, Any]:
    cid = ensure_conversation(conversation_id)
    with conversation_context(cid):
        return {"ok": True, "message": clear_chat_history(cid)}


@app.get("/api/state")
def state(conversation_id: str = "default") -> dict[str, Any]:
    cid = ensure_conversation(conversation_id)
    return WorkingStateStore(limits=agent_runtime.WORKING_STATE_CFG, conversation_id=cid).load()


@app.get("/api/jobs")
def jobs(limit: int = 25) -> list[dict[str, Any]]:
    return list_jobs(limit=max(1, min(int(limit), 100)))


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job_route(job_id: str) -> dict[str, Any]:
    """Cancel one active durable job without waiting for its worker slice."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not cancel_job(job_id):
        raise HTTPException(status_code=409, detail="Job is already terminal")
    return {"ok": True, "job_id": job_id, "status": "cancelled"}


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
    if artifact["preview_kind"] == "document":
        try:
            content, truncated = _document_preview_text(target)
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            raise HTTPException(status_code=422, detail=f"Unable to preview document: {exc}") from exc
        return {**artifact, "content": content, "truncated": truncated}
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
