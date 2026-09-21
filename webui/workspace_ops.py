"""Workspace browsing, upload, attachment, and artifact helpers."""
from __future__ import annotations

import mimetypes
import os
import re
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from fastapi import HTTPException, UploadFile

from .config import (
    ALLOWED_MEDIA_EXT,
    ALLOWED_TEXT_EXT,
    ARTIFACT_IGNORE_RELATIVE,
    ARTIFACT_MAX_PER_TURN,
    ARTIFACT_SCAN_LIMIT,
    AUDIO_PREVIEW_EXT,
    DOCUMENT_PREVIEW_EXT,
    IMAGE_PREVIEW_EXT,
    MARKDOWN_EXT,
    MAX_UPLOAD_BYTES,
    PREVIEW_TEXT_BYTES,
    TEXT_PREVIEW_EXT,
    VIDEO_PREVIEW_EXT,
    WORKSPACE,
    WORKSPACE_LIST_LIMIT,
)


def _safe_workspace_path(value: str) -> Path:
    raw = str(value or "").strip()
    if raw.startswith("/app/workspace/"):
        candidate = WORKSPACE / raw[len("/app/workspace/"):]
    else:
        candidate = WORKSPACE / raw.lstrip("/")
    resolved = candidate.resolve()
    if os.path.commonpath([str(WORKSPACE), str(resolved)]) != str(WORKSPACE):
        raise HTTPException(status_code=403, detail="Path is outside the workspace")
    return resolved

def _safe_upload_name(filename: str | None) -> str:
    original = Path(filename or "attachment").name
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(original).stem)[:100] or "attachment"
    suffix = re.sub(r"[^A-Za-z0-9.]", "", Path(original).suffix)[:16]
    return f"{stem}{suffix}"

def _unique_upload_target(directory: Path, filename: str | None, *, prefix: str = "") -> Path:
    if not directory.is_dir():
        raise HTTPException(status_code=400, detail="Upload destination is not a directory")
    safe_name = _safe_upload_name(filename)
    stem = Path(safe_name).stem
    suffix = Path(safe_name).suffix
    candidate = directory / f"{prefix}{safe_name}"
    counter = 2
    while candidate.exists():
        candidate = directory / f"{prefix}{stem}_{counter}{suffix}"
        counter += 1
    resolved = candidate.resolve()
    if os.path.commonpath([str(WORKSPACE), str(resolved)]) != str(WORKSPACE):
        raise HTTPException(status_code=403, detail="Upload path is outside the workspace")
    return resolved

async def _save_upload(file: UploadFile, target: Path) -> int:
    size = 0
    try:
        with target.open("wb") as handle:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Upload exceeds configured size limit")
                handle.write(chunk)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    return size

def _upload_payload(target: Path, original_name: str | None, size: int) -> dict[str, Any]:
    suffix = target.suffix.lower()
    return {
        "ok": True,
        "name": Path(original_name or target.name).name,
        "stored_name": target.name,
        "path": "/app/workspace/" + str(target.relative_to(WORKSPACE)),
        "relative": str(target.relative_to(WORKSPACE)),
        "size": size,
        "media": suffix in ALLOWED_MEDIA_EXT,
        "text": suffix in ALLOWED_TEXT_EXT,
    }

def _preview_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    media_type, _ = mimetypes.guess_type(path.name)
    media_type = media_type or "application/octet-stream"
    if suffix in MARKDOWN_EXT:
        return "markdown"
    if suffix in IMAGE_PREVIEW_EXT:
        return "image"
    if suffix == ".pdf":
        return "pdf"
    if suffix in AUDIO_PREVIEW_EXT or media_type.startswith("audio/"):
        return "audio"
    if suffix in VIDEO_PREVIEW_EXT or media_type.startswith("video/"):
        return "video"
    if suffix in DOCUMENT_PREVIEW_EXT:
        return "document"
    if suffix in TEXT_PREVIEW_EXT or media_type.startswith("text/") or media_type in {
        "application/json", "application/xml", "application/javascript",
    }:
        return "text"
    return "download"

def _xml_text(raw: bytes, *, max_chars: int) -> str:
    """Extract bounded visible text from a bounded Office/OpenDocument XML part."""
    try:
        root = ElementTree.fromstring(raw)
    except ElementTree.ParseError:
        return ""
    chunks: list[str] = []
    total = 0
    for node in root.iter():
        value = str(node.text or "").strip()
        if not value:
            continue
        remaining = max_chars - total
        if remaining <= 0:
            break
        clipped = value[:remaining]
        chunks.append(clipped)
        total += len(clipped) + 1
    return "\n".join(chunks)

def _document_preview_text(
    path: Path, *, max_chars: int = PREVIEW_TEXT_BYTES
) -> tuple[str, bool]:
    """Return a safe, bounded text preview for modern local document formats."""
    limit = max(1024, min(int(max_chars), PREVIEW_TEXT_BYTES))
    suffix = path.suffix.lower()
    if suffix == ".rtf":
        raw = path.read_bytes()[: limit * 4]
        text = raw.decode("utf-8", errors="replace")
        text = re.sub(r"\\'[0-9a-fA-F]{2}", " ", text)
        text = re.sub(r"\\[a-zA-Z]+-?\d*\s?", " ", text)
        text = re.sub(r"[{}]", "", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text[:limit], len(text) > limit

    selectors: tuple[str, ...]
    if suffix == ".docx":
        selectors = ("word/document.xml",)
    elif suffix == ".xlsx":
        selectors = ("xl/sharedStrings.xml", "xl/worksheets/")
    elif suffix == ".pptx":
        selectors = ("ppt/slides/",)
    elif suffix in {".odt", ".ods", ".odp"}:
        selectors = ("content.xml",)
    else:
        return "", False

    chunks: list[str] = []
    total = 0
    truncated = False
    byte_budget = max(limit * 8, 256 * 1024)
    with zipfile.ZipFile(path) as archive:
        selected = [
            info
            for info in archive.infolist()
            if any(
                info.filename == prefix or info.filename.startswith(prefix)
                for prefix in selectors
            )
        ]
        selected = sorted(selected, key=lambda info: info.filename)[:64]
        for info in selected:
            remaining = limit - total
            if remaining <= 0 or byte_budget <= 0:
                truncated = True
                break
            if info.file_size > max(limit * 8, 1024 * 1024):
                truncated = True
                continue
            read_limit = min(byte_budget, max(remaining * 4, 4096))
            with archive.open(info) as handle:
                raw = handle.read(read_limit + 1)
            if len(raw) > read_limit:
                raw = raw[:read_limit]
                truncated = True
            byte_budget -= len(raw)
            text = _xml_text(raw, max_chars=remaining)
            if text:
                chunks.append(text)
                total += len(text) + 2
            truncated = truncated or info.file_size > len(raw)
    result = "\n\n".join(chunks).strip()
    return result[:limit], truncated or len(result) > limit

def _artifact_payload(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if os.path.commonpath([str(WORKSPACE), str(resolved)]) != str(WORKSPACE):
        raise HTTPException(status_code=403, detail="Artifact path is outside the workspace")
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail="Artifact file not found")
    stat = resolved.stat()
    relative = str(resolved.relative_to(WORKSPACE))
    media_type, _ = mimetypes.guess_type(resolved.name)
    return {
        "name": resolved.name,
        "path": "/app/workspace/" + relative,
        "relative": relative,
        "size": stat.st_size,
        "modified": stat.st_mtime,
        "mime": media_type or "application/octet-stream",
        "preview_kind": _preview_kind(resolved),
    }

def _workspace_file_snapshot(limit: int = ARTIFACT_SCAN_LIMIT) -> dict[str, tuple[int, int]]:
    """Return a bounded fingerprint map for regular files under the workspace."""
    result: dict[str, tuple[int, int]] = {}
    seen = 0
    if not WORKSPACE.exists():
        return result
    for root, dirs, files in os.walk(WORKSPACE, topdown=True, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not (Path(root) / d).is_symlink())
        for name in sorted(files):
            path = Path(root) / name
            try:
                if path.is_symlink():
                    continue
                relative = str(path.relative_to(WORKSPACE))
                if relative in ARTIFACT_IGNORE_RELATIVE:
                    continue
                stat = path.stat()
            except (OSError, ValueError):
                continue
            result[relative] = (int(stat.st_size), int(stat.st_mtime_ns))
            seen += 1
            if seen >= limit:
                return result
    return result

def _new_artifacts(previous: dict[str, tuple[int, int]], current: dict[str, tuple[int, int]], *, limit: int = ARTIFACT_MAX_PER_TURN) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for relative in sorted(set(current) - set(previous)):
        try:
            items.append(_artifact_payload(WORKSPACE / relative))
        except HTTPException:
            continue
        if len(items) >= limit:
            break
    return items

def _workspace_listing(relative: str = "") -> dict[str, Any]:
    base = _safe_workspace_path(relative or ".")
    if not base.exists():
        raise HTTPException(status_code=404, detail="Workspace path not found")
    if not base.is_dir():
        raise HTTPException(status_code=400, detail="Workspace path is not a directory")
    items: list[dict[str, Any]] = []
    try:
        children = sorted(base.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Unable to list workspace: {exc}") from exc
    for child in children[:WORKSPACE_LIST_LIMIT]:
        try:
            resolved = child.resolve()
            if os.path.commonpath([str(WORKSPACE), str(resolved)]) != str(WORKSPACE):
                continue
            stat = child.stat()
        except (OSError, ValueError):
            continue
        rel = str(child.relative_to(WORKSPACE))
        items.append({
            "name": child.name,
            "path": "/app/workspace/" + rel,
            "relative": rel,
            "type": "directory" if child.is_dir() else "file",
            "size": None if child.is_dir() else stat.st_size,
            "modified": stat.st_mtime,
            "media": child.suffix.lower() in ALLOWED_MEDIA_EXT,
            "text": child.suffix.lower() in ALLOWED_TEXT_EXT,
        })
    rel_base = "" if base == WORKSPACE else str(base.relative_to(WORKSPACE))
    parent_path = Path(rel_base).parent if rel_base else None
    parent = None if parent_path is None else ("" if parent_path == Path(".") else str(parent_path))
    return {
        "path": rel_base,
        "parent": parent,
        "items": items,
        "truncated": len(children) > WORKSPACE_LIST_LIMIT,
        "limit": WORKSPACE_LIST_LIMIT,
    }

def _build_user_content(text: str, attachments: list[str]) -> str:
    text = str(text or "").strip()
    blocks = [text] if text else []
    for raw in attachments[:8]:
        path = _safe_workspace_path(raw)
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        app_path = "/app/workspace/" + str(path.relative_to(WORKSPACE))
        if ext in ALLOWED_TEXT_EXT and path.stat().st_size <= 128 * 1024:
            try:
                body = path.read_text(encoding="utf-8", errors="replace")[:65536]
                blocks.append(f"Attached text file `{path.name}` ({app_path}):\n\n```text\n{body}\n```")
                continue
            except OSError:
                pass
        blocks.append(f"Attached file: {app_path}")
    return "\n\n".join(blocks).strip()
