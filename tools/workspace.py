from __future__ import annotations

import mimetypes
import os

WORKSPACE_DIR = os.path.realpath("/app/workspace")
os.makedirs(WORKSPACE_DIR, exist_ok=True)

_BINARY_TEXT_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".ico", ".pdf",
    ".zip", ".gz", ".bz2", ".xz", ".7z", ".tar", ".mp3", ".wav", ".ogg",
    ".flac", ".mp4", ".webm", ".mov",
}

def _binary_text_reason(path: str) -> str:
    suffix = os.path.splitext(path)[1].lower()
    mime, _ = mimetypes.guess_type(path)
    if suffix in _BINARY_TEXT_EXTENSIONS or (mime and mime.split("/", 1)[0] in {"image", "audio", "video"}):
        return mime or suffix.lstrip(".") or "binary"
    with open(path, "rb") as handle:
        probe = handle.read(2048)
    if b"\x00" in probe:
        return mime or "binary data"
    if probe:
        controls = sum(1 for byte in probe if byte < 32 and byte not in {9, 10, 13})
        if controls / len(probe) > 0.08:
            return mime or "binary data"
    return ""


def _get_safe_path(filename: str) -> str:
    """Resolve relative or absolute workspace paths without duplicating /app/workspace."""
    raw = str(filename).strip()
    candidate = raw if os.path.isabs(raw) else os.path.join(WORKSPACE_DIR, raw)
    safe_path = os.path.realpath(candidate)
    if os.path.commonpath([WORKSPACE_DIR, safe_path]) != WORKSPACE_DIR:
        raise ValueError(f"Path traversal outside workspace blocked: {filename}")
    return safe_path


def read_file(filename: str = "") -> str:
    """Read a UTF-8 text file inside the workspace."""
    if not str(filename).strip():
        return "Error: Missing required 'filename' parameter."
    try:
        path = _get_safe_path(filename)
        binary_reason = _binary_text_reason(path)
        if binary_reason:
            return f"Error: read_file only supports text files; '{filename}' appears to be binary ({binary_reason}). Use image_info/attach_media for images, document_text for documents, or read_bytes for raw bytes."
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            data = handle.read(50000)
        return data + ("\n[truncated]" if len(data) >= 50000 else "")
    except Exception as exc:
        return f"Error: reading file '{filename}' failed: {exc}"


def write_file(filename: str = "", content: str = "") -> str:
    """Write UTF-8 text inside the workspace and create missing parent directories."""
    if not str(filename).strip():
        return "Error: Missing required 'filename' parameter."
    try:
        path = _get_safe_path(filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(str(content))
        return f"Successfully wrote {len(str(content))} characters to {filename}"
    except Exception as exc:
        return f"Error: writing file '{filename}' failed: {exc}"
