from __future__ import annotations

import mimetypes
import os
import shutil
import tempfile

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
        parent = os.path.dirname(path)
        os.makedirs(parent, exist_ok=True)
        payload = str(content)
        # Same-directory temporary + fsync + atomic replace means a cancelled or
        # crashed write cannot leave a half-written workspace file behind.
        fd, temp_path = tempfile.mkstemp(prefix=".agent-write-", dir=parent, text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
            try:
                dir_fd = os.open(parent, os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except OSError:
                pass
        finally:
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
        return f"Successfully wrote {len(payload)} characters to {filename}"
    except Exception as exc:
        return f"Error: writing file '{filename}' failed: {exc}"


def remove_path(path: str = "", recursive: bool = False) -> str:
    """Remove one file or directory strictly inside the agent workspace.

    Directory removal is intentionally explicit: non-empty directories require
    ``recursive=true``.  The workspace root itself can never be removed.
    """
    if not str(path).strip():
        return "Error: Missing required 'path' parameter."
    try:
        safe_path = _get_safe_path(path)
        if safe_path == WORKSPACE_DIR:
            return "Error: refusing to remove the workspace root."
        if not os.path.lexists(safe_path):
            return f"Error: path '{path}' does not exist."
        if os.path.isdir(safe_path) and not os.path.islink(safe_path):
            if recursive:
                shutil.rmtree(safe_path)
            else:
                os.rmdir(safe_path)
        else:
            os.unlink(safe_path)
        return f"Successfully removed {path}"
    except Exception as exc:
        return f"Error: removing path '{path}' failed: {exc}"
