from __future__ import annotations

import os

WORKSPACE_DIR = os.path.realpath("/app/workspace")
os.makedirs(WORKSPACE_DIR, exist_ok=True)


def _get_safe_path(filename: str) -> str:
    """Resolve a path and ensure symlinks cannot escape the workspace."""
    safe_path = os.path.realpath(os.path.join(WORKSPACE_DIR, str(filename).lstrip("/")))
    if os.path.commonpath([WORKSPACE_DIR, safe_path]) != WORKSPACE_DIR:
        raise ValueError(f"Path traversal outside workspace blocked: {filename}")
    return safe_path


def read_file(filename: str = "") -> str:
    """Read a UTF-8 text file inside the workspace."""
    if not str(filename).strip():
        return "Error: Missing required 'filename' parameter."
    try:
        path = _get_safe_path(filename)
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            data = handle.read(50000)
        return data + ("\n[truncated]" if len(data) >= 50000 else "")
    except Exception as exc:
        return f"Error reading file '{filename}': {exc}"


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
        return f"Error writing file '{filename}': {exc}"
