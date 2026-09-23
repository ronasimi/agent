"""External initial-tape sources for durable deterministic computation.

Machine execution stays independent of filesystems. This module is the small
adapter that lets a caller seed a machine from a workspace text file or JSON
sparse-tape map without placing the whole input in the LLM context. External
sources are hash-pinned when the job is created so a later file edit cannot
silently change a queued computation's semantics.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .machine import MachineProgramError

WORKSPACE_ROOT = Path(os.environ.get("AGENT_WORKSPACE", "/app/workspace")).resolve()
VALID_INPUT_FILE_FORMATS = {"auto", "text", "tape_json"}


def _safe_workspace_path(value: str) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise MachineProgramError("input_file must be a non-empty workspace path")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = WORKSPACE_ROOT / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(WORKSPACE_ROOT)
    except ValueError as exc:
        raise MachineProgramError(f"input_file escapes the workspace: {value}") from exc
    if not resolved.is_file():
        raise MachineProgramError(f"input_file does not exist or is not a file: {value}")
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_input_file(input_file: str, input_file_format: str = "auto") -> dict[str, str]:
    """Validate/hash one workspace input and return a durable descriptor."""
    path = _safe_workspace_path(input_file)
    fmt = str(input_file_format or "auto").strip().lower()
    if fmt not in VALID_INPUT_FILE_FORMATS:
        raise MachineProgramError("input_file_format must be auto, text, or tape_json")
    if fmt == "auto":
        fmt = "tape_json" if path.suffix.lower() == ".json" else "text"
    return {
        "path": str(input_file),
        "format": fmt,
        "sha256": _sha256_file(path),
    }


def load_input_file(descriptor: dict[str, Any]) -> tuple[str, dict[str, str]]:
    """Load a previously hash-pinned input descriptor.

    Returns ``(input_text, initial_tape)``. A changed source is rejected rather
    than accepted as a different computation after a worker restart.
    """
    if not isinstance(descriptor, dict):
        raise MachineProgramError("input_file descriptor must be an object")
    path = _safe_workspace_path(str(descriptor.get("path") or ""))
    expected = str(descriptor.get("sha256") or "").strip().lower()
    actual = _sha256_file(path)
    if expected and actual != expected:
        raise MachineProgramError(
            f"input_file changed after the computation was queued: expected {expected}, got {actual}"
        )
    fmt = str(descriptor.get("format") or "auto").strip().lower()
    if fmt == "auto":
        fmt = "tape_json" if path.suffix.lower() == ".json" else "text"
    if fmt == "text":
        return path.read_text(encoding="utf-8"), {}
    if fmt != "tape_json":
        raise MachineProgramError(f"unsupported input_file format {fmt!r}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MachineProgramError(f"invalid tape_json input_file: {exc}") from exc
    if not isinstance(payload, dict):
        raise MachineProgramError("tape_json input_file must contain an address-to-symbol object")
    return "", {str(key): value for key, value in payload.items()}
