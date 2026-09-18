"""Media-aware tool results that can be fed back into Ollama vision messages."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .netutil import validate_public_url

MEDIA_RESULT_MARKER = "__agent_media_result__"
WORKSPACE_ROOT = Path("/app/workspace")
SUPPORTED_LOCAL_MEDIA = {".png", ".jpg", ".jpeg", ".webp", ".pdf"}


def media_result(content: str, images: list[str] | tuple[str, ...]) -> dict[str, Any]:
    """Return text plus media references for the frontend to attach on the next model step."""
    refs = [str(item).strip() for item in images if str(item).strip()]
    return {
        MEDIA_RESULT_MARKER: True,
        "content": str(content or ""),
        "images": refs,
    }


def unpack_media_result(result: Any) -> tuple[str, list[str]]:
    """Split a tool return value into normal text and optional media references."""
    if isinstance(result, dict) and result.get(MEDIA_RESULT_MARKER) is True:
        content = str(result.get("content", ""))
        raw_images = result.get("images") or []
        if isinstance(raw_images, (str, os.PathLike)):
            raw_images = [raw_images]
        images = [str(item).strip() for item in raw_images if str(item).strip()]
        return content, images
    return str(result), []


def _resolve_local_media(path: str) -> Path:
    value = str(path or "").strip()
    if not value:
        raise ValueError("A media path is required.")

    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = WORKSPACE_ROOT / value.lstrip("/")
    candidate = candidate.resolve()
    root = WORKSPACE_ROOT.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("Local media must be inside /app/workspace.") from exc

    if candidate.suffix.lower() not in SUPPORTED_LOCAL_MEDIA:
        supported = ", ".join(sorted(SUPPORTED_LOCAL_MEDIA))
        raise ValueError(f"Unsupported media type. Supported local extensions: {supported}.")
    if not candidate.is_file():
        raise FileNotFoundError(f"Media file does not exist: {candidate}")
    return candidate


def attach_media(path: str, context: str = "") -> dict[str, Any] | str:
    """Attach a workspace image/PDF or public image URL for direct visual analysis by the main model."""
    value = str(path or "").strip()
    if not value:
        return "Error: a media path or URL is required."

    try:
        if value.startswith(("http://", "https://")):
            reference = validate_public_url(value)
        else:
            reference = str(_resolve_local_media(value))
    except Exception as exc:
        return f"Error: {exc}"

    note = str(context or "").strip()
    if note:
        content = f"Media queued for visual analysis: {reference}\nContext: {note}"
    else:
        content = f"Media queued for visual analysis: {reference}"
    return media_result(content, [reference])
