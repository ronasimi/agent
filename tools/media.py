"""Media-aware tool results that can be fed back into Ollama vision messages."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .netutil import validate_public_url

MEDIA_RESULT_MARKER = "__agent_media_result__"
PROFILE_MEDIA_REFERENCE = "profile-image://current"
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


def resolve_profile_media(reference: str) -> Path | None:
    """Resolve only the canonical durable profile image.

    This deliberately does not make ``/app/memory`` a generally readable media
    root. The stable logical reference and the exact configured profile-image
    path are the only accepted non-workspace inputs.
    """
    value = str(reference or "").strip()
    try:
        from .user_profile import PROFILE_IMAGE_PATH

        profile_path = PROFILE_IMAGE_PATH.resolve()
        candidate = Path(value).resolve() if value and value != PROFILE_MEDIA_REFERENCE else profile_path
        if value != PROFILE_MEDIA_REFERENCE and candidate != profile_path:
            return None
        if not profile_path.is_file():
            raise FileNotFoundError("No profile image is configured.")
        if profile_path.suffix.lower() not in SUPPORTED_LOCAL_MEDIA:
            raise ValueError("The configured profile image has an unsupported media type.")
        return profile_path
    except (OSError, ValueError):
        if value == PROFILE_MEDIA_REFERENCE:
            raise
        return None


def _resolve_local_media(path: str) -> Path:
    value = str(path or "").strip()
    if not value:
        raise ValueError("A media path is required.")

    profile_path = resolve_profile_media(value)
    if profile_path is not None:
        candidate = profile_path
    else:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = WORKSPACE_ROOT / value.lstrip("/")
        candidate = candidate.resolve()
        root = WORKSPACE_ROOT.resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                "Local media must be inside /app/workspace or be the current profile image."
            ) from exc

    if candidate.suffix.lower() not in SUPPORTED_LOCAL_MEDIA:
        supported = ", ".join(sorted(SUPPORTED_LOCAL_MEDIA))
        raise ValueError(f"Unsupported media type. Supported local extensions: {supported}.")
    if not candidate.is_file():
        raise FileNotFoundError(f"Media file does not exist: {candidate}")
    return candidate


def attach_media(path: str, context: str = "") -> dict[str, Any] | str:
    """Attach workspace media, the current profile image, or a public image URL for visual analysis."""
    value = str(path or "").strip()
    if not value:
        return "Error: a media path or URL is required."

    try:
        if value.startswith(("http://", "https://")):
            reference = validate_public_url(value)
        else:
            resolved = _resolve_local_media(value)
            reference = PROFILE_MEDIA_REFERENCE if resolve_profile_media(value) is not None else str(resolved)
    except Exception as exc:
        return f"Error: {exc}"

    note = str(context or "").strip()
    if note:
        content = f"Media queued for visual analysis: {reference}\nContext: {note}"
    else:
        content = f"Media queued for visual analysis: {reference}"
    return media_result(content, [reference])
