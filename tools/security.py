"""Small deterministic safety helpers for model-accessible local tools."""
from __future__ import annotations

import fnmatch
import json
import os
import re
from pathlib import PurePath
from typing import Any

_SENSITIVE_GLOBS = (
    ".env", ".env.*", "*.pem", "*.key", "id_rsa*", "id_ed25519*",
    "credentials*", "credential*", "secrets*", "secret*", "*.p12", "*.pfx",
)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?im)^(\s*(?:api[_-]?key|access[_-]?token|refresh[_-]?token|token|password|passwd|secret|client[_-]?secret|private[_-]?key)\s*[:=]\s*)([^\s#][^\r\n]*)$"
)
_PEM_RE = re.compile(r"-----BEGIN [^-]*(?:PRIVATE KEY|CERTIFICATE)-----.*?-----END [^-]*(?:PRIVATE KEY|CERTIFICATE)-----", re.I | re.S)


def is_sensitive_path(value: Any) -> bool:
    raw = str(value or "").strip().replace("\\", "/")
    if not raw:
        return False
    names = [part for part in PurePath(raw).parts if part not in {"/", ".", ".."}]
    for name in names[-3:]:
        lower = name.lower()
        if any(fnmatch.fnmatch(lower, pattern) for pattern in _SENSITIVE_GLOBS):
            return True
    return False


def arguments_reference_sensitive_path(arguments: Any) -> bool:
    if isinstance(arguments, dict):
        return any(arguments_reference_sensitive_path(value) for value in arguments.values())
    if isinstance(arguments, (list, tuple)):
        return any(arguments_reference_sensitive_path(value) for value in arguments)
    if not isinstance(arguments, str):
        return False
    text = arguments.strip()
    if is_sensitive_path(text):
        return True
    # Commands/specifications may contain a path among other tokens.
    return any(is_sensitive_path(token.strip("'\"`()[]{};,")) for token in re.split(r"\s+", text) if token)


def user_explicitly_requested_sensitive_access(user_text: str, arguments: Any) -> bool:
    """Require the user turn itself to name the sensitive target or category."""
    if not arguments_reference_sensitive_path(arguments):
        return False
    user = str(user_text or "").lower()
    if re.search(r"\b(?:\.env|credential|credentials|secret|secrets|private key|pem|id_rsa|id_ed25519)\b", user):
        return True
    # Exact basename mention also counts as explicit intent.
    values = []
    if isinstance(arguments, dict):
        values = [str(v) for v in arguments.values() if isinstance(v, str)]
    elif isinstance(arguments, str):
        values = [arguments]
    for value in values:
        base = os.path.basename(value.strip().strip("'\""))
        if base and is_sensitive_path(base) and base.lower() in user:
            return True
    return False


def redact_secrets(text: str) -> str:
    """Redact common credential forms before tool output enters model context/logs."""
    value = str(text or "")
    value = _PEM_RE.sub("[REDACTED PEM MATERIAL]", value)
    value = _SECRET_ASSIGNMENT_RE.sub(lambda m: m.group(1) + "[REDACTED]", value)
    # JSON credential fields.
    value = re.sub(
        r'(?i)("(?:api[_-]?key|access[_-]?token|refresh[_-]?token|password|secret|client[_-]?secret|private[_-]?key)"\s*:\s*)"[^"]*"',
        r'\1"[REDACTED]"', value,
    )
    return value
