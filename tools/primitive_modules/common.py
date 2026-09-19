"""Shared helpers for primitive modules."""
from __future__ import annotations

import ast
import csv
import hashlib
import ipaddress
import json
import math
import mimetypes
import os
import platform
import re
import shutil
import socket
import ssl
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import psutil
import requests

from ..workspace import _get_safe_path

MAX_TEXT = 100_000

_BINARY_TEXT_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".ico",
    ".pdf", ".zip", ".gz", ".bz2", ".xz", ".7z", ".tar",
    ".mp3", ".wav", ".ogg", ".flac", ".mp4", ".webm", ".mov",
}

def _binary_text_reason(path: Path) -> str:
    """Return a MIME/reason string when a path should not be decoded as text."""
    suffix = path.suffix.lower()
    mime, _ = mimetypes.guess_type(str(path))
    if suffix in _BINARY_TEXT_EXTENSIONS or (mime and mime.split("/", 1)[0] in {"image", "audio", "video"}):
        return mime or suffix.lstrip(".") or "binary"
    try:
        with path.open("rb") as handle:
            probe = handle.read(2048)
    except OSError:
        return ""
    if b"\x00" in probe:
        return mime or "binary data"
    if probe:
        controls = sum(1 for byte in probe if byte < 32 and byte not in {9, 10, 13})
        if controls / len(probe) > 0.08:
            return mime or "binary data"
    return ""

def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)

def _bounded_int(value: int, low: int, high: int) -> int:
    return max(low, min(int(value), high))

def _safe_workspace(path: str) -> Path:
    return Path(_get_safe_path(path))

def _source_text(text: str = "", path: str = "", limit: int = MAX_TEXT) -> str:
    if path:
        p=_safe_workspace(path)
        return p.read_text(encoding="utf-8",errors="replace")[:limit]
    return str(text)[:limit]

def _load_json(data: Any = "", path: str = "") -> Any:
    if path:
        return json.loads(_source_text("", path))
    if isinstance(data, (dict, list, int, float, bool)) or data is None:
        return data
    return json.loads(str(data))

def _get_path(obj: Any, path: str) -> Any:
    if path in {"", ".", "$"}: return obj
    current=obj
    for part in path.strip("$.").split("."):
        if not part: continue
        if isinstance(current,list): current=current[int(part)]
        elif isinstance(current,dict): current=current[part]
        else: raise KeyError(part)
    return current
