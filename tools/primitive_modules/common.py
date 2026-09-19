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

def _load_json(data: str = "", path: str = "") -> Any:
    raw=_source_text(data,path)
    return json.loads(raw)

def _get_path(obj: Any, path: str) -> Any:
    if path in {"", ".", "$"}: return obj
    current=obj
    for part in path.strip("$.").split("."):
        if not part: continue
        if isinstance(current,list): current=current[int(part)]
        elif isinstance(current,dict): current=current[part]
        else: raise KeyError(part)
    return current
