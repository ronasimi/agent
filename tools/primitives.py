"""Small deterministic primitives for common environment questions.

These tools intentionally avoid shell execution and do not expose arbitrary
process environment variables or other secret-bearing runtime state.
"""
from __future__ import annotations

import json
import os
import platform
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .config import load_config


def _read_text(path: Path, limit: int = 256) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore").strip()[:limit]
    except OSError:
        return ""


def _configured_timezone() -> str:
    try:
        configured = str(load_config().get("agent", {}).get("timezone") or "").strip()
    except Exception:
        configured = ""
    return configured or str(os.environ.get("TZ") or "").strip() or "UTC"


def _host_timezone_name() -> str:
    """Best-effort host timezone name without executing external commands."""
    # Debian-style hosts may provide an explicit name.
    for path in (Path("/host/etc/timezone"), Path("/etc/timezone")):
        value = _read_text(path, 128)
        if value and "/" in value:
            return value

    # Arch and many other distributions use an /etc/localtime symlink.
    for path in (Path("/host/etc/localtime"), Path("/etc/localtime")):
        try:
            resolved = path.resolve(strict=True)
        except OSError:
            continue
        marker = "/usr/share/zoneinfo/"
        text = str(resolved)
        if marker in text:
            return text.split(marker, 1)[1]

    return str(os.environ.get("TZ") or "").strip() or _configured_timezone()


def _zone(name: str):
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return datetime.now().astimezone().tzinfo or timezone.utc


def clock_payload(timezone_name: str = "") -> dict[str, Any]:
    """Return one internally consistent UTC/local timestamp payload.

    ``timezone_name`` may override the configured local zone with an explicit
    IANA timezone. Invalid values fall back safely to the configured/host zone.
    """
    now_utc = datetime.now(timezone.utc)
    configured_tz = _configured_timezone()
    host_tz = _host_timezone_name()
    requested_tz = str(timezone_name or "").strip()
    target_tz = requested_tz or configured_tz or host_tz or "UTC"
    try:
        ZoneInfo(target_tz)
    except (ZoneInfoNotFoundError, ValueError):
        target_tz = configured_tz or host_tz or "UTC"
    local = now_utc.astimezone(_zone(target_tz))
    system_local = now_utc.astimezone()
    return {
        "utc": now_utc.isoformat(timespec="seconds"),
        "local": local.isoformat(timespec="seconds"),
        "date": local.date().isoformat(),
        "time": local.strftime("%H:%M:%S"),
        "day_of_week": local.strftime("%A"),
        "timezone": target_tz,
        "timezone_abbreviation": local.tzname() or "",
        "utc_offset": local.strftime("%z"),
        "unix_timestamp": int(now_utc.timestamp()),
        "host_timezone": host_tz,
        "system_local": system_local.isoformat(timespec="seconds"),
    }


def current_time(timezone_name: str = "") -> str:
    """Return the current clock/date for the configured zone or an explicit IANA timezone."""
    return json.dumps(clock_payload(timezone_name), ensure_ascii=False, indent=2)


def hostname() -> str:
    """Return the host hostname and the runtime/container hostname without DNS lookups."""
    runtime_hostname = socket.gethostname()
    host_hostname = _read_text(Path("/host/etc/hostname"), 253) or runtime_hostname
    return json.dumps(
        {
            "host_hostname": host_hostname,
            "runtime_hostname": runtime_hostname,
            "same_hostname": host_hostname == runtime_hostname,
        },
        ensure_ascii=False,
        indent=2,
    )


def environment_summary() -> str:
    """Return a small non-secret runtime summary: time, host identity, platform, Python, models, and workspace availability."""
    config = load_config()
    agent_cfg = config.get("agent", {}) if isinstance(config, dict) else {}
    host_name = _read_text(Path("/host/etc/hostname"), 253) or socket.gethostname()
    payload = {
        "clock": clock_payload(),
        "host_hostname": host_name,
        "runtime_hostname": socket.gethostname(),
        "platform": platform.platform(),
        "kernel": platform.release(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "pid": os.getpid(),
        "main_model": str(agent_cfg.get("model") or ""),
        "fast_model": str(agent_cfg.get("fast_model") or ""),
        "context_tokens": int((agent_cfg.get("context") or {}).get("num_ctx") or 0),
        "workspace_available": Path("/app/workspace").is_dir() or (Path.cwd() / "workspace").is_dir(),
        "note": "This summary intentionally excludes arbitrary environment variables, credentials, and secrets.",
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
