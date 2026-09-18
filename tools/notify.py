# ==========================================
# FILE: tools/notify.py
# ==========================================
"""Desktop notification integration."""
from __future__ import annotations

import os
import subprocess
from typing import Any


def _display_number(value: Any, suffix: str = "") -> str | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return f"{number:.1f}{suffix}"


def format_monitor_notification(event_type: str, details: dict[str, Any] | None = None) -> str:
    """Render structured monitor details as a compact desktop-notification body."""
    details = details if isinstance(details, dict) else {}

    if event_type == "host_high_temperature":
        lines = []
        for entry in details.get("sensors", [])[:8]:
            if not isinstance(entry, dict):
                continue
            temperature = _display_number(entry.get("current"), " °C")
            if temperature is None:
                continue
            sensor = str(entry.get("sensor") or "Temperature sensor").strip()
            label = str(entry.get("label") or "").strip()
            name = f"{sensor} ({label})" if label and label.casefold() != sensor.casefold() else sensor
            lines.append(f"{name}: {temperature}")
        return "\n".join(lines) or "A temperature sensor crossed the configured threshold."

    if event_type == "host_high_memory":
        percent = _display_number(details.get("used_percent"), "%")
        return f"Memory usage: {percent}" if percent else "Memory usage crossed the configured threshold."

    if event_type == "host_high_disk":
        percent = _display_number(details.get("used_percent"), "%")
        return f"Disk usage: {percent}" if percent else "Disk usage crossed the configured threshold."

    if event_type == "ollama_unavailable":
        snapshot = details.get("snapshot")
        error = snapshot.get("error") if isinstance(snapshot, dict) else None
        return f"Ollama health check failed: {str(error)[:500]}" if error else "Ollama health check failed."

    scalars = [
        f"{str(key).replace('_', ' ').title()}: {value}"
        for key, value in details.items()
        if isinstance(value, (str, int, float, bool))
    ]
    return "\n".join(scalars[:8]) or "A monitor threshold was crossed."


def notify_desktop(title: str, message: str = "") -> str:
    """Send a native desktop notification through the host user's D-Bus session."""
    if not str(title).strip():
        return "Error: Missing required 'title' parameter."
    uid = os.getuid()
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{uid}")
    env = os.environ.copy()
    env.setdefault("XDG_RUNTIME_DIR", runtime_dir)
    env.setdefault("DBUS_SESSION_BUS_ADDRESS", f"unix:path={runtime_dir}/bus")
    env.setdefault("DISPLAY", ":0")
    env.setdefault("WAYLAND_DISPLAY", "wayland-1")
    try:
        proc = subprocess.run(
            ["notify-send", "--app-name=LocalAgent", str(title), str(message)],
            env=env,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode == 0:
            return f"Notification sent: {title}"
        return f"Notification failed (exit {proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
    except Exception as exc:
        return f"Notification error: {exc}"
