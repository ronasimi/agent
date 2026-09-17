# ==========================================
# FILE: tools/notify.py
# ==========================================
"""Desktop notification integration."""
from __future__ import annotations

import os
import subprocess


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
