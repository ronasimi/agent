"""Durable desktop reminders backed by systemd user timers."""
from __future__ import annotations

import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo


from .runtime import DB_PATH, DB_TIMEOUT, init_runtime_db
from .config import load_config

_config = load_config()

_REMINDER_CFG = _config.get("reminders", {})
TIMEZONE = _REMINDER_CFG.get("timezone") or _config.get("agent", {}).get("timezone") or "America/Toronto"
HOST_UID = os.getuid()
XDG_RUNTIME_DIR = os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{HOST_UID}")
TIMER_DIR = Path(_REMINDER_CFG.get("timer_dir") or (Path.home() / ".config/systemd/user")).resolve()
UNIT_PREFIX = str(_REMINDER_CFG.get("unit_prefix", "agent-reminder"))
DB_SYSTEMCTL_TIMEOUT = int(_REMINDER_CFG.get("systemctl_timeout", 10))

def _connect():
    init_runtime_db()
    import sqlite3
    conn = sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=15000")
    return conn


def _slug(value: str, max_len: int = 48) -> str:
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(value).strip().lower()).strip("-._")
    return (value[:max_len] or "reminder")


def _parse_when(when: str) -> datetime:
    raw = str(when).strip()
    if not raw:
        raise ValueError("'when' is required and must be an ISO-8601 datetime.")
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Invalid 'when'. Use ISO-8601, e.g. 2026-09-18T09:00:00-04:00.") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo(TIMEZONE))
    return parsed.astimezone(ZoneInfo(TIMEZONE))


def _calendar_expression(dt: datetime, repeat: str) -> str:
    repeat = repeat.lower()
    if repeat == "once":
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    if repeat == "daily":
        return dt.strftime("*-*-* %H:%M:%S")
    if repeat == "weekly":
        return dt.strftime("%A *-*-* %H:%M:%S")
    raise ValueError("'repeat' must be one of: once, daily, weekly.")


def _systemd_quote(value: str) -> str:
    escaped = str(value).replace("\\", "\\\\").replace("\"", "\\\"").replace("%", "%%")
    escaped = escaped.replace("\n", " ").replace("\r", " ")
    return f'"{escaped}"'


def _unit_files(unit_name: str, title: str, message: str, calendar: str, repeat: str) -> tuple[Path, Path]:
    TIMER_DIR.mkdir(parents=True, exist_ok=True)
    service_path = TIMER_DIR / f"{unit_name}.service"
    timer_path = TIMER_DIR / f"{unit_name}.timer"
    unit_title = str(title).replace("\n", " ").replace("\r", " ").replace("%", "%%")
    service = f"""[Unit]\nDescription=Local Agent reminder: {unit_title}\n\n[Service]\nType=oneshot\nEnvironment=XDG_RUNTIME_DIR={XDG_RUNTIME_DIR}\nEnvironment=DBUS_SESSION_BUS_ADDRESS=unix:path={XDG_RUNTIME_DIR}/bus\nExecStart=/usr/bin/notify-send --app-name=LocalAgent -- {_systemd_quote(title)} {_systemd_quote(message)}\n"""
    persistent = "true" if repeat == "once" else "false"
    timer = f"""[Unit]\nDescription=Local Agent reminder: {unit_title}\n\n[Timer]\nOnCalendar={calendar}\nPersistent={persistent}\nAccuracySec=1s\nUnit={unit_name}.service\n\n[Install]\nWantedBy=timers.target\n"""
    service_path.write_text(service, encoding="utf-8")
    timer_path.write_text(timer, encoding="utf-8")
    return service_path, timer_path


def _systemctl(*args: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.setdefault("XDG_RUNTIME_DIR", XDG_RUNTIME_DIR)
    env.setdefault("DBUS_SESSION_BUS_ADDRESS", f"unix:path={XDG_RUNTIME_DIR}/bus")
    return subprocess.run(
        ["systemctl", "--user", *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=DB_SYSTEMCTL_TIMEOUT,
    )


def schedule_reminder(
    title: str = "",
    message: str = "",
    when: str = "",
    repeat: str = "once",
    reminder_id: str = "",
    delay_seconds: int = 0,
) -> str:
    """Create and activate a durable desktop reminder using a systemd user timer."""
    if not str(title).strip():
        return "Error: Missing required 'title' parameter."
    try:
        if int(delay_seconds) > 0:
            from datetime import timedelta
            dt = datetime.now(timezone.utc) + timedelta(seconds=int(delay_seconds))
            dt = dt.astimezone(ZoneInfo(TIMEZONE))
        else:
            dt = _parse_when(when)
        calendar = _calendar_expression(dt, repeat)
    except ValueError as exc:
        return f"Error: {exc}"

    repeat = repeat.lower()
    if dt.timestamp() <= datetime.now(timezone.utc).timestamp() and repeat == "once":
        return "Error: Reminder time must be in the future."

    reminder_id = _slug(reminder_id or f"{title}-{dt.strftime('%Y%m%d-%H%M%S')}")
    unit_name = f"{UNIT_PREFIX}-{reminder_id}"
    try:
        service_path, timer_path = _unit_files(unit_name, title, message, calendar, repeat)
        reload_result = _systemctl("daemon-reload")
        if reload_result.returncode != 0:
            raise RuntimeError(reload_result.stderr.strip() or reload_result.stdout.strip() or "systemctl --user daemon-reload failed")
        enable_result = _systemctl("enable", "--now", f"{unit_name}.timer")
        if enable_result.returncode != 0:
            raise RuntimeError(enable_result.stderr.strip() or enable_result.stdout.strip() or "systemd timer activation failed")

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with _connect() as conn:
            conn.execute(
                """
                INSERT INTO reminders(id, title, message, when_iso, repeat_mode, unit_name, status, created_at, updated_at, last_error)
                VALUES (?, ?, ?, ?, ?, ?, 'scheduled', ?, ?, NULL)
                ON CONFLICT(id) DO UPDATE SET
                    title=excluded.title, message=excluded.message, when_iso=excluded.when_iso,
                    repeat_mode=excluded.repeat_mode, unit_name=excluded.unit_name, status='scheduled',
                    updated_at=excluded.updated_at, last_error=NULL
                """,
                (reminder_id, title, message, dt.isoformat(), repeat, unit_name, now, now),
            )
        return f"Reminder scheduled: {reminder_id} ({calendar}); unit={unit_name}"
    except Exception as exc:
        try:
            with _connect() as conn:
                conn.execute(
                    "INSERT INTO reminders(id, title, message, when_iso, repeat_mode, unit_name, status, created_at, updated_at, last_error) VALUES (?, ?, ?, ?, ?, ?, 'error', ?, ?, ?)",
                    (reminder_id, title, message, dt.isoformat(), repeat, unit_name, datetime.now(timezone.utc).isoformat(timespec="seconds"), datetime.now(timezone.utc).isoformat(timespec="seconds"), str(exc)),
                )
        except Exception:
            pass
        return f"Error: scheduling reminder failed: {exc}"


def cancel_reminder(reminder_id: str = "") -> str:
    """Cancel and remove a previously scheduled reminder."""
    if not str(reminder_id).strip():
        return "Error: Missing required 'reminder_id' parameter."
    reminder_id = _slug(reminder_id)
    with _connect() as conn:
        row = conn.execute("SELECT unit_name FROM reminders WHERE id = ?", (reminder_id,)).fetchone()
    if not row:
        return f"Error: reminder '{reminder_id}' not found."
    unit_name = row[0]
    errors = []
    for args in (("disable", "--now", f"{unit_name}.timer"), ("daemon-reload",)):
        try:
            result = _systemctl(*args)
            if result.returncode != 0:
                errors.append(result.stderr.strip() or result.stdout.strip() or str(args))
        except Exception as exc:
            errors.append(str(exc))
    for suffix in (".timer", ".service"):
        try:
            (TIMER_DIR / f"{unit_name}{suffix}").unlink(missing_ok=True)
        except Exception as exc:
            errors.append(str(exc))
    with _connect() as conn:
        conn.execute("UPDATE reminders SET status='cancelled', updated_at=CURRENT_TIMESTAMP, last_error=? WHERE id=?", ("; ".join(errors) if errors else None, reminder_id))
    return f"Reminder '{reminder_id}' cancelled." + (f" Warnings: {'; '.join(errors)}" if errors else "")


def list_reminders(status: str = "") -> str:
    """List durable reminders and their systemd unit names."""
    with _connect() as conn:
        if status:
            rows = conn.execute(
                "SELECT id, title, message, when_iso, repeat_mode, unit_name, status, last_error FROM reminders WHERE status=? ORDER BY when_iso",
                (status,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, title, message, when_iso, repeat_mode, unit_name, status, last_error FROM reminders WHERE status IN ('scheduled','error') ORDER BY when_iso",
            ).fetchall()
    items = [dict(row) for row in rows]
    return __import__("json").dumps(items, ensure_ascii=False, indent=2) if items else "No reminders found."
