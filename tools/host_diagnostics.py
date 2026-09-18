"""Structured host-monitoring primitives for small local models."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


def _run(argv: list[str], timeout: float = 8) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
        return proc.returncode, proc.stdout, proc.stderr
    except Exception as exc:
        return 1, "", str(exc)


def process_snapshot(limit: int = 10, sort_by: str = "cpu", include_command: bool = False) -> str:
    """Return top host-visible processes by CPU, memory, or I/O using the shared PID namespace."""
    if psutil is None:
        return "Error: psutil is unavailable."
    limit = max(1, min(int(limit), 30))
    sort_by = str(sort_by or "cpu").lower()
    if sort_by not in {"cpu", "memory", "io"}:
        return "Error: sort_by must be one of cpu, memory, io."
    processes = []
    for proc in psutil.process_iter():
        try:
            proc.cpu_percent(None)
            processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    # A short measurement interval avoids the all-zero first-sample behavior of
    # psutil without making host monitoring noticeably slow.
    time.sleep(0.12)
    rows: list[dict[str, Any]] = []
    for proc in processes:
        try:
            info = proc.as_dict(attrs=["pid", "name", "username", "status", "create_time", "memory_info", "cmdline"])
            mem = info.get("memory_info")
            io = proc.io_counters() if sort_by == "io" else None
            row: dict[str, Any] = {
                "pid": int(info.get("pid") or 0),
                "name": str(info.get("name") or ""),
                "username": str(info.get("username") or ""),
                "state": str(info.get("status") or ""),
                "cpu_percent": round(float(proc.cpu_percent(None) or 0.0), 1),
                "rss_mb": round((getattr(mem, "rss", 0) or 0) / 1024**2, 1),
                "age_seconds": max(0, int(time.time() - float(info.get("create_time") or time.time()))),
            }
            if io is not None:
                row["read_mb"] = round((getattr(io, "read_bytes", 0) or 0) / 1024**2, 1)
                row["write_mb"] = round((getattr(io, "write_bytes", 0) or 0) / 1024**2, 1)
            if include_command:
                command = " ".join(info.get("cmdline") or [])[:600]
                command = re.sub(r"(?i)(password|passwd|token|secret|api[_-]?key)=\S+", r"\1=[REDACTED]", command)
                row["command"] = command
            rows.append(row)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    key = {
        "cpu": lambda r: float(r.get("cpu_percent", 0)),
        "memory": lambda r: float(r.get("rss_mb", 0)),
        "io": lambda r: float(r.get("read_mb", 0)) + float(r.get("write_mb", 0)),
    }[sort_by]
    rows.sort(key=key, reverse=True)
    return json.dumps({"sort_by": sort_by, "processes": rows[:limit]}, ensure_ascii=False, indent=2)


def _parse_pressure_line(line: str) -> dict[str, Any]:
    parts = line.split()
    result: dict[str, Any] = {"scope": parts[0]} if parts else {}
    for token in parts[1:]:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        try:
            result[key] = float(value) if key.startswith("avg") else int(value)
        except ValueError:
            result[key] = value
    return result


def pressure_snapshot() -> str:
    """Return Linux PSI CPU, memory, and I/O pressure from the host when available."""
    roots = [Path("/host/proc/pressure"), Path("/proc/pressure")]
    root = next((candidate for candidate in roots if candidate.is_dir()), None)
    if root is None:
        return "Error: Linux pressure stall information is unavailable."
    result: dict[str, Any] = {"source": str(root)}
    for resource in ("cpu", "memory", "io"):
        path = root / resource
        try:
            result[resource] = [_parse_pressure_line(line) for line in path.read_text(errors="ignore").splitlines() if line.strip()]
        except Exception as exc:
            result[resource] = {"error": str(exc)}
    return json.dumps(result, ensure_ascii=False, indent=2)


def filesystem_snapshot(limit: int = 20) -> str:
    """Return bounded host filesystem capacity, inode usage, and read-only state."""
    limit = max(1, min(int(limit), 50))
    mounts_path = Path("/host/proc/mounts") if Path("/host/proc/mounts").is_file() else Path("/proc/mounts")
    pseudo = {"proc", "sysfs", "devtmpfs", "devpts", "tmpfs", "cgroup", "cgroup2", "overlay", "squashfs", "tracefs", "debugfs", "securityfs", "pstore", "mqueue", "hugetlbfs", "fusectl"}
    rows = []
    seen = set()
    try:
        lines = mounts_path.read_text(errors="ignore").splitlines()
    except Exception as exc:
        return f"Error: could not read mount table: {exc}"
    for line in lines:
        fields = line.split()
        if len(fields) < 4:
            continue
        device, mountpoint, fstype, options = fields[:4]
        mountpoint = mountpoint.replace("\\040", " ")
        if fstype in pseudo or mountpoint in seen:
            continue
        seen.add(mountpoint)
        probe_path = Path("/host") if mountpoint == "/" and Path("/host").exists() else Path("/host") / mountpoint.lstrip("/")
        if not probe_path.exists():
            continue
        try:
            stat = os.statvfs(probe_path)
            total = stat.f_frsize * stat.f_blocks
            free = stat.f_frsize * stat.f_bavail
            inode_total = stat.f_files
            inode_free = stat.f_favail
            rows.append({
                "device": device,
                "mountpoint": mountpoint,
                "fstype": fstype,
                "readonly": "ro" in options.split(","),
                "total_gb": round(total / 1024**3, 2),
                "free_gb": round(free / 1024**3, 2),
                "used_percent": round((1 - free / total) * 100, 1) if total else 0.0,
                "inode_used_percent": round((1 - inode_free / inode_total) * 100, 1) if inode_total else None,
            })
        except OSError:
            continue
    rows.sort(key=lambda item: float(item.get("used_percent") or 0), reverse=True)
    return json.dumps({"filesystems": rows[:limit]}, ensure_ascii=False, indent=2)


def service_health(service: str = "", lines: int = 40) -> str:
    """Inspect systemd service health with live-manager access when possible and journal/static fallbacks otherwise."""
    service = str(service or "").strip()
    lines = max(5, min(int(lines), 120))
    result: dict[str, Any] = {"service": service or None, "live_manager_access": False}

    if shutil.which("systemctl"):
        argv = ["systemctl", "--no-pager", "--plain"]
        if service:
            argv += ["show", service, "--property=LoadState,ActiveState,SubState,UnitFileState,NRestarts,ExecMainStatus,StateChangeTimestamp"]
        else:
            argv += ["--failed", "--no-legend"]
        code, stdout, stderr = _run(argv, timeout=6)
        if code == 0:
            result["live_manager_access"] = True
            if service:
                state = {}
                for row in stdout.splitlines():
                    if "=" in row:
                        key, value = row.split("=", 1)
                        state[key] = value
                result["state"] = state
            else:
                result["failed_units"] = stdout.strip().splitlines()[:50]
        else:
            result["live_manager_error"] = (stderr or stdout).strip()[:600]

    if service and shutil.which("systemctl"):
        code, stdout, stderr = _run(["systemctl", "--root=/host", "is-enabled", service], timeout=5)
        if stdout.strip():
            result["static_unit_state"] = stdout.strip().splitlines()[0][:80]

    if shutil.which("journalctl") and Path("/host_log/journal").exists():
        argv = ["journalctl", "-D", "/host_log/journal", "--no-pager", "-n", str(lines), "-p", "warning"]
        if service:
            argv.extend(["-u", service])
        code, stdout, stderr = _run(argv, timeout=10)
        if stdout.strip():
            result["recent_warning_or_higher"] = stdout.strip().splitlines()[-lines:]
        elif code != 0:
            result["journal_error"] = stderr.strip()[:600]
    return json.dumps(result, ensure_ascii=False, indent=2)
