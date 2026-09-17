"""Deterministic host inspection tools."""
from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import time
from pathlib import Path

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

from .runtime import record_monitor_event, record_monitor_state, get_monitor_state, list_monitor_events
from .netutil import validate_public_url


def _run(argv: list[str], timeout: float = 5) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr
    except Exception as exc:
        return 1, "", str(exc)


def host_snapshot() -> str:
    """Return a bounded JSON snapshot of host CPU, RAM, disk, load, uptime, and optional GPU state."""
    result = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "kernel": platform.release(),
        "cpu_count": os.cpu_count() or 1,
        "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else [],
        "uptime_seconds": None,
        "memory": {},
        "disk": {},
        "gpu": {},
    }
    if psutil:
        result["uptime_seconds"] = max(0, time.time() - psutil.boot_time())
        vm = psutil.virtual_memory()
        result["memory"] = {
            "total_mb": round(vm.total / 1024**2),
            "available_mb": round(vm.available / 1024**2),
            "used_mb": round(vm.used / 1024**2),
            "used_percent": vm.percent,
        }
        disk = psutil.disk_usage("/host" if os.path.isdir("/host") else "/")
        result["disk"] = {
            "total_gb": round(disk.total / 1024**3, 2),
            "free_gb": round(disk.free / 1024**3, 2),
            "used_percent": disk.percent,
        }
        temps = {}
        try:
            for name, entries in psutil.sensors_temperatures().items():
                temps[name] = [{"label": e.label, "current": e.current} for e in entries[:8]]
        except Exception:
            pass
        if temps:
            result["temperatures"] = temps

    result["gpu"] = gpu_snapshot_dict()
    return json.dumps(result, ensure_ascii=False, indent=2)


def gpu_snapshot_dict() -> dict:
    """Return optional NVIDIA/AMD GPU telemetry without failing on unsupported hosts."""
    code, stdout, stderr = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        timeout=3,
    )
    if code == 0 and stdout.strip():
        gpus = []
        for line in stdout.strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 4:
                try:
                    gpus.append({
                        "vendor": "nvidia",
                        "name": parts[0],
                        "vram_total_mb": float(parts[1]),
                        "vram_used_mb": float(parts[2]),
                        "utilization_percent": float(parts[3]),
                    })
                except ValueError:
                    pass
        if gpus:
            return {"gpus": gpus}

    code, stdout, stderr = _run(["rocm-smi", "--showmeminfo", "vram", "--showuse", "--json"], timeout=5)
    if code == 0 and stdout.strip():
        try:
            return {"vendor": "amd", "raw": json.loads(stdout)}
        except json.JSONDecodeError:
            return {"vendor": "amd", "raw_text": stdout[:4000]}
    return {"available": False}


def ollama_runtime_snapshot() -> str:
    """Inspect models currently loaded by the configured Ollama server via /api/ps."""
    import requests
    url = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/") + "/api/ps"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return json.dumps(response.json(), ensure_ascii=False, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def network_snapshot() -> str:
    """Return bounded JSON describing interfaces, addresses, routes, DNS, and listening sockets."""
    snapshot: dict = {"interfaces": [], "routes": [], "dns": [], "listening": []}
    code, stdout, stderr = _run(["ip", "-j", "address"], timeout=5)
    if code == 0:
        try:
            snapshot["interfaces"] = json.loads(stdout)
        except json.JSONDecodeError:
            snapshot["interfaces_error"] = stdout[:2000]

    code, stdout, stderr = _run(["ip", "-j", "route"], timeout=5)
    if code == 0:
        try:
            snapshot["routes"] = json.loads(stdout)
        except json.JSONDecodeError:
            snapshot["routes_error"] = stdout[:2000]

    resolv = Path("/etc/resolv.conf")
    try:
        snapshot["dns"] = [
            line.split()[1]
            for line in resolv.read_text(errors="ignore").splitlines()
            if line.strip().startswith("nameserver ") and len(line.split()) > 1
        ]
    except Exception:
        pass

    code, stdout, stderr = _run(["ss", "-H", "-lntu"], timeout=5)
    if code == 0:
        snapshot["listening"] = stdout.strip().splitlines()[:200]

    return json.dumps(snapshot, ensure_ascii=False, indent=2)


def network_reachability(targets: list[str] | None = None) -> str:
    """Check DNS/HTTPS reachability for a small set of configured public endpoints."""
    from .netutil import fetch_bytes
    targets = targets or ["https://www.cloudflare.com/cdn-cgi/trace", "https://example.com/"]
    results = []
    for target in targets[:5]:
        item = {"target": str(target)}
        started = time.monotonic()
        try:
            safe_target = validate_public_url(str(target))
            final_url, response, _ = fetch_bytes(
                safe_target, timeout=5, max_bytes=64 * 1024, max_redirects=3, allow_private=False
            )
            item["final_url"] = final_url
            item["status"] = response.status_code
            item["ok"] = response.ok
        except Exception as exc:
            item["ok"] = False
            item["error"] = str(exc)
        finally:
            item["elapsed_ms"] = round((time.monotonic() - started) * 1000, 1)
        results.append(item)
    return json.dumps(results, ensure_ascii=False, indent=2)


def list_host_monitor_events(event_type: str = "", limit: int = 25) -> str:
    """Return recent durable host/network/Ollama monitoring events."""
    return json.dumps(list_monitor_events(event_type=event_type, limit=limit), ensure_ascii=False, indent=2)


def read_host_file(filepath: str = "/host/etc/resolv.conf") -> str:
    """Safely read a text file from the read-only /host mount."""
    if not filepath or not str(filepath).strip():
        filepath = "/host/etc/resolv.conf"
    path = str(filepath)
    if not path.startswith("/host"):
        path = os.path.join("/host", path.lstrip("/"))
    safe = os.path.abspath(path)
    if os.path.commonpath(["/host", safe]) != "/host":
        return "Error: Path traversal outside /host is forbidden."
    try:
        text = Path(safe).read_text(errors="ignore")[:20000]
        return text + ("\n[truncated]" if len(text) >= 20000 else "")
    except Exception as exc:
        return f"Error reading host file: {exc}"


def read_host_journal(lines: int = 50, service: str = "", grep: str = "", priority: str = "") -> str:
    """Read recent host systemd journal entries through the read-only log mount when available."""
    lines = max(1, min(int(lines), 300))
    cmd = ["journalctl", "-D", "/host_log/journal", "--no-pager", "-n", str(lines)]
    if service.strip():
        cmd.extend(["-u", service.strip()])
    if grep.strip():
        cmd.extend(["-g", grep.strip()])
    if priority.strip():
        cmd.extend(["-p", priority.strip()])
    code, stdout, stderr = _run(cmd, timeout=15)
    if code != 0 and not stdout.strip():
        return f"Error reading journal: {stderr.strip()}"
    return stdout.strip() or "No logs found."


def tail_host_log(log_path: str = "syslog", lines: int = 50) -> str:
    """Read the tail of a file under /host_log, falling back to journalctl."""
    lines = max(1, min(int(lines), 300))
    raw = str(log_path or "syslog")
    if raw.startswith("/var/log/"):
        raw = raw.replace("/var/log/", "/host_log/", 1)
    if not raw.startswith("/host_log"):
        raw = os.path.join("/host_log", raw.lstrip("/"))
    safe = os.path.abspath(raw)
    if os.path.commonpath(["/host_log", safe]) != "/host_log":
        return "Error: Path traversal outside /host_log is forbidden."
    try:
        data = Path(safe).read_text(errors="ignore").splitlines()
        return "\n".join(data[-lines:])
    except Exception:
        return read_host_journal(lines=lines)


def record_network_change(snapshot_json: str) -> str:
    """Persist a network snapshot and notify only when the normalized topology changes."""
    try:
        snapshot = json.loads(snapshot_json)
    except json.JSONDecodeError:
        return "Error: snapshot_json must be valid JSON."
    normalized = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
    previous = get_monitor_state("network.snapshot")
    changed = previous != normalized
    record_monitor_state("network.snapshot", normalized)
    if changed and previous is not None:
        record_monitor_event("network_change", "Network topology snapshot changed.", {"snapshot": snapshot})
    return json.dumps({"changed": changed})
