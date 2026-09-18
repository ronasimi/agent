"""Deterministic repository and harness health checks."""
from __future__ import annotations

import ast
import importlib.metadata
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import yaml

from .repo_map import source_root, iter_source_files


def _run(argv: list[str], cwd: Path, timeout: int = 60, env: dict | None = None) -> dict:
    try:
        proc = subprocess.run(argv, cwd=str(cwd), capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL, env=env)
        return {"command": argv, "returncode": proc.returncode, "stdout": proc.stdout[-8000:], "stderr": proc.stderr[-4000:], "ok": proc.returncode == 0}
    except subprocess.TimeoutExpired:
        return {"command": argv, "returncode": None, "ok": False, "error": f"timeout after {timeout}s"}
    except Exception as exc:
        return {"command": argv, "returncode": None, "ok": False, "error": str(exc)}


def repo_status() -> str:
    """Return bounded Git status for the immutable source snapshot."""
    root = source_root()
    if not (root / ".git").exists():
        return json.dumps({"root": str(root), "git_repo": False, "note": "No .git directory is mounted in the source snapshot."}, indent=2)
    result = _run(["git", "status", "--porcelain=v1", "-b"], root, timeout=10)
    result["root"] = str(root)
    return json.dumps(result, ensure_ascii=False, indent=2)


def repo_diff(max_chars: int = 20000) -> str:
    """Return a bounded Git diff and staged diff for the immutable source snapshot."""
    max_chars = max(1000, min(int(max_chars), 50000))
    root = source_root()
    if not (root / ".git").exists():
        return json.dumps({"root": str(root), "git_repo": False, "diff": ""}, indent=2)
    work = _run(["git", "diff", "--no-ext-diff", "--"], root, timeout=15)
    staged = _run(["git", "diff", "--cached", "--no-ext-diff", "--"], root, timeout=15)
    return json.dumps({
        "root": str(root),
        "working_diff": (work.get("stdout") or "")[:max_chars],
        "staged_diff": (staged.get("stdout") or "")[:max_chars],
        "working_ok": work.get("ok"), "staged_ok": staged.get("ok"),
    }, ensure_ascii=False, indent=2)


def _copy_source(destination: Path) -> None:
    root = source_root()
    for rel, path in iter_source_files(root):
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    for special in ("Dockerfile", "requirements.txt", "docker-compose.yml", "README.md", "agent.py", "worker.py"):
        src = root / special
        if src.is_file():
            dst = destination / special; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)


def repo_checks(checks: list[str] | None = None, timeout: int = 180) -> str:
    """Run known compile/config/lint/test checks against a temporary repository copy; arbitrary commands are not accepted."""
    requested = [str(x).lower().strip() for x in (checks or ["compile", "config", "ruff", "pytest"])]
    allowed = {"compile", "config", "ruff", "pytest"}
    unknown = [x for x in requested if x not in allowed]
    if unknown:
        return f"Error: unsupported checks: {', '.join(unknown)}"
    timeout = max(10, min(int(timeout), 300))
    results = []
    with tempfile.TemporaryDirectory(prefix="agent-repo-checks-") as temp:
        root = Path(temp)
        _copy_source(root)
        if "compile" in requested:
            errors = []
            for path in root.rglob("*.py"):
                try: ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
                except SyntaxError as exc: errors.append(f"{path.relative_to(root)}:{exc.lineno}:{exc.msg}")
            results.append({"check": "compile", "ok": not errors, "errors": errors[:100]})
        if "config" in requested:
            errors = []
            for path in list(root.rglob("*.yaml")) + list(root.rglob("*.yml")):
                try: yaml.safe_load(path.read_text(encoding="utf-8", errors="replace"))
                except Exception as exc: errors.append(f"{path.relative_to(root)}: {exc}")
            results.append({"check": "config", "ok": not errors, "errors": errors[:50]})
        if "ruff" in requested:
            if shutil.which("ruff"):
                item = _run(["ruff", "check", "."], root, timeout=min(timeout, 120)); item["check"] = "ruff"; results.append(item)
            else:
                results.append({"check": "ruff", "ok": False, "unavailable": True, "error": "ruff is not installed"})
        if "pytest" in requested:
            env = os.environ.copy(); env["PYTHONDONTWRITEBYTECODE"] = "1"; env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
            item = _run(["python", "-m", "pytest", "-q", "-p", "no:cacheprovider"], root, timeout=timeout, env=env); item["check"] = "pytest"; results.append(item)
    return json.dumps({"ok": all(bool(item.get("ok")) for item in results), "checks": results}, ensure_ascii=False, indent=2)


def dependency_audit() -> str:
    """Audit required Python packages and command-line utilities used by the harness and diagnostics."""
    root = source_root()
    requirements = []
    req = root / "requirements.txt"
    if req.is_file():
        for line in req.read_text(errors="ignore").splitlines():
            value = line.strip()
            if value and not value.startswith("#"):
                requirements.append(value.split("[", 1)[0].split("=", 1)[0].split(">", 1)[0].split("<", 1)[0].strip())
    py_rows = []
    for name in requirements:
        try:
            py_rows.append({"name": name, "installed": True, "version": importlib.metadata.version(name)})
        except importlib.metadata.PackageNotFoundError:
            py_rows.append({"name": name, "installed": False})
    binaries = ["git", "bwrap", "pdftotext", "pdfinfo", "nmap", "dig", "mtr", "ss", "ip", "journalctl", "systemctl", "dot", "ruff"]
    bin_rows = [{"name": name, "path": shutil.which(name) or "", "available": bool(shutil.which(name))} for name in binaries]
    pip_check = _run(["python", "-m", "pip", "check"], root, timeout=30)
    return json.dumps({
        "python_packages": py_rows,
        "binaries": bin_rows,
        "pip_check": {"ok": pip_check.get("ok"), "stdout": pip_check.get("stdout", "")[:4000], "stderr": pip_check.get("stderr", "")[:2000]},
    }, ensure_ascii=False, indent=2)


def tool_health() -> str:
    """Report registered-tool availability, metadata, and known external dependencies without invoking the tools."""
    # Runtime import avoids circular registry initialization.
    from . import AVAILABLE_TOOLS_MAP, TOOL_METADATA
    binary_deps = {
        "dns_diagnose": ["dig"], "network_path": ["mtr"], "connection_snapshot": ["ss"], "neighbor_snapshot": ["ip"],
        "map_network": ["nmap", "dot"], "extract_document": ["pdftotext", "pdfinfo"], "repo_status": ["git"],
        "repo_diff": ["git"], "repo_checks": ["ruff"], "read_host_journal": ["journalctl"], "service_health": ["systemctl", "journalctl"],
    }
    rows = []
    healthy = degraded = unavailable = 0
    for name in sorted(AVAILABLE_TOOLS_MAP):
        missing = [dep for dep in binary_deps.get(name, []) if not shutil.which(dep)]
        status = "healthy"
        if missing:
            # Some tools explicitly degrade/fallback rather than becoming unusable.
            status = "degraded" if name in {"service_health", "repo_checks"} else "unavailable"
        if status == "healthy": healthy += 1
        elif status == "degraded": degraded += 1
        else: unavailable += 1
        rows.append({
            "name": name, "status": status, "readonly": bool(TOOL_METADATA.get(name, {}).get("readonly", True)),
            "repeat_safe": bool(TOOL_METADATA.get(name, {}).get("repeat_safe", False)), "missing_dependencies": missing,
        })
    return json.dumps({"registered": len(rows), "healthy": healthy, "degraded": degraded, "unavailable": unavailable, "tools": rows}, ensure_ascii=False, indent=2)
