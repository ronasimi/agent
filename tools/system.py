"""Privileged-in-container execution helpers.

These tools deliberately never infer a missing command from model prose. The
agent must issue an explicit native tool call with a concrete argument.
"""
from __future__ import annotations

import os
import subprocess
import tempfile

WORKSPACE_DIR = "/app/workspace"
os.makedirs(WORKSPACE_DIR, exist_ok=True)


_FORBIDDEN_SHELL_PATTERNS = (
    "/host", "/host_log", "nsenter", "docker", "podman",
    "systemctl --user", "loginctl", "machinectl", "mount ", "umount ",
)


def _shell_policy_violation(command: str) -> str | None:
    lowered = str(command).lower()
    for pattern in _FORBIDDEN_SHELL_PATTERNS:
        if pattern.lower() in lowered:
            return f"Generic shell access to host-control path/command '{pattern}' is blocked; use a dedicated typed tool."
    return None


def execute_shell(command: str = "", timeout: int = 30) -> str:
    """Execute an explicit shell command inside the agent container workspace; no command auto-recovery is performed."""
    if not str(command).strip():
        return "Error: Missing required 'command' parameter."
    violation = _shell_policy_violation(command)
    if violation:
        return "Error: " + violation
    timeout = max(1, min(int(timeout), 120))
    try:
        result = subprocess.run(
            ["bash", "-lc", str(command)],
            cwd=WORKSPACE_DIR,
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
            env=os.environ.copy(),
        )
        output = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}".strip()
        return output or "Command completed successfully with no output."
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout} seconds."
    except Exception as exc:
        return f"Execution error: {exc}"


def execute_python(code: str = "", timeout: int = 30) -> str:
    """Execute explicit Python code inside the agent container workspace."""
    if not str(code).strip():
        return "Error: Missing required 'code' parameter."
    violation = _shell_policy_violation(code)
    if violation:
        return "Error: " + violation
    timeout = max(1, min(int(timeout), 120))
    script_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=WORKSPACE_DIR, suffix=".py", mode="w", delete=False, encoding="utf-8") as handle:
            handle.write(str(code))
            script_path = handle.name
        result = subprocess.run(
            ["python", script_path],
            cwd=WORKSPACE_DIR,
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
            env=os.environ.copy(),
        )
        output = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}".strip()
        return output or "Python execution completed with no output."
    except subprocess.TimeoutExpired:
        return f"Error: Python execution timed out after {timeout} seconds."
    except Exception as exc:
        return f"Python execution error: {exc}"
    finally:
        if script_path:
            try:
                os.remove(script_path)
            except OSError:
                pass
