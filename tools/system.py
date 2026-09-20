"""Privileged-in-container execution helpers.

These tools deliberately never infer a missing command from model prose. The
agent must issue an explicit native tool call with a concrete argument.
"""
from __future__ import annotations

import os
import tempfile

from .subprocess_utils import run_argv

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


def _format_process_output(stdout: str, stderr: str, returncode: int | None) -> str:
    code = -1 if returncode is None else int(returncode)
    return f"EXIT_CODE: {code}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}".strip()

def execute_shell(command: str = "", timeout: int = 30) -> str:
    """Execute an explicit shell command inside the agent container workspace; no command auto-recovery is performed."""
    if not str(command).strip():
        return "Error: Missing required 'command' parameter."
    violation = _shell_policy_violation(command)
    if violation:
        return "Error: " + violation
    timeout = max(1, min(int(timeout), 120))
    try:
        proc = run_argv(
            ["bash", "--noprofile", "--norc", "-c", str(command)],
            cwd=WORKSPACE_DIR, timeout=timeout, env=os.environ.copy(),
        )
        returncode = proc.returncode
        output = _format_process_output(proc.stdout, proc.stderr, returncode)
        if proc.timed_out:
            return f"Error: Command timed out after {timeout} seconds; the command process group was terminated.\n{output}"
        if int(returncode or 0) != 0:
            if proc.stdout.strip():
                return f"Partial: command exited with status {returncode} but produced usable stdout.\n{output}"
            return f"Error: command exited with status {returncode}.\n{output}"
        return output
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
        proc = run_argv(["python", script_path], cwd=WORKSPACE_DIR, timeout=timeout, env=os.environ.copy())
        returncode = proc.returncode
        output = _format_process_output(proc.stdout, proc.stderr, returncode)
        if proc.timed_out:
            return f"Error: Python execution timed out after {timeout} seconds; the process group was terminated.\n{output}"
        if int(returncode or 0) != 0:
            if proc.stdout.strip():
                return f"Partial: Python exited with status {returncode} but produced usable stdout.\n{output}"
            return f"Error: Python exited with status {returncode}.\n{output}"
        return output
    except Exception as exc:
        return f"Python execution error: {exc}"
    finally:
        if script_path:
            try:
                os.remove(script_path)
            except OSError:
                pass

