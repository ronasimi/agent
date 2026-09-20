"""Privileged-in-container execution helpers.

These tools deliberately never infer a missing command from model prose. The
agent must issue an explicit native tool call with a concrete argument.
"""
from __future__ import annotations

import os
import signal
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


def _run_process(args: list[str], *, cwd: str, timeout: int) -> tuple[bytes, bytes, int | None, bool]:
    """Run a subprocess as its own process group and kill the whole tree on timeout."""
    proc = subprocess.Popen(
        args,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        env=os.environ.copy(),
        start_new_session=(os.name == "posix"),
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return stdout or b"", stderr or b"", proc.returncode, False
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            proc.kill()
        stdout, stderr = proc.communicate()
        return stdout or b"", stderr or b"", proc.returncode, True


def _decode_output(value: bytes) -> str:
    return bytes(value or b"").decode("utf-8", errors="replace")


def _format_process_output(stdout: bytes, stderr: bytes, returncode: int | None) -> str:
    code = -1 if returncode is None else int(returncode)
    return f"EXIT_CODE: {code}\nSTDOUT:\n{_decode_output(stdout)}\nSTDERR:\n{_decode_output(stderr)}".strip()


def execute_shell(command: str = "", timeout: int = 30) -> str:
    """Execute an explicit shell command inside the agent container workspace; no command auto-recovery is performed."""
    if not str(command).strip():
        return "Error: Missing required 'command' parameter."
    violation = _shell_policy_violation(command)
    if violation:
        return "Error: " + violation
    timeout = max(1, min(int(timeout), 120))
    try:
        stdout, stderr, returncode, timed_out = _run_process(
            ["bash", "--noprofile", "--norc", "-c", str(command)],
            cwd=WORKSPACE_DIR,
            timeout=timeout,
        )
        output = _format_process_output(stdout, stderr, returncode)
        if timed_out:
            return f"Error: Command timed out after {timeout} seconds; the command process group was terminated.\n{output}"
        if int(returncode or 0) != 0:
            if _decode_output(stdout).strip():
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
        stdout, stderr, returncode, timed_out = _run_process(
            ["python", script_path], cwd=WORKSPACE_DIR, timeout=timeout
        )
        output = _format_process_output(stdout, stderr, returncode)
        if timed_out:
            return f"Error: Python execution timed out after {timeout} seconds; the process group was terminated.\n{output}"
        if int(returncode or 0) != 0:
            if _decode_output(stdout).strip():
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

