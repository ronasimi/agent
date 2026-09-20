"""Bounded subprocess execution shared by tool backends.

External utilities are treated as untrusted: output is capped in memory, UTF-8
is decoded lossily, and POSIX descendants are killed as a process group on
 timeout.  This keeps one broken binary from wedging or exhausting the agent.
"""
from __future__ import annotations

import os
import signal
import subprocess
import threading
from dataclasses import dataclass
from typing import Mapping, Sequence

DEFAULT_MAX_OUTPUT_BYTES = 1_048_576  # 1 MiB per stream


@dataclass(frozen=True)
class ProcessResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    stdout_truncated: bool = False
    stderr_truncated: bool = False


def _decode(data: bytes) -> str:
    return bytes(data or b"").decode("utf-8", errors="replace")


def run_argv(
    argv: Sequence[str],
    *,
    cwd: str | None = None,
    timeout: float = 10,
    env: Mapping[str, str] | None = None,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
) -> ProcessResult:
    """Run argv with bounded capture and descendant-safe timeout handling."""
    limit = max(4096, min(int(max_output_bytes), 16 * 1024 * 1024))
    proc = subprocess.Popen(
        [str(item) for item in argv],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        env=dict(env) if env is not None else None,
        start_new_session=(os.name == "posix"),
    )
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    truncated = {"stdout": False, "stderr": False}

    def drain(stream, key: str) -> None:
        if stream is None:
            return
        try:
            while True:
                chunk = stream.read(65536)
                if not chunk:
                    break
                remaining = limit - len(buffers[key])
                if remaining > 0:
                    buffers[key].extend(chunk[:remaining])
                if len(chunk) > max(remaining, 0):
                    truncated[key] = True
        except Exception:
            # The process result/return code is still useful if a reader loses
            # a race with forced descriptor closure during timeout cleanup.
            pass

    readers = [
        threading.Thread(target=drain, args=(proc.stdout, "stdout"), daemon=True),
        threading.Thread(target=drain, args=(proc.stderr, "stderr"), daemon=True),
    ]
    for reader in readers:
        reader.start()

    timed_out = False
    try:
        proc.wait(timeout=max(0.1, float(timeout)))
    except subprocess.TimeoutExpired:
        timed_out = True
        if os.name == "posix":
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            proc.kill()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    finally:
        for reader in readers:
            reader.join(timeout=2)
        for stream in (proc.stdout, proc.stderr):
            try:
                if stream is not None:
                    stream.close()
            except Exception:
                pass

    stdout = _decode(bytes(buffers["stdout"]))
    stderr = _decode(bytes(buffers["stderr"]))
    if truncated["stdout"]:
        stdout += f"\n[stdout truncated after {limit} bytes]"
    if truncated["stderr"]:
        stderr += f"\n[stderr truncated after {limit} bytes]"
    return ProcessResult(
        returncode=int(proc.returncode if proc.returncode is not None else -1),
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
        stdout_truncated=truncated["stdout"],
        stderr_truncated=truncated["stderr"],
    )
