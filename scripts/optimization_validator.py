#!/usr/bin/env python3
"""Networkless validation runner for model-generated optimization candidates."""
from __future__ import annotations

import json
import os
import resource
import signal
import shutil
import subprocess
import time
from pathlib import Path

import yaml

CONFIG_PATH = Path(os.environ.get("AGENT_CONFIG", "/app/config/config.yaml"))
INBOX = Path(os.environ.get("OPTIMIZER_VALIDATION_INBOX", "/validation/inbox"))
OUTBOX = Path(os.environ.get("OPTIMIZER_VALIDATION_OUTBOX", "/validation/outbox"))
CANDIDATES = Path(os.environ.get("OPTIMIZER_CANDIDATES", "/candidates"))


def load_policy() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return (yaml.safe_load(handle) or {}).get("self_optimization", {})


def commands(policy: dict) -> list[list[str]]:
    result = []
    for key in ("test_commands", "benchmark_commands"):
        for command in policy.get(key) or []:
            if isinstance(command, list) and command and all(isinstance(value, (str, int, float)) for value in command):
                result.append([str(value) for value in command])
    return result


def drop_privileges(memory_limit_mb: int, timeout: int) -> None:
    os.setgroups([])
    os.setgid(65534)
    os.setuid(65534)
    memory_bytes = max(256, int(memory_limit_mb)) * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    cpu_seconds = max(2, int(timeout))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 2))


def run_command(argv: list[str], root: Path, policy: dict) -> dict:
    timeout = max(1, int(policy.get("command_timeout_seconds", 300)))
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            argv,
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env={
                "HOME": "/tmp/validator-home",
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                "PYTHONPATH": str(root),
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONUNBUFFERED": "1",
            },
            preexec_fn=lambda: drop_privileges(int(policy.get("test_memory_limit_mb", 1536)), timeout),
            start_new_session=True,
        )
        output, _ = process.communicate(timeout=timeout)
        return {
            "command": argv,
            "returncode": process.returncode,
            "seconds": round(time.monotonic() - started, 3),
            "output": (output or "")[-12000:],
            "passed": process.returncode == 0,
        }
    except subprocess.TimeoutExpired as exc:
        os.killpg(process.pid, signal.SIGKILL)
        trailing, _ = process.communicate()
        output = exc.stdout if isinstance(exc.stdout, str) else ""
        return {
            "command": argv,
            "returncode": 124,
            "seconds": round(time.monotonic() - started, 3),
            "output": (output + (trailing or ""))[-12000:] + "\nTimed out.",
            "passed": False,
        }


def validate_request(request: dict, policy: dict) -> dict:
    request_id = str(request.get("request_id") or "")
    candidate_id = str(request.get("candidate_id") or "")
    if not request_id.isalnum() or not candidate_id or any(char not in "0123456789abcdef-" for char in candidate_id.lower()):
        raise ValueError("Invalid validation identifiers.")
    source = (CANDIDATES / candidate_id / "repo").resolve()
    if os.path.commonpath([str(CANDIDATES.resolve()), str(source)]) != str(CANDIDATES.resolve()) or not source.is_dir():
        raise ValueError("Candidate repository is unavailable.")
    for path in source.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Candidate contains a prohibited symlink: {path.relative_to(source)}")

    scratch = Path("/tmp") / f"validation-{request_id}"
    if scratch.exists():
        shutil.rmtree(scratch)
    shutil.copytree(source, scratch)
    for path in [scratch, *scratch.rglob("*")]:
        try:
            os.chown(path, 65534, 65534)
        except FileNotFoundError:
            pass
    Path("/tmp/validator-home").mkdir(mode=0o700, exist_ok=True)
    os.chown("/tmp/validator-home", 65534, 65534)
    results = [run_command(command, scratch, policy) for command in commands(policy)]
    shutil.rmtree(scratch, ignore_errors=True)
    return {
        "request_id": request_id,
        "candidate_id": candidate_id,
        "phase": str(request.get("phase") or "validation"),
        "passed": bool(results) and all(result["passed"] for result in results),
        "commands": results,
    }


def write_result(request_id: str, result: dict) -> None:
    temporary = OUTBOX / f".{request_id}.tmp"
    destination = OUTBOX / f"{request_id}.json"
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.chown(temporary, 1000, 1000)
    os.chmod(temporary, 0o600)
    temporary.replace(destination)


def main() -> None:
    policy = load_policy()
    OUTBOX.mkdir(parents=True, exist_ok=True)
    os.chown(OUTBOX, 0, 0)
    os.chmod(OUTBOX, 0o711)
    print("[optimizer-validator] ready; network is disabled by Compose")
    while True:
        handled = False
        for request_path in sorted(INBOX.glob("*.json")):
            request_id = request_path.stem
            result_path = OUTBOX / request_path.name
            if result_path.exists():
                continue
            handled = True
            try:
                request = json.loads(request_path.read_text(encoding="utf-8"))
                result = validate_request(request, policy)
            except Exception as exc:
                result = {"request_id": request_id, "passed": False, "commands": [], "error": str(exc)}
            write_result(request_id, result)
        if not handled:
            time.sleep(0.5)


if __name__ == "__main__":
    main()
