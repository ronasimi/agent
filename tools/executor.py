"""Single execution path for registered tools.

Every caller (interactive loop, recipes, pipelines) goes through this module so
timeouts and future execution policy cannot drift between orchestration paths.
Timeout-decorated Python/custom tools execute in a single-use subprocess; the
shared subprocess runner can therefore terminate the entire process group on a
wall-clock timeout instead of leaving an unkillable CPython worker thread.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from .subprocess_utils import run_argv

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _execute_isolated_tool(name: str, args: dict[str, Any], seconds: int) -> Any:
    """Execute one registered tool in a killable child interpreter."""
    with tempfile.TemporaryDirectory(prefix="agent-tool-") as temp_dir:
        request_path = Path(temp_dir) / "request.json"
        result_path = Path(temp_dir) / "result.json"
        request_path.write_text(
            json.dumps({"name": name, "args": args}, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        try:
            os.chmod(request_path, 0o600)
        except OSError:
            pass

        proc = run_argv(
            [sys.executable, "-m", "tools.executor_worker", str(request_path), str(result_path)],
            cwd=str(_REPO_ROOT),
            timeout=seconds,
            env=os.environ.copy(),
            max_output_bytes=262144,
        )
        if proc.timed_out:
            raise TimeoutError(
                f"Tool '{name}' exceeded its {seconds}-second harness timeout; "
                "the isolated tool process group was terminated."
            )
        if not result_path.exists():
            detail = (proc.stderr or proc.stdout or f"isolated worker exited with status {proc.returncode}").strip()
            raise RuntimeError(f"Tool '{name}' isolated worker failed without a result: {detail[:1200]}")

        try:
            packet = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Tool '{name}' isolated worker returned an invalid result: {exc}") from exc
        if packet.get("ok") is True:
            return packet.get("value")
        error_type = str(packet.get("error_type") or "RuntimeError")
        message = str(packet.get("error") or "isolated tool failed")
        raise RuntimeError(f"{error_type}: {message}")


def execute_registered_tool(name: str, args: dict[str, Any]) -> Any:
    # Import lazily to avoid a catalog -> executor -> catalog cycle at startup.
    from .catalog import AVAILABLE_TOOLS_MAP, TOOL_METADATA

    func = AVAILABLE_TOOLS_MAP[name]
    timeout = TOOL_METADATA.get(name, {}).get("timeout")
    if not timeout:
        return func(**args)
    seconds = max(1, min(int(timeout), 300))
    return _execute_isolated_tool(name, args, seconds)
