"""Single-use subprocess entry point for timeout-bounded registered tools.

The parent executor owns the wall-clock timeout and kills this entire process
session on expiry.  The worker intentionally calls the registered function
directly so it cannot recurse back through :mod:`tools.executor`.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any


def _write_result(path: str, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    data = json.dumps(payload, ensure_ascii=False, default=str)
    temporary.write_text(data, encoding="utf-8")
    os.replace(temporary, target)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        return 64
    request_path, result_path = args
    try:
        request = json.loads(Path(request_path).read_text(encoding="utf-8"))
        name = str(request.get("name") or "")
        tool_args = request.get("args") or {}
        if not name or not isinstance(tool_args, dict):
            raise ValueError("invalid isolated-tool request")

        from .catalog import AVAILABLE_TOOLS_MAP, load_tools

        load_tools()
        func = AVAILABLE_TOOLS_MAP.get(name)
        if func is None:
            raise KeyError(f"registered tool '{name}' was not available in isolated worker")
        from .conversation_context import conversation_context
        with conversation_context(request.get("conversation_id")):
            value = func(**tool_args)
        _write_result(result_path, {"ok": True, "value": value})
        return 0
    except BaseException as exc:
        try:
            _write_result(
                result_path,
                {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=8),
                },
            )
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
