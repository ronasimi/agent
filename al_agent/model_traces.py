"""Append-only per-model-call traces for evaluation and local fine-tuning.

The writer is deliberately independent of the turn engine and Ollama client. A
single locked append keeps the foreground overhead tiny, while bounded rotation
prevents an unattended local harness from growing the trace file forever.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from tools.security import redact_diagnostic_value

_LOCK = threading.Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _rotate(path: Path, max_bytes: int) -> None:
    if max_bytes <= 0 or not path.exists():
        return
    try:
        if path.stat().st_size < max_bytes:
            return
    except OSError:
        return
    previous = path.with_suffix(path.suffix + ".1")
    try:
        previous.unlink(missing_ok=True)
        os.replace(path, previous)
    except OSError:
        pass


def record_model_trace(
    *,
    path: str,
    enabled: bool,
    max_bytes: int,
    conversation_id: str,
    turn_id: int,
    call_index: int,
    model: str,
    role: str,
    purpose: str,
    thinking_enabled: bool,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    options: dict[str, Any],
    completion: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    error: str = "",
    request_extra: dict[str, Any] | None = None,
) -> None:
    """Write one request/completion pair as JSONL, redacting credential forms.

    Records contain the actual wire messages/tool schemas/options after harness
    compaction and Qwen schema adaptation. This makes traces suitable for replay,
    regression tests, and later distillation of the harness's real workload.
    """
    if not enabled:
        return
    target = Path(str(path or "/app/memory/model_calls.jsonl"))
    record = {
        "version": 1,
        "at": _utc_now(),
        "conversation_id": str(conversation_id),
        "turn_id": int(turn_id),
        "call_index": int(call_index),
        "model": str(model),
        "role": str(role),
        "purpose": str(purpose),
        "thinking_enabled": bool(thinking_enabled),
        "request": {
            **(request_extra or {}),
            "messages": messages,
            "tools": tools,
            "options": options,
        },
        "completion": completion or {},
        "metrics": metrics or {},
        "error": str(error or ""),
    }
    encoded = (
        json.dumps(redact_diagnostic_value(record), ensure_ascii=False, separators=(",", ":"), default=str)
        + "\n"
    )
    with _LOCK:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            _rotate(target, max(1024 * 1024, int(max_bytes)))
            with os.fdopen(os.open(target, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600), "a", encoding="utf-8") as handle:
                os.fchmod(handle.fileno(), 0o600)
                handle.write(encoded)
        except OSError:
            # Tracing must never fail an interactive turn.
            return
