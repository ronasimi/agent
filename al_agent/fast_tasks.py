"""Bounded fast-model classification/extraction helpers.

These helpers never authorize actions. They produce small advisory JSON that is
validated by deterministic harness code before use. Failure is always a cheap
fallback to deterministic behavior.
"""
from __future__ import annotations

import json
import re
from typing import Any

from .model_capabilities import capability_chat_overrides


def _message_content(response: Any) -> str:
    if isinstance(response, dict):
        msg = response.get("message") or {}
        if isinstance(msg, dict):
            return str(msg.get("content") or "")
        return str(response.get("response") or "")
    msg = getattr(response, "message", None)
    if msg is not None:
        return str(getattr(msg, "content", "") or "")
    return str(getattr(response, "response", "") or "")


def _extract_json(text: str) -> Any:
    value = str(text or "").strip()
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass
    match = re.search(r"(?:```json\s*)?(\{.*\}|\[.*\])(?:\s*```)?", value, re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def infer_recipe_parameter_hints(
    client: Any,
    *,
    model: str,
    objective: str,
    trace: list[dict[str, Any]],
    options: dict[str, Any] | None = None,
    keep_alive: Any = -1,
) -> list[dict[str, Any]]:
    """Ask the fast role for semantic names only; deterministic code decides use.

    The fast model is intentionally prevented from rewriting the pipeline. It may
    only point at literal values already present in the successful trace and
    suggest a semantic parameter name.
    """
    compact = []
    for entry in trace[:8]:
        if not entry.get("success") or not entry.get("readonly", True):
            continue
        compact.append({"tool": str(entry.get("tool") or ""), "args": entry.get("args") or {}})
    if len(compact) < 2:
        return []
    prompt = (
        "Classify reusable task inputs in this successful read-only workflow. "
        "Return JSON only: {\"parameters\":[{\"name\":\"...\",\"value\":...}]}. "
        "Only copy literal values that appear in the supplied tool arguments. "
        "Prefer task-defining values (hostname, location, query, path, symbol). "
        "Do not suggest timeouts, limits, booleans, port 443 unless user explicitly requested the port, "
        "credentials, tokens, secrets, timestamps, observation IDs, or tool outputs. "
        "When a hostname also appears inside URLs, suggest only the hostname; deterministic code will derive URLs.\n"
        f"Objective: {str(objective)[:1600]}\n"
        f"Workflow: {json.dumps(compact, ensure_ascii=False, default=str)[:5000]}"
    )
    fast_options = dict(options or {})
    fast_options["temperature"] = 0.0
    fast_options["num_predict"] = min(int(fast_options.get("num_predict") or 160), 192)
    try:
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            options=fast_options,
            keep_alive=keep_alive,
            **capability_chat_overrides(model, think=False, tools=[]),
        )
    except Exception:
        return []
    parsed = _extract_json(_message_content(response))
    rows = parsed.get("parameters") if isinstance(parsed, dict) else None
    if not isinstance(rows, list):
        return []
    out = []
    for row in rows[:16]:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        if not name or "value" not in row:
            continue
        out.append({"name": name[:48], "value": row.get("value")})
    return out
