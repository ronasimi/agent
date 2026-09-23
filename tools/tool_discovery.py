"""Progressive discovery over the complete native tool catalog."""
from __future__ import annotations

import json
import re


def _tokens(value: str) -> set[str]:
    return {x for x in re.findall(r"[a-z0-9]+", str(value or "").lower().replace("_", " ")) if len(x) > 1}


def tool_search(query: str = "", limit: int = 5) -> str:
    """Find relevant available tool names without executing them; use when the current tool set lacks the needed capability."""
    query = " ".join(str(query or "").split())
    if not query:
        return "Error: Missing required 'query' parameter."
    try:
        limit = max(1, min(int(limit), 8))
    except (TypeError, ValueError):
        limit = 5
    # Local import avoids a catalog import cycle during builtin discovery.
    from .catalog import TOOL_METADATA, TOOL_SCHEMAS

    q = _tokens(query)
    ranked: list[tuple[int, str, dict]] = []
    for schema in TOOL_SCHEMAS:
        fn = schema.get("function", {}) if isinstance(schema, dict) else {}
        name = str(fn.get("name") or "")
        if not name or name == "tool_search":
            continue
        description = str(fn.get("description") or "")
        name_tokens = _tokens(name)
        desc_tokens = _tokens(description)
        score = 6 * len(q & name_tokens) + 2 * len(q & desc_tokens)
        if query.lower() in name.lower().replace("_", " "):
            score += 8
        if score <= 0:
            continue
        params = fn.get("parameters", {}) if isinstance(fn.get("parameters"), dict) else {}
        ranked.append((score, name, {
            "name": name,
            "description": " ".join(description.split())[:260],
            "readonly": bool(TOOL_METADATA.get(name, {}).get("readonly", True)),
            "required": [str(x) for x in params.get("required", [])[:8]],
        }))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return json.dumps([item for _score, _name, item in ranked[:limit]], ensure_ascii=False, separators=(",", ":"))
