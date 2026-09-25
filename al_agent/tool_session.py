"""Per-turn tool discovery and strict validation, with no user-intent routing."""

from __future__ import annotations

import copy
import json
import re
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from jsonschema import Draft202012Validator


def _coerce_schema_value(value: Any, schema: dict) -> Any:
    """Deterministically coerce Qwen XML parameter text from its tool schema.

    XML tool calls carry parameter bodies as text.  The schema is authoritative:
    strings remain byte-for-byte strings, while integer/number/boolean and JSON
    array/object parameters are parsed only when the schema requires that type.
    """
    if not isinstance(value, str) or not isinstance(schema, dict):
        return value

    branches = schema.get("oneOf") or schema.get("anyOf")
    if isinstance(branches, list):
        # If string is explicitly accepted, preserve it rather than guessing.
        if any(branch.get("type") == "string" for branch in branches if isinstance(branch, dict)):
            return value
        for branch in branches:
            if not isinstance(branch, dict):
                continue
            candidate = _coerce_schema_value(value, branch)
            if not list(Draft202012Validator(branch).iter_errors(candidate)):
                return candidate
        return value

    kind = schema.get("type")
    if isinstance(kind, list):
        if "string" in kind:
            return value
        kinds = [item for item in kind if item != "null"]
        if len(kinds) == 1:
            kind = kinds[0]

    if kind == "string" or kind is None:
        return value
    stripped = value.strip()
    if kind == "integer":
        if not re.fullmatch(r"[-+]?\d+", stripped):
            return value
        return int(stripped, 10)
    if kind == "number":
        try:
            parsed = json.loads(stripped)
        except (TypeError, ValueError, json.JSONDecodeError):
            return value
        return parsed if isinstance(parsed, (int, float)) and not isinstance(parsed, bool) else value
    if kind == "boolean":
        lowered = stripped.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        return value
    if kind == "null":
        return None if stripped.lower() == "null" else value
    if kind in {"array", "object"}:
        try:
            parsed = json.loads(stripped)
        except (TypeError, ValueError, json.JSONDecodeError):
            return value
        if kind == "array" and isinstance(parsed, list):
            return parsed
        if kind == "object" and isinstance(parsed, dict):
            return parsed
    return value


def validate_arguments(schema: dict, arguments: Any) -> dict:
    """Validate exactly what the model supplied; never manufacture parameters."""
    if not isinstance(arguments, dict):
        raise TypeError("Tool arguments must be a JSON object")
    parameters = copy.deepcopy(schema["function"].get("parameters", {"type": "object"}))
    parameters.setdefault("additionalProperties", False)
    properties = parameters.get("properties") if isinstance(parameters.get("properties"), dict) else {}
    normalized = {
        key: _coerce_schema_value(value, properties.get(key, {}))
        for key, value in arguments.items()
    }
    # JSON Schema treats Python NaN as a number; JSON itself does not permit it.
    json.dumps(normalized, allow_nan=False)
    errors = list(Draft202012Validator(parameters).iter_errors(normalized))
    if errors:
        error = errors[0]
        path = ".".join(map(str, error.absolute_path)) or "arguments"
        raise ValueError(f"{path}: {error.message}")
    return copy.deepcopy(normalized)


def _schema(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


DISCOVERY_SCHEMAS = [
    _schema(
        "tool_search",
        "Search the entire available tool catalog by name or description. "
        "Returns full schemas and activates the matches. Empty query lists tools alphabetically; "
        "use offset to browse further. This does not execute the discovered tools.",
        {
            "query": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 6},
            "offset": {"type": "integer", "minimum": 0},
        },
        [],
    ),
    _schema(
        "load_tools",
        "Load full schemas for exact tool names from the available catalog. "
        "This only activates tools; call them separately to perform work.",
        {
            "names": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 6,
                "uniqueItems": True,
            }
        },
        ["names"],
    ),
]


class ToolSession:
    """Snapshot registry definitions so concurrent reloads cannot change a turn."""

    def __init__(
        self,
        schemas: list[dict],
        execute: Callable[[str, dict], Any],
        metadata: dict | None = None,
        max_active: int = 16,
        max_schema_chars: int = 20000,
    ):
        self.catalog = {s["function"]["name"]: copy.deepcopy(s) for s in schemas}
        self.execute_registered = execute
        self.metadata = copy.deepcopy(
            {
                k: {p: v for p, v in m.items() if p != "function"}
                for k, m in (metadata or {}).items()
            }
        )
        self.max_active = max(6, int(max_active))
        self.max_schema_chars = max(2000, int(max_schema_chars))
        self.active: OrderedDict[str, dict] = OrderedDict()
        self.control = {
            s["function"]["name"]: copy.deepcopy(s) for s in DISCOVERY_SCHEMAS
        }
        self.catalog.update(self.control)
        self.uncertain_mutations: set[str] = set()

    @property
    def schemas(self) -> list[dict]:
        return [*self.control.values(), *self.active.values()]

    def inventory(self) -> str:
        return "Available tool names (load exact schemas before calling): " + ", ".join(
            sorted(self.catalog)
        )

    def validate(self, name: str, args: Any) -> dict:
        schema = self.control.get(name) or self.active.get(name)
        if schema is None:
            raise ValueError(
                f"Tool {name!r} is not loaded. Use tool_search or load_tools to inspect its schema."
            )
        return validate_arguments(schema, args)

    def _activate(self, names: list[str]) -> dict:
        missing = [n for n in names if n not in self.catalog]
        if missing:
            raise ValueError("Unknown tools: " + ", ".join(missing))
        candidate = self.active.copy()
        for name in names:
            if name in self.control:
                continue
            candidate[name] = self.catalog[name]
            candidate.move_to_end(name)
        evicted = []
        while (
            len(candidate) > self.max_active
            or len(json.dumps(list(candidate.values()))) > self.max_schema_chars
        ):
            oldest = next(iter(candidate))
            if oldest in names:
                raise ValueError(
                    "Requested schemas exceed the context allowance; load fewer tools at once"
                )
            evicted.append(candidate.popitem(last=False)[0])
        self.active = candidate
        return {
            "ok": True,
            "schemas": [self.catalog[n] for n in names],
            "evicted": evicted,
        }

    def _search(self, query: str = "", limit: int = 6, offset: int = 0) -> dict:
        # Lexical retrieval over metadata is invoked by the model, never the user
        # prompt. There are no domain triggers, tool bundles, or mandatory tools.
        tokens = set(re.findall(r"\w+", query.lower().replace("_", " ")))
        rows = []
        for name, schema in self.catalog.items():
            if name in self.control:
                continue
            fn = schema["function"]
            words = set(
                re.findall(
                    r"\w+",
                    (name.replace("_", " ") + " " + fn.get("description", "")).lower(),
                )
            )
            score = len(tokens & words) + (100 if query.strip() == name else 0)
            if not tokens or score:
                rows.append((-score, name))
        rows.sort()
        names = [name for _, name in rows[offset : offset + limit]]
        result = self._activate(names)
        result.update(
            total=len(rows),
            next_offset=offset + limit if offset + limit < len(rows) else None,
        )
        return result

    def invoke(self, name: str, args: dict) -> Any:
        validated = self.validate(name, args)
        dispatch = {
            "tool_search": self._search,
            "load_tools": lambda names: self._activate(names),
        }
        if name in dispatch:
            return dispatch[name](**validated)
        signature = json.dumps([name, validated], sort_keys=True, allow_nan=False)
        if signature in self.uncertain_mutations:
            return {
                "ok": False,
                "error": "Previous execution has an unknown side-effect outcome. "
                "Inspect the target state before requesting this operation again.",
                "outcome_unknown": True,
            }
        try:
            return self.execute_registered(name, validated)
        except Exception:
            if not self.metadata.get(name, {}).get("readonly", False):
                self.uncertain_mutations.add(signature)
            raise
