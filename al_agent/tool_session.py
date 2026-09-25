"""Per-turn tool discovery, compact schema activation, and strict validation."""

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
        "Search the available tool catalog by name or description. Returns compact ranked "
        "candidate metadata and automatically activates the relevant candidate schemas for the next "
        "resident-model call. Full schemas are supplied only through the native tools field. "
        "Empty query browses names.",
        {
            "query": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 8},
            "offset": {"type": "integer", "minimum": 0},
        },
        [],
    ),
    _schema(
        "load_tools",
        "Activate exact tool names from the available catalog. Full schemas are then "
        "supplied only through the next native tools field; call them separately to perform work.",
        {
            "names": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 8,
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
        decision_engine: Any | None = None,
        router: Any | None = None,
        initial_active: list[str] | None = None,
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
        # Compatibility keywords are retained for integrations, but production
        # now supplies a non-generative deterministic routing engine.
        self.router = router or decision_engine
        self.routed_tools: set[str] = set()
        if initial_active:
            activation = self._activate(list(initial_active), allow_trim=True)
            self.routed_tools.update(activation.get("activated", []))

    @property
    def schemas(self) -> list[dict]:
        return [*self.control.values(), *self.active.values()]

    def inventory(self) -> str:
        return (
            "Only currently supplied native tool schemas may be called. "
            "Additional capabilities are available through tool_search/load_tools; "
            "schema absence does not mean a capability is unavailable."
        )

    def validate(self, name: str, args: Any) -> dict:
        schema = self.control.get(name) or self.active.get(name)
        if schema is None:
            raise ValueError(
                f"Tool {name!r} is not loaded. Use tool_search or load_tools to inspect its schema."
            )
        return validate_arguments(schema, args)

    def _activate(
        self, names: list[str], *, replace: bool = False, allow_trim: bool = False
    ) -> dict:
        missing = [n for n in names if n not in self.catalog]
        if missing:
            raise ValueError("Unknown tools: " + ", ".join(missing))
        candidate = OrderedDict() if replace else self.active.copy()
        evicted = [name for name in self.active if replace and name not in names]
        for name in names:
            if name in self.control:
                continue
            candidate[name] = self.catalog[name]
            candidate.move_to_end(name)
        while (
            len(candidate) > self.max_active
            or len(json.dumps(list(candidate.values()))) > self.max_schema_chars
        ):
            if allow_trim and candidate:
                # Automatic candidates are ordered by relevance; trim the least
                # relevant tail until the schema budget fits. Explicit load_tools
                # remains strict and still reports oversize requests.
                dropped = candidate.popitem(last=True)[0]
                evicted.append(dropped)
                continue
            oldest = next(iter(candidate))
            if oldest in names:
                raise ValueError(
                    "Requested schemas exceed the context allowance; load fewer tools at once"
                )
            evicted.append(candidate.popitem(last=False)[0])
        self.active = candidate
        return {
            "ok": True,
            "activated": [n for n in names if n in self.active and n not in self.control],
            "evicted": evicted,
        }

    def _search(self, query: str = "", limit: int = 8, offset: int = 0) -> dict:
        """Return compact discovery metadata and activate a bounded relevant set.

        Search itself never invokes a model. Complete schemas remain out of the
        observation payload and are supplied only through the next request's
        native ``tools`` field.
        """
        available = [
            schema for name, schema in self.catalog.items() if name not in self.control
        ]
        candidates: list[dict] = []
        selected_names: list[str] = []
        confidence = 0.0
        selection_mode = "browse"
        if self.router is not None and query.strip():
            decision = self.router.decide(query, available, self.metadata)
            ranked = list(decision.candidates)
            window = ranked[offset : offset + limit]
            candidates = [
                {
                    "name": row.name,
                    "description": row.description[:180],
                    "relevance": round(row.score, 3),
                }
                for row in window
            ]
            total = len(ranked)
            selected = set(decision.selected)
            selected_names = [row.name for row in window if row.name in selected]
            confidence = float(decision.confidence)
            selection_mode = decision.tier
        else:
            names = sorted(name for name in self.catalog if name not in self.control)
            window_names = names[offset : offset + limit]
            candidates = [
                {
                    "name": name,
                    "description": str(
                        self.catalog[name].get("function", {}).get("description", "")
                    )[:180],
                }
                for name in window_names
            ]
            total = len(names)
            # Standalone/unit-test sessions without a routing engine preserve the
            # historical deterministic browse behavior. Production always supplies
            # DeterministicToolRouter, so this path performs no intent routing.
            if self.router is None and candidates:
                selected_names = [candidates[0]["name"]]
                selection_mode = "browse_first"

        # Replace rather than accumulate schemas: each discovery step presents a
        # fresh bounded candidate set and prevents prompt growth across a turn.
        activation = self._activate(selected_names, replace=True, allow_trim=True)
        activated_names = list(activation.get("activated", []))
        self.routed_tools.update(activated_names)
        return {
            "ok": True,
            "candidates": candidates,
            "selected": activated_names[0] if len(activated_names) == 1 else None,
            "selection_mode": selection_mode,
            "confidence": round(confidence, 3) if selected_names else None,
            "activated": activation.get("activated", []),
            "evicted": activation.get("evicted", []),
            "total": total,
            "next_offset": offset + limit if offset + limit < total else None,
        }

    def invoke(self, name: str, args: dict) -> Any:
        validated = self.validate(name, args)
        dispatch = {
            "tool_search": self._search,
            "load_tools": lambda names: self._activate(names, replace=True),
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
