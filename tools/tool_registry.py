"""Typed tool registration and lightweight argument validation."""
from __future__ import annotations

import inspect
import types
from typing import Any, Callable, Union, get_args, get_origin, get_type_hints


def agent_tool(*, name: str | None = None, description: str = "", readonly: bool = True, timeout: int | None = None):
    """Decorator for optional custom tools loaded from the workspace."""
    def decorate(func: Callable) -> Callable:
        func._agent_tool = True
        func._agent_tool_name = name or func.__name__
        func._agent_tool_description = description or (inspect.getdoc(func) or "").splitlines()[0] or func.__name__
        func._agent_tool_readonly = readonly
        func._agent_tool_timeout = timeout
        return func
    return decorate


def _json_type(annotation: Any) -> tuple[str, dict | None]:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (Union, types.UnionType):
        non_none = [arg for arg in args if arg is not type(None)]
        return _json_type(non_none[0] if non_none else str)
    if origin is list:
        item_type, _ = _json_type(args[0] if args else str)
        return "array", {"type": item_type}
    if origin is dict:
        return "object", None
    if origin is tuple:
        return "array", None
    if annotation in (int, float):
        return ("integer" if annotation is int else "number"), None
    if annotation is bool:
        return "boolean", None
    return "string", None


def function_schema(func: Callable, description: str | None = None) -> dict:
    """Convert a Python callable signature to an Ollama-compatible tool schema."""
    hints = get_type_hints(func)
    properties = {}
    required = []
    signature = inspect.signature(func)
    for param in signature.parameters.values():
        if param.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            continue
        annotation = hints.get(param.name, str)
        json_type, items = _json_type(annotation)
        entry = {"type": json_type}
        if items:
            entry["items"] = items
        properties[param.name] = entry
        if param.default is inspect.Parameter.empty:
            required.append(param.name)
    doc = description or getattr(func, "_agent_tool_description", None) or (inspect.getdoc(func) or "").splitlines()[0] or func.__name__
    return {
        "type": "function",
        "function": {
            "name": getattr(func, "_agent_tool_name", func.__name__),
            "description": doc,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def normalize_arguments(func: Callable, args: Any) -> dict:
    """Validate required argument names and return a clean keyword dictionary."""
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise TypeError("Tool arguments must be a JSON object.")
    signature = inspect.signature(func)
    accepted = {
        name for name, param in signature.parameters.items()
        if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    unknown = set(args) - accepted
    if unknown:
        raise TypeError(f"Unknown argument(s): {', '.join(sorted(unknown))}")
    missing = [
        name for name, param in signature.parameters.items()
        if param.default is inspect.Parameter.empty
        and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and name not in args
    ]
    if missing:
        raise TypeError(f"Missing required argument(s): {', '.join(missing)}")
    return dict(args)
