"""Typed tool registration and lightweight argument validation."""
from __future__ import annotations

import inspect
import json
import re
import types
from typing import Annotated, Any, Callable, Literal, Union, get_args, get_origin, get_type_hints


_REQUIRED_OVERRIDES = {
    # Several legacy helpers use empty-string defaults so callers receive a
    # friendly error instead of a Python exception. Native schemas should still
    # tell a small model these arguments are semantically required.
    "remember": {"fact"},
    "remember_semantic": {"fact"},
    "read_observation": {"observation_id"},
    "enqueue_research": {"topic"},
    "get_research_status": {"job_id"},
    "cancel_background_job": {"job_id"},
    "enqueue_self_optimization": {"objective"},
    "get_self_optimization_status": {"candidate_id"},
    "schedule_reminder": {"title"},
    "cancel_reminder": {"reminder_id"},
    "web_search": {"query"},
    "news_search": {"query"},
    "wiki_search": {"query"},
    "market_quote": {"instruments"},
    "browse_url": {"url"},
    "geocode_location": {"query"},
    "weather_forecast": {"latitude", "longitude"},
    "read_file": {"filename"},
    # Requiring content prevents a malformed small-model call from silently
    # truncating an existing file to zero bytes. An explicit empty string still
    # permits intentional empty-file creation.
    "write_file": {"filename", "content"},
    "execute_shell": {"command"},
    "execute_python": {"code"},
}

_SCHEMA_OVERRIDES: dict[tuple[str, str], dict[str, Any]] = {
    ("list_background_jobs", "status"): {"enum": ["", "pending", "running", "completed", "failed", "cancelled"]},
    ("news_search", "timelimit"): {"enum": ["", "d", "w", "m"]},
    ("schedule_reminder", "repeat"): {"enum": ["once", "daily", "weekly"]},
    ("list_self_optimization_candidates", "status"): {"enum": ["", "building", "benchmarking", "generating", "awaiting_approval", "approved", "rejected", "failed"]},
    ("list_work_queue", "status"): {"enum": ["pending", "running", "completed", "cancelled"]},
    ("update_work_status", "status"): {"enum": ["pending", "running", "completed", "cancelled"]},
    ("list_reminders", "status"): {"enum": ["", "scheduled", "error", "cancelled"]},
    ("process_snapshot", "sort_by"): {"enum": ["cpu", "memory", "io"]},
    ("connection_snapshot", "state"): {"enum": ["", "established", "listen", "time-wait", "close-wait", "syn-sent", "syn-recv"]},
    ("dns_diagnose", "record_types"): {"items": {"type": "string", "enum": ["A", "AAAA", "CNAME", "MX", "NS", "TXT", "SOA", "SRV", "PTR"]}},
    ("repo_checks", "checks"): {"items": {"type": "string", "enum": ["compile", "config", "ruff", "pytest"]}},
    ("gmail_search_messages", "query"): {"maxLength": 1000},
    ("gmail_search_messages", "limit"): {"maximum": 20},
    ("gmail_search_messages", "account"): {"maxLength": 160},
    ("gmail_read_message", "message_id"): {"maxLength": 1024},
    ("gmail_read_message", "account"): {"maxLength": 160},
    ("google_calendar_list_events", "time_min"): {"maxLength": 160},
    ("google_calendar_list_events", "time_max"): {"maxLength": 160},
    ("google_calendar_list_events", "query"): {"maxLength": 500},
    ("google_calendar_list_events", "calendar_id"): {"maxLength": 1024},
    ("google_calendar_list_events", "limit"): {"maximum": 50},
    ("google_calendar_list_events", "account"): {"maxLength": 160},
    ("google_calendar_get_event", "event_id"): {"maxLength": 1024},
    ("google_calendar_get_event", "calendar_id"): {"maxLength": 1024},
    ("google_calendar_get_event", "account"): {"maxLength": 160},
    ("google_calendar_list_calendars", "limit"): {"maximum": 50},
    ("google_calendar_list_calendars", "account"): {"maxLength": 160},
}

_SCHEMA_LIMITS_BY_NAME: dict[str, dict[str, Any]] = {
    "command": {"maxLength": 100000},
    "code": {"maxLength": 100000},
    "query": {"maxLength": 4000},
    "url": {"maxLength": 8192},
    "filename": {"maxLength": 4096},
    "filepath": {"maxLength": 4096},
    "path": {"maxLength": 4096},
    "content": {"maxLength": 1000000},
    "timeout": {"minimum": 1, "maximum": 300},
    "port": {"minimum": 1, "maximum": 65535},
    "limit": {"minimum": 1, "maximum": 10000},
    "lines": {"minimum": 1, "maximum": 5000},
    "max_lines": {"minimum": 1, "maximum": 5000},
    "max_files": {"minimum": 1, "maximum": 5000},
    "max_body_chars": {"minimum": 500, "maximum": 20000},
    "max_hops": {"minimum": 1, "maximum": 64},
    "probes": {"minimum": 1, "maximum": 20},
}

_PARAMETER_HINTS = {
    "query": "Search query text.",
    "instruments": "List of market instruments or explicit ticker/futures symbols to quote.",
    "topic": "Short topic/category label, or the research topic where applicable.",
    "fact": "Fact/text to store; supply the actual content rather than a placeholder.",
    "lines": "Maximum number of lines to return.",
    "title": "Short human-readable title.",
    "message": "Optional human-readable message/body text.",
    "when": "Reminder time in ISO-8601 form (for example 2026-09-18T09:00:00-04:00); alternatively use delay_seconds.",
    "repeat": "Reminder recurrence mode.",
    "reminder_id": "Reminder identifier returned by schedule_reminder; omit only when creating a new reminder.",
    "delay_seconds": "Relative delay in seconds; when >0 it is used instead of when.",
    "priority": "Tool-specific priority integer; larger usually means higher priority unless the tool says otherwise.",
    "status": "Status filter/value accepted by this specific tool.",
    "objective": "Bounded optimization objective describing the desired change.",
    "target_metric": "Optional measurable success criterion for optimization.",
    "candidate_id": "Self-optimization candidate identifier.",
    "work_id": "Legacy work-queue item identifier.",
    "description": "Optional detailed description.",
    "due_in_hours": "Optional number of hours from now until due.",
    "tags": "Optional list of short tags.",
    "estimated_hours": "Estimated work duration in hours.",
    "event_type": "Optional monitor-event type filter.",
    "filepath": "Path expected by this tool.",
    "log_path": "Host log path/name expected by this tool.",
    "service": "Optional systemd service/unit filter.",
    "grep": "Optional text/regular-expression filter supported by the tool.",
    "targets": "Optional list of network reachability targets.",
    "target": "Hostname or IP address to probe.",
    "symbol": "Optional Python symbol name; omit to read a line slice.",
    "start_line": "1-based starting line for a bounded source read.",
    "max_lines": "Maximum number of source lines to return.",
    "max_files": "Maximum number of repository files to include.",
    "markdown_content": "Markdown source text to render.",
    "url": "HTTP(S) URL.",
    "latitude": "Latitude in decimal degrees.",
    "longitude": "Longitude in decimal degrees.",
    "forecast_days": "Number of forecast days to return (1-16).",
    "timezone_name": "IANA timezone name or auto for the forecast coordinates.",
    "path": "Path or URL expected by this tool.",
    "filename": "File path/name expected by this tool.",
    "output_filename": "Output filename; use a simple workspace-relative name unless the tool says otherwise.",
    "content": "Text content to write or process.",
    "code": "Python source code to execute.",
    "command": "Explicit shell command to execute.",
    "timeout": "Timeout in seconds.",
    "limit": "Maximum number of results to return.",
    "offset": "Starting offset for a bounded read.",
    "length": "Maximum number of characters/items to return.",
    "network": "CIDR network, for example 192.168.1.0/24.",
    "job_id": "Durable job identifier.",
    "task_id": "Durable task identifier.",
    "observation_id": "Observation handle previously returned by the harness.",
    "package_name": "One or more package names only; do not include shell flags.",
    "tool_name": "Custom tool name.",
    "specification": "Natural-language specification for the custom tool.",
    "context": "Short context explaining what should be inspected or why.",
    "location": "Canonical city/region/country scope used to disambiguate location-sensitive results.",
    "sort_by": "Sort mode accepted by this tool.",
    "include_command": "Whether to include a bounded, redacted process command line.",
    "state": "Optional connection-state filter.",
    "name": "DNS name or hostname expected by this tool.",
    "record_types": "List of DNS record types such as A, AAAA, CNAME, MX, NS, TXT, SOA, SRV, or PTR.",
    "resolver": "Optional DNS resolver IP/hostname; empty uses the system resolver.",
    "max_hops": "Maximum network-path hops to probe.",
    "probes": "Small number of probes/cycles per network-path hop.",
    "host": "Hostname or IP address.",
    "port": "TCP port number from 1 to 65535.",
    "tls": "Whether to perform a TLS handshake after TCP connect.",
    "allow_private": "Whether this explicit probe may target private/local addresses.",
    "same_domain": "Limit extracted links to the page's hostname.",
    "path_or_url": "Workspace-local file path or public HTTP(S) URL.",
    "max_pages": "Maximum number of PDF pages to extract.",
    "max_chars": "Maximum extracted text characters.",
    "max_diff_chars": "Maximum characters of unified diff to return.",
    "checks": "Known repository checks only: compile, config, ruff, pytest.",
    "account": "Configured local account selector; use default unless the user chose another account.",
    "message_id": "Gmail message identifier returned by gmail_search_messages.",
    "calendar_id": "Google Calendar identifier; primary selects the account's primary calendar.",
    "event_id": "Google Calendar event identifier returned by google_calendar_list_events.",
    "time_min": "Inclusive RFC3339 lower time bound with timezone; empty defaults to now.",
    "time_max": "Exclusive RFC3339 upper time bound with timezone; empty defaults to 14 days after time_min.",
    "include_spam_trash": "Whether Gmail search may include Spam and Trash; defaults to false.",
    "max_body_chars": "Maximum Gmail message body characters to return (500-20000).",
    "old_id": "Older durable observation identifier.",
    "new_id": "Newer durable observation identifier.",
}


def agent_tool(*, name: str | None = None, description: str = "", readonly: bool = False, timeout: int | None = 60):
    """Decorator for optional custom tools loaded from the workspace.

    Custom tools default to mutating (readonly=False). A tool author must opt
    into readonly=True only when calling it cannot change external state. This
    conservative default prevents a generated tool from gaining repeat/read-only
    treatment merely because a small model omitted metadata.
    """
    def decorate(func: Callable) -> Callable:
        func._agent_tool = True
        func._agent_tool_name = name or func.__name__
        raw_doc = inspect.getdoc(func) or ""
        first_doc_line = raw_doc.splitlines()[0] if raw_doc.splitlines() else ""
        func._agent_tool_description = description or first_doc_line or func.__name__
        func._agent_tool_readonly = readonly
        func._agent_tool_timeout = timeout
        return func
    return decorate


def _unwrap_annotation(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin is Annotated:
        args = get_args(annotation)
        return args[0] if args else str
    return annotation


def _json_type(annotation: Any) -> tuple[str, dict | None, list[Any] | None]:
    annotation = _unwrap_annotation(annotation)
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Literal:
        values = list(args)
        sample = next((value for value in values if value is not None), "")
        if isinstance(sample, bool):
            json_type = "boolean"
        elif isinstance(sample, int):
            json_type = "integer"
        elif isinstance(sample, float):
            json_type = "number"
        else:
            json_type = "string"
        return json_type, None, values
    if origin in (Union, types.UnionType):
        non_none = [arg for arg in args if arg is not type(None)]
        return _json_type(non_none[0] if non_none else str)
    if origin is list or annotation is list:
        item_type, _, item_enum = _json_type(args[0] if args else str)
        item_schema: dict[str, Any] = {"type": item_type}
        if item_enum is not None:
            item_schema["enum"] = item_enum
        return "array", item_schema, None
    if origin is dict or annotation is dict:
        return "object", None, None
    if origin is tuple or annotation is tuple:
        return "array", None, None
    if annotation in (int, float):
        return ("integer" if annotation is int else "number"), None, None
    if annotation is bool:
        return "boolean", None, None
    return "string", None, None


def function_schema(func: Callable, description: str | None = None) -> dict:
    """Convert a Python callable signature to an Ollama-compatible tool schema."""
    stored_schema = getattr(func, "_agent_tool_schema", None)
    if isinstance(stored_schema, dict):
        # Lazy builtin proxies carry the exact generated native schema so startup
        # need not import heavyweight implementation modules just for metadata.
        return json.loads(json.dumps(stored_schema))
    hints = get_type_hints(func, include_extras=True)
    properties = {}
    required = []
    signature = inspect.signature(func)
    public_name = getattr(func, "_agent_tool_name", func.__name__)
    required_overrides = _REQUIRED_OVERRIDES.get(public_name, set())
    for param in signature.parameters.values():
        if param.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            continue
        annotation = hints.get(param.name, str)
        json_type, items, enum = _json_type(annotation)
        entry: dict[str, Any] = {"type": json_type}
        if items:
            entry["items"] = items
        if enum is not None:
            entry["enum"] = enum
        if param.name in _PARAMETER_HINTS:
            entry["description"] = _PARAMETER_HINTS[param.name]
        if param.default is inspect.Parameter.empty or param.name in required_overrides:
            required.append(param.name)
        elif param.default is not None and isinstance(param.default, (str, int, float, bool, list, dict)):
            entry["default"] = param.default
        entry.update(_SCHEMA_LIMITS_BY_NAME.get(param.name, {}))
        entry.update(_SCHEMA_OVERRIDES.get((public_name, param.name), {}))
        properties[param.name] = entry
    raw_doc = inspect.getdoc(func) or ""
    first_doc_line = raw_doc.splitlines()[0] if raw_doc.splitlines() else ""
    doc = description or getattr(func, "_agent_tool_description", None) or first_doc_line or func.__name__
    return {
        "type": "function",
        "function": {
            "name": public_name,
            "description": doc,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def _coerce_value(value: Any, annotation: Any, name: str) -> Any:
    annotation = _unwrap_annotation(annotation)
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin in (Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        errors = []
        for candidate in (arg for arg in args if arg is not type(None)):
            try:
                return _coerce_value(value, candidate, name)
            except (TypeError, ValueError) as exc:
                errors.append(str(exc))
        raise TypeError(errors[0] if errors else f"Argument '{name}' has an invalid type.")

    if origin is Literal:
        allowed = list(args)
        if value in allowed:
            return value
        # Permit exact textual representation for string literals only.
        if all(isinstance(item, str) for item in allowed) and isinstance(value, str):
            for item in allowed:
                if value == item:
                    return item
        raise TypeError(f"Argument '{name}' must be one of: {', '.join(map(str, allowed))}.")

    if annotation is Any or annotation is inspect.Parameter.empty:
        return value
    if annotation is str:
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)):
            return str(value)
        raise TypeError(f"Argument '{name}' must be a string.")
    if annotation is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
            return value.strip().lower() == "true"
        raise TypeError(f"Argument '{name}' must be a boolean.")
    if annotation is int:
        if isinstance(value, bool):
            raise TypeError(f"Argument '{name}' must be an integer.")
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
            return int(value.strip())
        raise TypeError(f"Argument '{name}' must be an integer.")
    if annotation is float:
        if isinstance(value, bool):
            raise TypeError(f"Argument '{name}' must be a number.")
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                pass
        raise TypeError(f"Argument '{name}' must be a number.")
    if origin is list or annotation is list:
        if isinstance(value, str) and value.lstrip().startswith("["):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise TypeError(f"Argument '{name}' must be a JSON array.") from exc
        if not isinstance(value, list):
            raise TypeError(f"Argument '{name}' must be an array.")
        item_annotation = args[0] if args else Any
        return [_coerce_value(item, item_annotation, f"{name}[]") for item in value]
    if origin is tuple or annotation is tuple:
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"Argument '{name}' must be an array.")
        return list(value)
    if origin is dict or annotation is dict:
        if isinstance(value, str) and value.lstrip().startswith("{"):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise TypeError(f"Argument '{name}' must be a JSON object.") from exc
        if not isinstance(value, dict):
            raise TypeError(f"Argument '{name}' must be an object.")
        return value
    return value



def _coerce_schema_value(value: Any, spec: dict[str, Any], name: str) -> Any:
    """Coerce and recursively validate the JSON-schema subset sent to Ollama."""
    kind = str(spec.get("type") or "")
    if kind == "string":
        if isinstance(value, str):
            result = value
        elif isinstance(value, (int, float, bool)):
            result = str(value)
        elif isinstance(value, (dict, list)):
            result = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        else:
            raise TypeError(f"Argument '{name}' must be a string.")
        minimum = spec.get("minLength")
        maximum = spec.get("maxLength")
        if minimum is not None and len(result) < int(minimum):
            raise TypeError(f"Argument '{name}' must contain at least {int(minimum)} characters.")
        if maximum is not None and len(result) > int(maximum):
            raise TypeError(f"Argument '{name}' exceeds the {int(maximum)} character limit.")
        coerced = result
    elif kind == "boolean":
        if isinstance(value, bool): coerced = value
        elif isinstance(value, str) and value.strip().lower() in {"true", "false"}: coerced = value.strip().lower() == "true"
        else: raise TypeError(f"Argument '{name}' must be a boolean.")
    elif kind == "integer":
        if isinstance(value, bool): raise TypeError(f"Argument '{name}' must be an integer.")
        if isinstance(value, int): coerced = value
        elif isinstance(value, float) and value.is_integer(): coerced = int(value)
        elif isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()): coerced = int(value.strip())
        else: raise TypeError(f"Argument '{name}' must be an integer.")
    elif kind == "number":
        if isinstance(value, bool): raise TypeError(f"Argument '{name}' must be a number.")
        if isinstance(value, (int, float)): coerced = float(value)
        elif isinstance(value, str):
            try: coerced = float(value.strip())
            except ValueError as exc: raise TypeError(f"Argument '{name}' must be a number.") from exc
        else: raise TypeError(f"Argument '{name}' must be a number.")
    elif kind == "array":
        if isinstance(value, str) and value.lstrip().startswith("["):
            try: value = json.loads(value)
            except json.JSONDecodeError as exc: raise TypeError(f"Argument '{name}' must be a JSON array.") from exc
        if not isinstance(value, list): raise TypeError(f"Argument '{name}' must be an array.")
        if spec.get("minItems") is not None and len(value) < int(spec["minItems"]):
            raise TypeError(f"Argument '{name}' requires at least {int(spec['minItems'])} item(s).")
        if spec.get("maxItems") is not None and len(value) > int(spec["maxItems"]):
            raise TypeError(f"Argument '{name}' exceeds the {int(spec['maxItems'])} item limit.")
        item_spec = spec.get("items") if isinstance(spec.get("items"), dict) else {}
        coerced = [_coerce_schema_value(item, item_spec, f"{name}[]") for item in value] if item_spec else value
    elif kind == "object":
        if isinstance(value, str) and value.lstrip().startswith("{"):
            try: value = json.loads(value)
            except json.JSONDecodeError as exc: raise TypeError(f"Argument '{name}' must be a JSON object.") from exc
        if not isinstance(value, dict): raise TypeError(f"Argument '{name}' must be an object.")
        props = spec.get("properties") if isinstance(spec.get("properties"), dict) else {}
        if props:
            unknown = set(value) - set(props)
            if unknown and spec.get("additionalProperties") is False:
                raise TypeError(f"Argument '{name}' has unknown field(s): {', '.join(sorted(unknown))}.")
            missing = [str(key) for key in spec.get("required", []) if key not in value]
            if missing:
                raise TypeError(f"Argument '{name}' is missing required field(s): {', '.join(missing)}.")
            coerced = {key: _coerce_schema_value(item, props.get(key, {}), f"{name}.{key}") for key, item in value.items()}
        else:
            coerced = value
    else:
        coerced = value

    allowed = spec.get("enum")
    if isinstance(allowed, list) and coerced not in allowed:
        raise TypeError(f"Argument '{name}' must be one of: {', '.join(map(str, allowed))}.")
    if isinstance(coerced, (int, float)) and not isinstance(coerced, bool):
        if spec.get("minimum") is not None and coerced < float(spec["minimum"]):
            raise TypeError(f"Argument '{name}' must be >= {spec['minimum']}.")
        if spec.get("maximum") is not None and coerced > float(spec["maximum"]):
            raise TypeError(f"Argument '{name}' must be <= {spec['maximum']}.")
    return coerced

def _normalize_schema_arguments(schema: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    fn = schema.get("function", {}) if isinstance(schema, dict) else {}
    params = fn.get("parameters", {}) if isinstance(fn.get("parameters"), dict) else {}
    properties = params.get("properties", {}) if isinstance(params.get("properties"), dict) else {}
    accepted = set(properties)
    unknown = set(args) - accepted
    if unknown:
        raise TypeError(f"Unknown argument(s): {', '.join(sorted(unknown))}")
    required = [str(name) for name in params.get("required", []) if name]
    missing = [name for name in required if name not in args]
    if missing:
        raise TypeError(f"Missing required argument(s): {', '.join(missing)}")
    return {name: _coerce_schema_value(value, properties.get(name, {}), name) for name, value in args.items()}

def normalize_arguments(func: Callable, args: Any) -> dict:
    """Validate names/types and apply only unambiguous primitive coercions."""
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise TypeError("Tool arguments must be a JSON object.")
    stored_schema = getattr(func, "_agent_tool_schema", None)
    if isinstance(stored_schema, dict):
        return _normalize_schema_arguments(stored_schema, args)
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

    hints = get_type_hints(func, include_extras=True)
    normalized = {}
    for name, value in args.items():
        param = signature.parameters[name]
        if value is None and param.default is None:
            normalized[name] = None
            continue
        normalized[name] = _coerce_value(value, hints.get(name, str), name)
    return normalized
