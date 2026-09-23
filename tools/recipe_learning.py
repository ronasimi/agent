"""Successful-turn recipe candidate extraction and conservative workflow generalization.

The generalizer is deliberately deterministic.  A small/fast model may propose
semantic names for candidate values, but every proposal is re-derived and
validated against the successful tool trace before it can affect a stored
recipe.
"""
from __future__ import annotations

import ipaddress
import json
import re
from collections import Counter
from typing import Any
from urllib.parse import urlsplit

from .recipe_store import (
    create_candidate, dismiss_pending_candidate, pending_candidate,
    recipe_exists_for_task, save_pending_candidate,
)

# User/task-defining argument names. Operational controls stay constants unless
# the user explicitly supplied them as part of the objective.
_PARAMETER_KEYS = {
    "target", "host", "hostname", "url", "path", "filename", "query", "name",
    "network", "service", "record_type", "record_types", "resolver", "location",
    "instrument", "symbol", "topic", "calendar_id", "file_id", "document_id",
}
_OPERATIONAL_KEYS = {
    "timeout", "limit", "max_results", "allow_private", "recursive", "offset",
    "length", "forecast_days", "include_images", "page", "port", "party_size",
    "record_type", "record_types",
}
_SECRET_KEY_RE = re.compile(r"(?:pass(?:word)?|secret|token|credential|api[_-]?key|authorization|cookie)", re.I)
_SAFE_PARAM_RE = re.compile(r"[^a-z0-9_]+")
_AFFIRM = {"yes","yes save it","save it","save recipe","save this recipe","sure","do it","yes please"}
_DECLINE = {"no","no thanks","don't save it","do not save it","skip","not now"}


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) and value is not None


def _secret_like(key: str, value: Any) -> bool:
    if _SECRET_KEY_RE.search(str(key or "")):
        return True
    if isinstance(value, str):
        text = value.strip()
        if re.search(r"(?i)^bearer\s+[a-z0-9._~-]+$", text):
            return True
        # Long opaque strings are poor reusable parameters and may be secrets.
        if len(text) >= 32 and re.fullmatch(r"[A-Za-z0-9_./+=~-]+", text) and " " not in text:
            return True
    return False


def _hostname(value: Any) -> str:
    text = str(value or "").strip()
    if not text or " " in text or "/" in text:
        return ""
    candidate = text.strip("[]").rstrip(".")
    try:
        ipaddress.ip_address(candidate)
        return candidate
    except ValueError:
        pass
    if re.fullmatch(r"(?=.{1,253}$)(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[A-Za-z]{2,63}", candidate):
        return candidate.lower()
    return ""


def _url_host(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    try:
        parsed = urlsplit(value)
    except ValueError:
        return ""
    return str(parsed.hostname or "").lower()


def _walk_scalars(value: Any, *, path: tuple[str, ...] = ()):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _walk_scalars(child, path=path + (str(key),))
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            yield from _walk_scalars(child, path=path + (str(idx),))
    elif _is_scalar(value):
        yield path, value


def _param_name(key: str, value: Any, hints: dict[str, str]) -> str:
    value_key = json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)
    hinted = str(hints.get(value_key) or "").strip().lower()
    base = hinted or str(key or "value").lower()
    if _hostname(value):
        base = hinted or "hostname"
    base = _SAFE_PARAM_RE.sub("_", base).strip("_") or "value"
    if base in _OPERATIONAL_KEYS:
        base = f"input_{base}"
    return base[:48]


def _collect_fast_hints(raw_hints: Any) -> dict[str, str]:
    """Accept only {name,value} hints.  Values are validated later against trace."""
    out: dict[str, str] = {}
    if not isinstance(raw_hints, list):
        return out
    for item in raw_hints[:16]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip().lower()
        value = item.get("value")
        if not name or not _is_scalar(value):
            continue
        name = _SAFE_PARAM_RE.sub("_", name).strip("_")[:48]
        if not name or _secret_like(name, value):
            continue
        out[json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)] = name
    return out


def _objective_mentions(objective_lower: str, value: Any) -> bool:
    text = str(value).strip().lower()
    if not text:
        return False
    if len(text) <= 2 or text.isdigit():
        return bool(re.search(r"(?<![a-z0-9])" + re.escape(text) + r"(?![a-z0-9])", objective_lower))
    return text in objective_lower


def _candidate_values(objective: str, calls: list[tuple[str, dict[str, Any]]], raw_hints: Any = None) -> list[tuple[str, Any]]:
    """Infer reusable task inputs across the complete successful workflow.

    A value is a candidate when it is task-defining, occurs repeatedly, is
    explicitly present in the objective, or is a hostname embedded in both host
    and URL arguments. Operational/safety constants are excluded unless the
    objective explicitly names their value.
    """
    objective_lower = str(objective or "").lower()
    occurrences: list[tuple[str, Any]] = []
    scalar_counts: Counter[str] = Counter()
    hostname_counts: Counter[str] = Counter()
    fast_hints = _collect_fast_hints(raw_hints)

    for _tool, args in calls:
        for path, value in _walk_scalars(args):
            key = path[-1] if path else "value"
            if _secret_like(key, value):
                continue
            occurrences.append((key, value))
            scalar_counts[json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)] += 1
            host = _hostname(value) or _url_host(value)
            if host:
                hostname_counts[host] += 1

    selected: list[tuple[str, Any]] = []
    seen_values: set[str] = set()
    for key, value in occurrences:
        encoded = json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)
        if encoded in seen_values:
            continue
        text = str(value).strip() if isinstance(value, str) else str(value)
        explicit = _objective_mentions(objective_lower, value)
        repeated = scalar_counts[encoded] >= 2
        host = _hostname(value)
        host_related = bool(host and hostname_counts[host] >= 2)
        hinted = encoded in fast_hints
        key_task_defining = key.lower() in _PARAMETER_KEYS
        operational = key.lower() in _OPERATIONAL_KEYS

        # Booleans and numeric operational constants should almost never become
        # inputs automatically.  An explicit objective mention can still opt in.
        if operational and not explicit:
            continue
        if isinstance(value, bool) and not explicit:
            continue
        if isinstance(value, (int, float)) and not explicit and not repeated:
            continue
        if not (explicit or repeated or host_related or hinted or key_task_defining):
            continue
        if isinstance(value, str) and not value.strip():
            continue
        selected.append((key, value))
        seen_values.add(encoded)

    # If URLs and host arguments refer to the same host, prefer the hostname as
    # the single source parameter; URL values are derived by templates below.
    host_candidates = {(_hostname(v), k, v) for k, v in selected if _hostname(v)}
    for host, _key, _value in sorted(host_candidates):
        if not host:
            continue
        for key, value in occurrences:
            if _url_host(value) == host:
                encoded = json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)
                selected = [(k, v) for k, v in selected if json.dumps(v, sort_keys=True, default=str, ensure_ascii=False) != encoded]
    return selected


def _parameter_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    return "string"


def _allocate_parameters(candidates: list[tuple[str, Any]], raw_hints: Any = None) -> tuple[dict[str, Any], dict[str, str]]:
    hints = _collect_fast_hints(raw_hints)
    params: dict[str, Any] = {}
    value_to_name: dict[str, str] = {}
    for key, value in candidates:
        encoded = json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)
        if encoded in value_to_name:
            continue
        base = _param_name(key, value, hints)
        name = base
        suffix = 2
        while name in params and params[name].get("default") != value:
            name = f"{base}_{suffix}"
            suffix += 1
        params[name] = {
            "default": value,
            "type": _parameter_type(value),
            "description": f"Reusable workflow input for {key}",
            "inferred": True,
        }
        value_to_name[encoded] = name
    return params, value_to_name


def _replace_value(value: Any, key: str, value_to_name: dict[str, str], parameters: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return {k: _replace_value(v, str(k), value_to_name, parameters) for k, v in value.items()}
    if isinstance(value, list):
        return [_replace_value(v, key, value_to_name, parameters) for v in value]

    encoded = json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)
    pname = value_to_name.get(encoded)
    if pname:
        return {"$param": pname, "default": value}

    # Derive URL/string arguments from an already inferred hostname parameter.
    if isinstance(value, str):
        host = _url_host(value)
        if host:
            host_name = value_to_name.get(json.dumps(host, ensure_ascii=False))
            if host_name:
                # Replace only the hostname occurrence, not arbitrary substrings.
                parsed = urlsplit(value)
                netloc = parsed.netloc
                host_display = parsed.hostname or host
                template_netloc = netloc.replace(host_display, "{" + host_name + "}", 1)
                rendered = parsed._replace(netloc=template_netloc).geturl()
                return {
                    "$template": rendered,
                    "vars": {host_name: {"$param": host_name}},
                }
        # General string relationship: if a selected string is embedded as a
        # meaningful token, derive it from that parameter rather than inventing a
        # second default. Require >=3 chars to avoid unsafe tiny replacements.
        for param_name, meta in parameters.items():
            default = meta.get("default")
            if isinstance(default, str) and len(default) >= 3 and default in value and default != value:
                template = value.replace(default, "{" + param_name + "}")
                return {"$template": template, "vars": {param_name: {"$param": param_name}}}
    return value


def build_candidate(
    objective: str,
    trace: list[dict[str, Any]],
    *,
    semantic_hints: Any = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Convert successful read-only tool calls into a generalized reusable pipeline.

    ``semantic_hints`` may come from the fast model, but hints are advisory only:
    they cannot introduce values that did not occur in the successful trace and
    cannot override secret/operational-constant filtering.
    """
    calls: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for entry in trace:
        if not entry.get("success") or not entry.get("readonly", True):
            continue
        tool = str(entry.get("tool") or "")
        args = dict(entry.get("args") or {})
        signature = json.dumps([tool, args], sort_keys=True, default=str)
        if signature in seen:
            continue
        seen.add(signature)
        if tool in {"current_time", "run_pipeline", "run_recipe", "search_recipes", "list_recipes", "tool_search"}:
            continue
        calls.append((tool, args))
        if len(calls) >= 8:
            break

    candidates = _candidate_values(objective, calls, semantic_hints)
    parameters, value_to_name = _allocate_parameters(candidates, semantic_hints)
    stages: list[dict[str, Any]] = []
    for tool, args in calls:
        generalized = {k: _replace_value(v, str(k), value_to_name, parameters) for k, v in args.items()}
        stages.append({"id": f"s{len(stages)+1}", "tool": tool, "args": generalized})
    return stages, parameters


def maybe_create_recipe_candidate(
    objective: str,
    trace: list[dict[str, Any]],
    min_stages: int = 2,
    *,
    semantic_hints: Any = None,
) -> dict[str, Any] | None:
    stages, params = build_candidate(objective, trace, semantic_hints=semantic_hints)
    if len(stages) < max(1, int(min_stages)):
        return None
    stage_tools = [str(stage.get("tool") or "") for stage in stages]
    if "geocode_location" in stage_tools and "weather_forecast" in stage_tools:
        return None
    if recipe_exists_for_task(objective, stages):
        return None
    tools = [s["tool"] for s in stages]
    cid = create_candidate(objective, stages, params, tags=tools[:8])
    return {
        "candidate_id": cid,
        "objective": objective,
        "stages": len(stages),
        "tools": tools,
        "parameters": params,
        "generalized": bool(params),
    }


def pending_recipe_prompt() -> str:
    candidate = pending_candidate()
    if not candidate:
        return ""
    tools = " → ".join(str(s.get("tool") or "") for s in candidate["pipeline"])
    count = len(candidate.get("parameters") or {})
    suffix = f" with {count} inferred reusable parameter(s)" if count else ""
    return f"This successful workflow does not match an existing saved recipe. Save it as a reusable recipe{suffix}? ({tools})"


def handle_recipe_confirmation(text: str) -> tuple[bool, str]:
    """Handle a direct yes/no response to a harness recipe-save prompt."""
    if not pending_candidate():
        return False, ""
    normalized = re.sub(r"\s+", " ", str(text).strip().lower())
    named = re.match(r"^(?:yes[, ]+)?save (?:this )?recipe as\s+(.+)$", normalized)
    if named:
        recipe = save_pending_candidate(name=named.group(1).strip()[:80])
        if not recipe:
            return True, "No pending recipe was available to save."
        return True, f"Saved recipe **{recipe['name']}** with {len(recipe['pipeline'])} stage(s)."
    if normalized in _AFFIRM or normalized.startswith("yes, save"):
        recipe = save_pending_candidate()
        if not recipe:
            return True, "No pending recipe was available to save."
        return True, f"Saved recipe **{recipe['name']}** with {len(recipe['pipeline'])} stage(s)."
    if normalized in _DECLINE:
        dismiss_pending_candidate()
        return True, "Recipe not saved."
    dismiss_pending_candidate()
    return False, ""
