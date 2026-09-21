"""Harness-native bounded pipelines and stored recipe execution."""
from __future__ import annotations

import json
from typing import Any

MAX_STAGES = 16
MAX_INVOCATIONS = 48
MAX_FOREACH_ITEMS = 20
MAX_INTERMEDIATE_CHARS = 200_000
_FORBIDDEN = {
    "execute_shell", "execute_python", "run_pipeline", "run_recipe", "save_recipe",
    "save_pending_recipe", "create_or_update_tool", "install_package",
}


def _parse_result(value: Any) -> Any:
    if isinstance(value, (dict, list, int, float, bool)) or value is None:
        return value
    text = str(value)
    try:
        return json.loads(text)
    except Exception:
        return text


def _extract(value: Any, path: str) -> Any:
    if path in {"", ".", "$"}:
        return value
    cur = value
    for part in str(path).strip("$.").split("."):
        if not part:
            continue
        if isinstance(cur, list):
            cur = cur[int(part)]
        elif isinstance(cur, dict):
            cur = cur[part]
        else:
            raise KeyError(path)
    return cur


def _resolve(value: Any, outputs: dict[str, Any], parameters: dict[str, Any], item: Any = None) -> Any:
    if isinstance(value, dict):
        if "$ref" in value:
            key = str(value["$ref"])
            if key not in outputs:
                if "default" in value:
                    return _resolve(value["default"], outputs, parameters, item)
                raise KeyError(f"missing pipeline output: {key}")
            try:
                return _extract(outputs[key], str(value.get("path") or "$"))
            except (KeyError, IndexError, TypeError, ValueError):
                if "default" in value:
                    return _resolve(value["default"], outputs, parameters, item)
                raise
        if "$param" in value:
            key = str(value["$param"])
            if key in parameters:
                return parameters[key]
            if "default" in value:
                return _resolve(value["default"], outputs, parameters, item)
            raise KeyError(f"missing recipe parameter: {key}")
        if "$item" in value:
            if item is None and "default" in value:
                return _resolve(value["default"], outputs, parameters, item)
            return _extract(item, str(value.get("path") or value.get("$item") or "$"))
        return {k: _resolve(v, outputs, parameters, item) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(v, outputs, parameters, item) for v in value]
    return value


def _condition(value: Any, outputs: dict[str, Any], parameters: dict[str, Any], item: Any = None) -> bool:
    if isinstance(value, dict) and "$not" in value:
        return not _condition(value["$not"], outputs, parameters, item)
    if isinstance(value, dict) and "$equals" in value:
        pair = value["$equals"]
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("$equals requires exactly two values")
        return _resolve(pair[0], outputs, parameters, item) == _resolve(pair[1], outputs, parameters, item)
    if isinstance(value, dict) and "$in" in value:
        pair = value["$in"]
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("$in requires [value, collection]")
        return _resolve(pair[0], outputs, parameters, item) in _resolve(pair[1], outputs, parameters, item)
    return bool(_resolve(value, outputs, parameters, item))


def _serialize_size(parsed: Any) -> tuple[str, int]:
    serialized = json.dumps(parsed, default=str, ensure_ascii=False) if not isinstance(parsed, str) else parsed
    return serialized, len(serialized)


def execute_pipeline(stages: list[dict[str, Any]], parameters: dict[str, Any] | None = None) -> dict[str, Any]:
    from . import AVAILABLE_TOOLS_MAP, TOOL_METADATA
    from .loop_validator import classify_tool_outcome
    from .grounding import grounding_metadata
    from .tool_registry import normalize_arguments
    from .executor import execute_registered_tool

    parameters = parameters or {}
    if not isinstance(stages, list) or not stages:
        return {"ok": False, "error": "pipeline requires at least one stage"}
    if len(stages) > MAX_STAGES:
        return {"ok": False, "error": f"pipeline exceeds {MAX_STAGES} stages"}

    outputs: dict[str, Any] = {}
    summaries: list[dict[str, Any]] = []
    invocations = 0

    for idx, stage in enumerate(stages, 1):
        if not isinstance(stage, dict):
            return {"ok": False, "error": f"stage {idx} must be an object"}
        tool = str(stage.get("tool") or "")
        sid = str(stage.get("id") or f"s{idx}")
        if tool in _FORBIDDEN or tool not in AVAILABLE_TOOLS_MAP:
            return {"ok": False, "error": f"stage {idx}: tool unavailable or forbidden: {tool}"}
        meta = TOOL_METADATA.get(tool, {})
        if not bool(meta.get("readonly", True)):
            return {"ok": False, "error": f"stage {idx}: pipelines are read-only; {tool} is mutating"}

        try:
            if "when" in stage and not _condition(stage["when"], outputs, parameters):
                outputs[sid] = None
                summaries.append({"id": sid, "tool": tool, "ok": True, "skipped": True, "size": 0})
                continue

            foreach_spec = stage.get("foreach")
            items = None
            if foreach_spec is not None:
                items = _resolve(foreach_spec, outputs, parameters)
                if not isinstance(items, list):
                    return {"ok": False, "error": f"stage {idx}: foreach must resolve to an array", "stages": summaries}
                items = items[:MAX_FOREACH_ITEMS]
            else:
                items = [None]

            stage_results = []
            stage_args: list[dict[str, Any]] = []
            stage_size = 0
            stage_ok = True
            stage_status = "ok"
            optional_failures = 0
            stage_grounding: list[dict[str, Any]] = []
            for item in items:
                invocations += 1
                if invocations > MAX_INVOCATIONS:
                    return {"ok": False, "error": f"pipeline exceeds {MAX_INVOCATIONS} tool invocations", "stages": summaries}
                args = _resolve(stage.get("args") or {}, outputs, parameters, item)
                args = normalize_arguments(AVAILABLE_TOOLS_MAP[tool], args)
                stage_args.append(dict(args))
                try:
                    result = execute_registered_tool(tool, args)
                except Exception as exc:
                    if stage.get("optional"):
                        stage_ok = False
                        stage_status = "error"
                        optional_failures += 1
                        parsed = {"ok": False, "error": str(exc), "optional_failure": True}
                    else:
                        return {"ok": False, "error": f"stage {idx} ({tool}) failed: {exc}", "stages": summaries}
                else:
                    parsed = _parse_result(result)
                    raw_for_status = (
                        result if isinstance(result, str)
                        else json.dumps(result, ensure_ascii=False, default=str)
                    )
                    outcome = classify_tool_outcome(raw_for_status, tool_name=tool)
                    invocation_status = str(outcome.get("status") or ("ok" if outcome.get("success") else "error"))
                    if invocation_status == "partial" and stage_status == "ok":
                        stage_status = "partial"
                    meta = grounding_metadata(tool, raw_for_status, arguments=args)
                    stage_grounding.append({
                        key: meta.get(key)
                        for key in ("fact_types", "target", "time_scope", "source_url", "market_instruments")
                        if meta.get(key) not in (None, "", [], {})
                    })
                    if not bool(outcome.get("success")):
                        if stage.get("optional"):
                            stage_ok = False
                            stage_status = "error"
                            optional_failures += 1
                            parsed = {
                                "ok": False,
                                "error": raw_for_status[:2000],
                                "reason": str(outcome.get("reason") or "tool_error"),
                                "optional_failure": True,
                            }
                        else:
                            return {
                                "ok": False,
                                "error": f"stage {idx} ({tool}) reported {outcome.get('reason') or 'error'}",
                                "result": raw_for_status[:2000],
                                "stages": summaries,
                            }
                serialized, size = _serialize_size(parsed)
                if size > MAX_INTERMEDIATE_CHARS:
                    return {"ok": False, "error": f"stage {idx} output exceeded bounded intermediate size", "stages": summaries}
                stage_size += size
                stage_results.append(parsed)

            outputs[sid] = stage_results if foreach_spec is not None else stage_results[0]
            summaries.append({
                "id": sid, "tool": tool, "ok": stage_ok, "status": stage_status,
                "calls": len(stage_results), "size": stage_size,
                "args": stage_args[0] if len(stage_args) == 1 else stage_args,
                "grounding": stage_grounding[0] if len(stage_grounding) == 1 else stage_grounding,
                **({"optional_failure": True, "failed_calls": optional_failures} if optional_failures else {}),
            })
        except Exception as exc:
            return {"ok": False, "error": f"stage {idx} ({tool}) resolution failed: {exc}", "stages": summaries}

    final_id = str(stages[-1].get("id") or f"s{len(stages)}")
    return {"ok": True, "stages": summaries, "invocations": invocations, "result": outputs[final_id]}


def run_pipeline(stages: list[dict[str, Any]], parameters: dict[str, Any] | None = None) -> str:
    """Execute up to sixteen read-only typed stages locally, including bounded foreach/conditional stages and prior-output references."""
    return json.dumps(execute_pipeline(stages, parameters), ensure_ascii=False, indent=2, default=str)


def run_recipe(name: str, parameters: dict[str, Any] | None = None) -> str:
    """Execute one saved semantic recipe by name using optional parameter overrides."""
    from .recipe_store import get_recipe, mark_recipe_used
    recipe = get_recipe(name)
    if not recipe:
        return "Error: recipe not found."
    merged = {}
    for key, spec in (recipe.get("parameters") or {}).items():
        if isinstance(spec, dict) and "default" in spec:
            merged[key] = spec["default"]
    merged.update(parameters or {})
    result = execute_pipeline(recipe["pipeline"], merged)
    mark_recipe_used(recipe["id"], bool(result.get("ok")))
    result["recipe"] = {"id": recipe["id"], "name": recipe["name"], "origin": recipe.get("origin", "user"), "target_tool": recipe.get("target_tool", "")}
    return json.dumps(result, ensure_ascii=False, indent=2, default=str)


def list_recipes_tool(limit: int = 50) -> str:
    """List saved reusable recipes from the dedicated recipe database."""
    from .recipe_store import list_recipes
    rows = list_recipes(limit)
    return json.dumps([
        {"id": r["id"], "name": r["name"], "description": r["description"], "tags": r["tags"],
         "origin": r.get("origin", "user"), "target_tool": r.get("target_tool", ""),
         "use_count": r["use_count"], "success_count": r["success_count"]}
        for r in rows
    ], ensure_ascii=False, indent=2)


def search_recipes_tool(query: str, limit: int = 8) -> str:
    """Semantically search saved recipes by objective, description, tags, and tool names."""
    from .recipe_store import search_recipes
    rows = search_recipes(query, limit)
    return json.dumps([
        {"id": r["id"], "name": r["name"], "description": r["description"], "tags": r["tags"],
         "origin": r.get("origin", "user"), "target_tool": r.get("target_tool", ""), "semantic_score": r["semantic_score"]}
        for r in rows
    ], ensure_ascii=False, indent=2)


def save_recipe_tool(name: str, description: str, stages: list[dict[str, Any]], parameters: dict[str, Any] | None = None, tags: list[str] | None = None) -> str:
    """Save an explicit reusable read-only recipe in the semantic recipe database."""
    from .recipe_store import save_recipe
    from . import AVAILABLE_TOOLS_MAP, TOOL_METADATA
    if not stages or len(stages) > MAX_STAGES:
        return f"Error: recipe must contain 1-{MAX_STAGES} stages."
    for stage in stages:
        tool = str(stage.get("tool") or "")
        if tool not in AVAILABLE_TOOLS_MAP or tool in _FORBIDDEN or not bool(TOOL_METADATA.get(tool, {}).get("readonly", True)):
            return f"Error: recipe contains unavailable, forbidden, or mutating tool: {tool}"
    recipe = save_recipe(name, description, stages, parameters, tags)
    return json.dumps({"saved": True, "id": recipe["id"], "name": recipe["name"]}, indent=2)


def list_recipes(limit: int = 50) -> str:
    """List saved reusable recipes from the dedicated recipe database."""
    return list_recipes_tool(limit)


def search_recipes(query: str, limit: int = 8) -> str:
    """Semantically search saved recipes by objective, description, tags, and tool names."""
    return search_recipes_tool(query, limit)


def save_recipe(name: str, description: str, stages: list[dict[str, Any]], parameters: dict[str, Any] | None = None, tags: list[str] | None = None) -> str:
    """Save an explicit reusable read-only recipe in the semantic recipe database."""
    return save_recipe_tool(name, description, stages, parameters, tags)
