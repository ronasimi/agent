"""Harness-native bounded pipelines and stored recipe execution."""
from __future__ import annotations

import json
from typing import Any

MAX_STAGES=8
MAX_INTERMEDIATE_CHARS=200_000
_FORBIDDEN={"execute_shell","execute_python","run_pipeline","run_recipe","save_recipe","save_pending_recipe","create_or_update_tool","install_package"}


def _parse_result(value: Any) -> Any:
    if isinstance(value,(dict,list,int,float,bool)) or value is None:return value
    text=str(value)
    try:return json.loads(text)
    except Exception:return text


def _extract(value: Any, path: str) -> Any:
    if path in {"", ".", "$"}: return value
    cur=value
    for part in str(path).strip("$.").split("."):
        if not part:continue
        if isinstance(cur,list):cur=cur[int(part)]
        elif isinstance(cur,dict):cur=cur[part]
        else:raise KeyError(path)
    return cur


def _resolve(value: Any, outputs: dict[str, Any], parameters: dict[str, Any]) -> Any:
    if isinstance(value,dict):
        if "$ref" in value:
            data=outputs[str(value["$ref"])]
            return _extract(data,str(value.get("path") or "$"))
        if "$param" in value:
            key=str(value["$param"])
            if key in parameters:return parameters[key]
            if "default" in value:return value["default"]
            raise KeyError(f"missing recipe parameter: {key}")
        return {k:_resolve(v,outputs,parameters) for k,v in value.items()}
    if isinstance(value,list):return [_resolve(v,outputs,parameters) for v in value]
    return value


def execute_pipeline(stages: list[dict[str, Any]], parameters: dict[str, Any] | None = None) -> dict[str, Any]:
    from . import AVAILABLE_TOOLS_MAP, TOOL_METADATA
    from .tool_registry import normalize_arguments
    parameters=parameters or {}
    if not isinstance(stages,list) or not stages:return {"ok":False,"error":"pipeline requires at least one stage"}
    if len(stages)>MAX_STAGES:return {"ok":False,"error":f"pipeline exceeds {MAX_STAGES} stages"}
    outputs:dict[str,Any]={}; summaries=[]
    for idx,stage in enumerate(stages,1):
        if not isinstance(stage,dict):return {"ok":False,"error":f"stage {idx} must be an object"}
        tool=str(stage.get("tool") or ""); sid=str(stage.get("id") or f"s{idx}")
        if tool in _FORBIDDEN or tool not in AVAILABLE_TOOLS_MAP:return {"ok":False,"error":f"stage {idx}: tool unavailable or forbidden: {tool}"}
        meta=TOOL_METADATA.get(tool,{})
        if not bool(meta.get("readonly",True)):return {"ok":False,"error":f"stage {idx}: pipelines are read-only; {tool} is mutating"}
        try:
            args=_resolve(stage.get("args") or {},outputs,parameters)
            args=normalize_arguments(AVAILABLE_TOOLS_MAP[tool],args)
            result=AVAILABLE_TOOLS_MAP[tool](**args)
        except Exception as exc:return {"ok":False,"error":f"stage {idx} ({tool}) failed: {exc}","stages":summaries}
        parsed=_parse_result(result)
        serialized=json.dumps(parsed,default=str,ensure_ascii=False) if not isinstance(parsed,str) else parsed
        if len(serialized)>MAX_INTERMEDIATE_CHARS:return {"ok":False,"error":f"stage {idx} output exceeded bounded intermediate size","stages":summaries}
        if isinstance(result,str) and result.lstrip().lower().startswith(("error:","tool execution error:")):
            return {"ok":False,"error":f"stage {idx} ({tool}) reported error","result":result[:2000],"stages":summaries}
        outputs[sid]=parsed; summaries.append({"id":sid,"tool":tool,"ok":True,"size":len(serialized)})
    final_id=str(stages[-1].get("id") or f"s{len(stages)}")
    return {"ok":True,"stages":summaries,"result":outputs[final_id]}


def run_pipeline(stages: list[dict[str, Any]], parameters: dict[str, Any] | None = None) -> str:
    """Execute up to eight read-only typed tool stages locally; later stages may reference earlier output with {$ref:'s1', path:'field'} or {$param:'name'}."""
    return json.dumps(execute_pipeline(stages,parameters),ensure_ascii=False,indent=2,default=str)


def run_recipe(name: str, parameters: dict[str, Any] | None = None) -> str:
    """Execute one saved semantic recipe by name using optional parameter overrides."""
    from .recipe_store import get_recipe, mark_recipe_used
    recipe=get_recipe(name)
    if not recipe:return "Error: recipe not found."
    merged={}
    for key,spec in (recipe.get("parameters") or {}).items():
        if isinstance(spec,dict) and "default" in spec:merged[key]=spec["default"]
    merged.update(parameters or {})
    result=execute_pipeline(recipe["pipeline"],merged)
    mark_recipe_used(recipe["id"],bool(result.get("ok")))
    result["recipe"]={"id":recipe["id"],"name":recipe["name"]}
    return json.dumps(result,ensure_ascii=False,indent=2,default=str)


def list_recipes_tool(limit: int = 50) -> str:
    """List saved reusable recipes from the dedicated recipe database."""
    from .recipe_store import list_recipes
    rows=list_recipes(limit)
    return json.dumps([{"id":r["id"],"name":r["name"],"description":r["description"],"tags":r["tags"],"use_count":r["use_count"],"success_count":r["success_count"]} for r in rows],ensure_ascii=False,indent=2)


def search_recipes_tool(query: str, limit: int = 8) -> str:
    """Semantically search saved recipes by objective, description, tags, and tool names."""
    from .recipe_store import search_recipes
    rows=search_recipes(query,limit)
    return json.dumps([{"id":r["id"],"name":r["name"],"description":r["description"],"tags":r["tags"],"semantic_score":r["semantic_score"]} for r in rows],ensure_ascii=False,indent=2)


def save_recipe_tool(name: str, description: str, stages: list[dict[str, Any]], parameters: dict[str, Any] | None = None, tags: list[str] | None = None) -> str:
    """Save an explicit reusable read-only recipe in the semantic recipe database."""
    from .recipe_store import save_recipe
    # Validate before persisting.
    from . import AVAILABLE_TOOLS_MAP, TOOL_METADATA
    if not stages or len(stages)>MAX_STAGES:return "Error: recipe must contain 1-8 stages."
    for stage in stages:
        tool=str(stage.get("tool") or "")
        if tool not in AVAILABLE_TOOLS_MAP or tool in _FORBIDDEN or not bool(TOOL_METADATA.get(tool,{}).get("readonly",True)):
            return f"Error: recipe contains unavailable, forbidden, or mutating tool: {tool}"
    recipe=save_recipe(name,description,stages,parameters,tags)
    return json.dumps({"saved":True,"id":recipe["id"],"name":recipe["name"]},indent=2)


def list_recipes(limit: int = 50) -> str:
    """List saved reusable recipes from the dedicated recipe database."""
    return list_recipes_tool(limit)

def search_recipes(query: str, limit: int = 8) -> str:
    """Semantically search saved recipes by objective, description, tags, and tool names."""
    return search_recipes_tool(query, limit)

def save_recipe(name: str, description: str, stages: list[dict[str, Any]], parameters: dict[str, Any] | None = None, tags: list[str] | None = None) -> str:
    """Save an explicit reusable read-only recipe in the semantic recipe database."""
    return save_recipe_tool(name, description, stages, parameters, tags)
