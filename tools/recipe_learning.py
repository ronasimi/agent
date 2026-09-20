"""Successful-turn recipe candidate extraction and deterministic opt-in saving."""
from __future__ import annotations

import json
import re
from typing import Any

from .recipe_store import create_candidate, dismiss_pending_candidate, pending_candidate, recipe_exists_for_task, save_pending_candidate

_PARAMETER_KEYS = {"target","host","url","path","filename","query","name","network","service","record_type","record_types","resolver","port"}
_AFFIRM = {"yes","yes save it","save it","save recipe","save this recipe","sure","do it","yes please"}
_DECLINE = {"no","no thanks","don't save it","do not save it","skip","not now"}


def _parameterize_args(tool: str, args: dict[str, Any], objective: str, parameters: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    objective_lower = objective.lower()
    for key, value in (args or {}).items():
        should_parameterize = key in _PARAMETER_KEYS and isinstance(value,(str,int,float,bool,list))
        if isinstance(value,str) and value and value.lower() in objective_lower:
            should_parameterize = True
        if should_parameterize:
            base=re.sub(r"[^a-z0-9_]+","_",key.lower()).strip("_") or "value"
            pname=base
            suffix=2
            while pname in parameters and parameters[pname].get("default") != value:
                pname=f"{base}_{suffix}"; suffix+=1
            parameters[pname]={"default":value,"description":f"Value for {tool}.{key}"}
            out[key]={"$param":pname,"default":value}
        else:
            out[key]=value
    return out


def build_candidate(objective: str, trace: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Convert successful read-only tool calls into a reusable parameterized pipeline."""
    stages=[]; params={}; seen=set()
    for entry in trace:
        if not entry.get("success") or not entry.get("readonly",True):
            continue
        tool=str(entry.get("tool") or "")
        args=dict(entry.get("args") or {})
        signature=json.dumps([tool,args],sort_keys=True,default=str)
        if signature in seen: continue
        seen.add(signature)
        if tool in {"current_time","run_pipeline","run_recipe","search_recipes","list_recipes"}:
            continue
        stages.append({"id":f"s{len(stages)+1}","tool":tool,"args":_parameterize_args(tool,args,objective,params)})
        if len(stages)>=8:break
    return stages,params


def maybe_create_recipe_candidate(objective: str, trace: list[dict[str, Any]], min_stages: int = 2) -> dict[str, Any] | None:
    stages,params=build_candidate(objective,trace)
    if len(stages)<max(1,int(min_stages)):return None
    stage_tools = [str(stage.get("tool") or "") for stage in stages]
    # The harness already ships and versions this workflow as
    # ``weather.current_forecast``. Do not ask the user to save a duplicate just
    # because a model happened to issue its component primitives manually.
    if "geocode_location" in stage_tools and "weather_forecast" in stage_tools:
        return None
    if recipe_exists_for_task(objective,stages):return None
    tools=[s["tool"] for s in stages]
    cid=create_candidate(objective,stages,params,tags=tools[:8])
    return {"candidate_id":cid,"objective":objective,"stages":len(stages),"tools":tools,"parameters":params}


def pending_recipe_prompt() -> str:
    candidate=pending_candidate()
    if not candidate:return ""
    tools=" → ".join(str(s.get("tool") or "") for s in candidate["pipeline"])
    return f"This successful workflow does not match an existing saved recipe. Save it as a reusable recipe? ({tools})"


def handle_recipe_confirmation(text: str) -> tuple[bool,str]:
    """Handle a direct yes/no response to a harness recipe-save prompt."""
    if not pending_candidate():return False,""
    normalized=re.sub(r"\s+"," ",str(text).strip().lower())
    named = re.match(r"^(?:yes[, ]+)?save (?:this )?recipe as\s+(.+)$", normalized)
    if named:
        recipe=save_pending_candidate(name=named.group(1).strip()[:80])
        if not recipe:return True,"No pending recipe was available to save."
        return True,f"Saved recipe **{recipe['name']}** with {len(recipe['pipeline'])} stage(s)."
    if normalized in _AFFIRM or normalized.startswith("yes, save"):
        recipe=save_pending_candidate()
        if not recipe:return True,"No pending recipe was available to save."
        return True,f"Saved recipe **{recipe['name']}** with {len(recipe['pipeline'])} stage(s)."
    if normalized in _DECLINE:
        dismiss_pending_candidate(); return True,"Recipe not saved."
    # The suggestion applies only to the immediately following user response.
    # An unrelated request expires it so a later casual "yes" cannot save stale work.
    dismiss_pending_candidate()
    return False,""
