"""Auto-discovered builtin compatibility recipes for monolithic tools."""
from __future__ import annotations

import importlib
import json
import pkgutil
from typing import Any

from . import recipe_provider_groups
from .recipe_store import save_builtin_recipe

_MONOLITH_MODULES = {
    "observation_tools", "host_tools", "host_diagnostics", "network_diagnostics",
    "network_mapper", "mdns_scanner", "web", "web_research", "web_screenshot",
    "media", "pdf_generator", "packages", "repo_map", "repo_diagnostics",
}

def expected_monolithic_tools() -> set[str]:
    from .providers import BUILTINS
    return {function for module, function in BUILTINS if module in _MONOLITH_MODULES}


def discover_recipe_specs() -> tuple[list[dict[str, Any]], dict[str, str]]:
    specs: list[dict[str, Any]] = []
    native_only: dict[str, str] = {}
    prefix = recipe_provider_groups.__name__ + "."
    for info in sorted(pkgutil.iter_modules(recipe_provider_groups.__path__), key=lambda item: item.name):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(prefix + info.name)
        for raw in getattr(module, "RECIPE_SPECS", ()):  # additive provider contract
            spec = dict(raw)
            required = {"key", "version", "name", "target_tool", "description", "pipeline"}
            missing = required - set(spec)
            if missing:
                raise RuntimeError(f"Recipe provider {module.__name__} missing fields: {sorted(missing)}")
            specs.append(spec)
        for tool, reason in dict(getattr(module, "NATIVE_ONLY", {})).items():
            if tool in native_only:
                raise RuntimeError(f"Duplicate native-only recipe coverage entry for {tool}")
            native_only[str(tool)] = str(reason)
    keys = [str(s["key"]) for s in specs]
    names = [str(s["name"]) for s in specs]
    targets = [str(s["target_tool"]) for s in specs if str(s.get("coverage") or "full") == "full"]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate builtin recipe key")
    if len(names) != len(set(names)):
        raise RuntimeError("Duplicate builtin recipe name")
    if len(targets) != len(set(targets)):
        raise RuntimeError("More than one full compatibility recipe targets the same monolithic tool")
    return specs, native_only


def seed_builtin_recipes() -> dict[str, Any]:
    """Idempotently seed versioned harness-owned compatibility recipes."""
    specs, native_only = discover_recipe_specs()
    saved = []
    errors = []
    for spec in specs:
        try:
            recipe = save_builtin_recipe(
                key=str(spec["key"]), version=int(spec["version"]), name=str(spec["name"]),
                target_tool=str(spec["target_tool"]), description=str(spec["description"]),
                pipeline=list(spec["pipeline"]), parameters=dict(spec.get("parameters") or {}), tags=list(spec.get("tags") or []),
            )
            saved.append(recipe["name"])
        except Exception as exc:
            errors.append({"key": spec.get("key"), "error": str(exc)})
    return {"seeded": len(saved), "recipes": saved, "native_only": len(native_only), "errors": errors}


def compatibility_coverage() -> dict[str, Any]:
    specs, native_only = discover_recipe_specs()
    rows = []
    covered_targets = set()
    for spec in sorted(specs, key=lambda s: (str(s["target_tool"]), str(s["name"]))):
        coverage = str(spec.get("coverage") or "full")
        rows.append({
            "tool": str(spec["target_tool"]), "status": "recipe" if coverage == "full" else "partial_recipe",
            "recipe": str(spec["name"]), "reason": str(spec["description"]),
        })
        if coverage == "full":
            covered_targets.add(str(spec["target_tool"]))
    for tool, reason in sorted(native_only.items()):
        rows.append({"tool": tool, "status": "native_only", "recipe": "", "reason": reason})
    classified = {r["tool"] for r in rows}
    expected = expected_monolithic_tools()
    unclassified = sorted(expected - classified)
    return {
        "full_recipe_count": len(covered_targets),
        "partial_recipe_count": sum(1 for r in rows if r["status"] == "partial_recipe"),
        "native_only_count": len(native_only),
        "expected_monolithic_count": len(expected),
        "classified_count": len(expected & classified),
        "complete": not unclassified,
        "unclassified": unclassified,
        "coverage": sorted(rows, key=lambda r: (r["status"], r["tool"])),
    }


def recipe_coverage() -> str:
    """Report which high-level monolithic tools have primitive compatibility recipes and why remaining tools stay native."""
    return json.dumps(compatibility_coverage(), ensure_ascii=False, indent=2)
